import argparse
import json
import os
import re
import signal
import sys
import threading
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures
import random
import time
try:
    from json_repair import repair_json
except ImportError:
    def repair_json(text: str) -> str:
        return text
from typing import Any, Dict, List, Tuple, Set, Optional

from scripts.common.llm import openai_client
from scripts.common.paths import project_path, resolve_path

random.seed(46689)

EVENT_PATH = project_path("event", "events_with_anchors.jsonl")
SAVE_PATH = project_path("event", "dialogue_with_anchors.jsonl")
MODEL = os.environ.get("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")

MAX_RETRIES = 8
MIN_USER_TURNS = 8
MIN_ASSISTANT_TURNS = 8
MAX_MESSAGE_CHARS = 100

# 给单次 OpenAI 调用一个相对短的整体超时，避免 worker 在网络流上无限阻塞
OPENAI_TIMEOUT = 60.0

LOG_PATH = project_path("event", "gen_dialog.log")

_log_lock = threading.Lock()
_log_file = None

# Ctrl+C 触发后置位的全局停止信号；worker 看到后会提前结束重试 / 跳出 stream
stop_event = threading.Event()
_interrupt_count = 0


def _interruptible_sleep(seconds: float, step: float = 0.2) -> bool:
    """可被 stop_event 提前打断的 sleep；中途被打断返回 True。"""
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return False
        if stop_event.is_set():
            return True
        time.sleep(min(step, remaining))


def _install_sigint_handler():
    """注册 SIGINT 处理器：第一次 Ctrl+C 优雅停止，再按一次直接 os._exit 强退。

    注意：handler 内部不能 raise KeyboardInterrupt——在 Windows 上 handler 可能在
    主线程任意位置被调度，raise 会抛在意想不到的地方；这里只置位 stop_event，
    由主循环在 wait(timeout=...) 返回后主动检查并退出。
    """
    def _handler(signum, frame):
        global _interrupt_count
        _interrupt_count += 1
        stop_event.set()
        if _interrupt_count == 1:
            try:
                tqdm.write(
                    "[!] 收到 Ctrl+C，正在取消未启动任务并尽快退出... "
                    "再按一次 Ctrl+C 立即强制退出。"
                )
            except Exception:
                pass
        else:
            try:
                tqdm.write("[!] 强制退出。")
            except Exception:
                pass
            os._exit(130)

    signal.signal(signal.SIGINT, _handler)


def _ensure_log_file():
    """懒加载打开日志文件；首次打开时写入一段运行分隔头。"""
    global _log_file
    if _log_file is None:
        log_dir = os.path.dirname(LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        _log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)  # 行缓冲
        header = (
            f"\n{'=' * 80}\n"
            f"# Run started at {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(pid={os.getpid()})\n"
            f"{'=' * 80}\n"
        )
        _log_file.write(header)


def _close_log_file():
    """在主流程结束时关闭日志文件句柄。"""
    global _log_file
    with _log_lock:
        if _log_file is not None:
            _log_file.write(
                f"{'=' * 80}\n"
                f"# Run finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 80}\n"
            )
            _log_file.close()
            _log_file = None


def _emit(line):
    """把一行文本同时输出到 tqdm 友好的终端 + 日志文件（线程安全）。"""
    with _log_lock:
        _ensure_log_file()
        tqdm.write(line)          # 不会打乱 tqdm 进度条
        _log_file.write(line + "\n")


def _log(level, e_id, msg, attempt=None):
    """统一日志工具：时间戳 + 级别 + 事件 ID（+ 重试次数）+ 消息。"""
    ts = time.strftime("%H:%M:%S")
    tag = f"[{e_id}]"
    if attempt is not None:
        tag = f"[{e_id}][尝试 {attempt}/{MAX_RETRIES}]"
    _emit(f"[{ts}] [{level:<5}] {tag} {msg}")


def _log_raw(line):
    """输出一行原始文本（用于分隔线等不需要格式化的行）。"""
    _emit(line)

prompt_dialog = '''根据给定的用户画像和场景描述，生成一段用户与AI助手之间的自然对话。

[要求]
1. 对话总轮数不少于16轮，其中用户至少8轮，助手至少8轮。
2. 对话应始终围绕给定的场景主题和显式偏好展开，不要扩展到其他可能偏好或话题。
3. 每一轮内容保持简洁，读起来像真实自然发生的交流（避免剧本化或过于正式），每条消息不超过80字。
4. 对话中只能体现显式偏好信息，严禁提及任何隐式偏好。
5. 不要让对话暗示用户有其他未提及的偏好或习惯。
6. 对话内容应为地道的中文。
7. 注意：**对话角色固定为用户和AI助手**：即使场景中涉及其他人物（如家人、朋友、同事等），这些人物只能作为话题在对话中被提及，绝不能成为对话的参与者。绝对不要出现user和场景中其他人物对话的现象；assistant角色只能是AI助手，不能扮演其他人物。
8. 如果本次显式偏好是人物或宠物（如 Relationship-* / BasicPets-* / Pets-*），对话中需要自然带出该人物/宠物与用户的关系，以及人物的职业/身份或宠物的基本特征；只需顺口提及，不要像档案介绍一样生硬堆砌。

[特别注意]
隐式偏好绝对不能在dialogue中直接被提及！无论是在user还是在assistant的话语中，都绝对不能提及隐式偏好！
隐式偏好只能在图片边缘和背景音中体现！

[输出格式]
严格按照以下JSON格式输出，不要添加任何额外内容：
[
    {{"role": "user", "content": "用户说的话"}},
    {{"role": "assistant", "content": "助手说的话"}},
    ...
]

[用户画像]
{profile_str}

[场景描述]
{scene_str}

[本次显式偏好详情]
{explicit_prefs_str}

[该人物全部隐式偏好（对话必须完全避开）]
{all_implicit_prefs_str}

[上一轮失败原因]
{retry_feedback}

'''


def _format_explicit_prefs(explicit_prefs: list) -> str:
    """将 explicit_preferences 列表渲染为 LLM 可读的详情文本。
    对于 Relationship / Pets 等包含人物/宠物外貌和职业信息的偏好，
    将 content 字段完整传递，以便对话中能准确引用。
    """
    if not explicit_prefs:
        return "（无显式偏好）"
    lines = []
    for p in explicit_prefs:
        cat     = p.get('category', '')
        subcat  = p.get('subcategory', '')
        content = p.get('content', '')
        sources = ", ".join(p.get('sources') or [])
        lines.append(f"- [{cat}] {subcat}（来源模态：{sources}）")
        if content:
            lines.append(f"  详细信息：{content}")
    return "\n".join(lines)


def _format_implicit_prefs_forbidden(implicit_prefs: list) -> str:
    """Render all implicit preferences as forbidden dialogue content."""
    if not implicit_prefs:
        return "（无隐式偏好）"
    lines = []
    seen = set()
    for p in implicit_prefs:
        if not isinstance(p, dict):
            continue
        cat = str(p.get('category', '') or '').strip()
        content = str(p.get('content', '') or '').strip()
        key = (cat, content)
        if key in seen:
            continue
        seen.add(key)

        subcat = str(p.get('subcategory', '') or '').strip()
        sources = ", ".join(str(s) for s in (p.get('sources') or []))
        lines.append(f"- [{cat}] {subcat}（来源模态：{sources}）")
        if content:
            lines.append(f"  禁止泄露内容：{content}")
        anchors = p.get('entity_anchors', p.get('entity_anchor', []))
        if isinstance(anchors, list) and anchors:
            lines.append("  禁止在对话中提及的实体锚点：" + "、".join(str(a) for a in anchors if str(a).strip()))
        elif isinstance(anchors, str) and anchors.strip():
            lines.append(f"  禁止在对话中提及的实体锚点：{anchors.strip()}")
    return "\n".join(lines) if lines else "（无隐式偏好）"


def _dialog_messages_to_text(dialog_messages: List[Dict[str, Any]]) -> str:
    lines = []
    for i, msg in enumerate(dialog_messages):
        role = "用户" if msg.get('role') == 'user' else "AI助手"
        content = str(msg.get('content', '') or '').strip()
        lines.append(f"{i}. {role}: {content}")
    return "\n".join(lines)


PROMPT_DIALOG_LEAKAGE_VALIDATOR = '''你是一个严格的隐式偏好泄露检查器。请判断下面这段“用户与AI助手的对话”是否泄露了该人物的任何隐式偏好。

[检查对象：对话]
{dialog_text}

[该人物全部隐式偏好]
{all_implicit_prefs_str}

[判定标准]
1. 如果对话直接提到、复述、改写、暗示或解释任何隐式偏好的物品、动作、习惯、环境线索、声音线索、entity anchor，都判为泄露。
2. 不只检查本次事件的隐式偏好，也要检查该人物所有隐式偏好。
3. 用户和AI助手任意一方泄露都算泄露。
4. 如果只是非常泛泛的生活表达，且无法对应到具体隐式偏好，不要误判。
5. 只输出 JSON，不要输出解释或 markdown。

[输出格式]
{{
  "is_valid": true,
  "leaked_preferences": [
    {{
      "category": "隐式偏好类别ID",
      "evidence": "对话中泄露的原句或短语",
      "reason": "为什么这句泄露了该隐式偏好"
    }}
  ],
  "summary": "简短结论"
}}
'''


def validate_dialogue_no_implicit_leakage(
    dialog_messages: List[Dict[str, Any]],
    all_implicit_prefs: list,
    e_id: str,
) -> Tuple[bool, List[str], int, int]:
    """Use an LLM validator to check whether dialogue leaks any implicit preference."""
    if not all_implicit_prefs:
        return True, [], 0, 0
    if stop_event.is_set():
        return False, ["收到停止信号，跳过隐式泄露检查"], 0, 0

    prompt = PROMPT_DIALOG_LEAKAGE_VALIDATOR.format(
        dialog_text=_dialog_messages_to_text(dialog_messages),
        all_implicit_prefs_str=_format_implicit_prefs_forbidden(all_implicit_prefs),
    )
    try:
        client = openai_client(timeout=OPENAI_TIMEOUT)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        text = raw.replace("```json", "").replace("```", "").strip()
        obj = json.loads(repair_json(text))
        is_valid = bool(obj.get("is_valid"))
        leaked = obj.get("leaked_preferences") or []
        issues: List[str] = []
        if not is_valid:
            if isinstance(leaked, list) and leaked:
                for item in leaked[:10]:
                    if not isinstance(item, dict):
                        continue
                    category = item.get("category", "")
                    evidence = item.get("evidence", "")
                    reason = item.get("reason", "")
                    issues.append(f"{category}: {evidence}；{reason}")
            if not issues:
                issues.append(str(obj.get("summary") or "LLM validator 判定 dialogue 泄露了隐式偏好"))

        usage = getattr(response, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0
        if is_valid:
            _log("OK", e_id, "隐式偏好泄露检查通过")
        else:
            _log("WARN", e_id, "隐式偏好泄露检查失败: " + " | ".join(issues[:3]))
        return is_valid, issues, pt, ct
    except Exception as exc:
        _log("WARN", e_id, f"隐式偏好泄露检查异常，按失败处理: {type(exc).__name__}: {exc}")
        return False, [f"隐式偏好泄露检查异常: {type(exc).__name__}: {exc}"], 0, 0


def get_event_str(event, explicit_prefs: list = None):
    """Build scene description string from event - ONLY explicit information"""
    parts = []
    parts.append(f"场景描述: {event.get('scene_description', '')}")

    # Only include image description if it exists (but don't reveal implicit details)
    img_desc = event.get('user_shared_image_description', 'none')
    if img_desc and img_desc != 'none':
        parts.append("用户分享的图片: 用户将分享一张图片，图片主体与显式偏好相关")

    # DO NOT include human_speech_content - TTS will be done later
    # DO NOT include background_audio_info (implicit audio)
    # DO NOT include implicit_preferences_reflected

    reflected_ids = event.get('explicit_preferences_reflected', [])
    parts.append(f"显式偏好ID: {reflected_ids}")

    return "\n".join(parts), _format_explicit_prefs(explicit_prefs or [])


def find_image_turn_by_llm(dialog_messages, e_id):
    """
    Use LLM to find the best user turn to insert image.
    Returns: turn index or -1 if not found
    """
    if not dialog_messages:
        return -1

    # Build dialog summary
    dialog_text = []
    for i, msg in enumerate(dialog_messages):
        role = "用户" if msg.get('role') == 'user' else "助手"
        dialog_text.append(f"{i}. {role}: {msg.get('content', '')}")

    prompt = f'''请分析以下对话，找出最合适插入图片的用户轮次。

[对话内容]
{chr(10).join(dialog_text)}

[要求]
1. 找出用户提到"照片"、"图片"、"给你看"、"看看"等暗示分享图片的轮次
2. 如果没有明确暗示，选择一个用户分享信息最丰富的轮次
3. 只能选择用户轮次（序号为偶数的轮次，如0, 2, 4...）
4. 仅输出一个数字，即选中的轮次序号

[输出]
仅输出一个数字，不要其他内容。
'''

    if stop_event.is_set():
        return -1

    try:
        client = openai_client(timeout=OPENAI_TIMEOUT)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        # Extract number
        import re
        match = re.search(r'\d+', result)
        if match:
            idx = int(match.group())
            # Validate it's a user turn
            if 0 <= idx < len(dialog_messages) and dialog_messages[idx].get('role') == 'user':
                _log("OK", e_id, f"图片插入轮次选定: 第 {idx} 条（用户轮次）")
                return idx
            else:
                _log("WARN", e_id,
                     f"LLM 返回的图片轮次索引非法: idx={idx}, "
                     f"对话长度={len(dialog_messages)}, "
                     f"该轮 role={dialog_messages[idx].get('role') if 0 <= idx < len(dialog_messages) else 'out-of-range'}")
        else:
            _log("WARN", e_id, f"LLM 图片轮次检测未返回数字，原始输出: {result[:100]!r}")
    except Exception as e:
        _log("ERROR", e_id, f"LLM 图片轮次检测异常 ({type(e).__name__}: {e})")

    return -1


def _user_turn_indices(dialog_messages):
    return [i for i, m in enumerate(dialog_messages) if m.get('role') == 'user']


# Maximum allowed length (in Chinese chars) of a single background_audio
# description. Anything more verbose drifts away from the "短促单一声音"
# requirement (e.g. "咖啡研磨机的声音"、"切菜的声音").
MAX_BG_AUDIO_LEN = 18

# How many user turns we ask the LLM to pick per session.
MIN_BG_AUDIO_TURNS = 3
MAX_BG_AUDIO_TURNS = 8


def _pref_sources(pref: Dict[str, Any]) -> List[str]:
    return [str(s).strip().lower() for s in (pref.get('sources') or pref.get('evidence_sources') or [])]


def _pref_analysis_lines(pref: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for key in ('analysis', 'rationale'):
        value = pref.get(key)
        if isinstance(value, list):
            lines.extend(str(x).strip() for x in value if str(x).strip())
        elif isinstance(value, str) and value.strip():
            lines.append(value.strip())
    return lines


def _extract_leading_audio_keyword(text: str) -> Optional[str]:
    """Extract the leading audio cue in strings like '(切菜声) ...'.

    Visual/dialogue rationale lines may also use leading parentheses, so this
    keeps only likely audio cue words and skips obvious non-audio markers.
    """
    match = re.match(r'^\s*[（(]\s*([^）)]+?)\s*[）)]', str(text or ''))
    if not match:
        return None
    keyword = match.group(1).strip()
    if not keyword:
        return None
    lowered = keyword.lower()
    non_audio_markers = {'视觉', 'visual', 'dialogue', '对话', '文本', '文字'}
    if lowered in non_audio_markers or any(marker in keyword for marker in ['视觉', '对话', '文字']):
        return None
    audio_markers = [
        '声', '音', '响', '音乐', '掌声', '铃', '键盘', '脚步', '敲击',
        '水流', '风铃', '鸟鸣', '雨', '机器', '发动', '沸腾', '切菜',
    ]
    if any(marker in keyword for marker in audio_markers):
        return keyword
    return None


def extract_required_audio_keywords(implicit_prefs: list) -> List[str]:
    """Return unique required audio keywords from leading analysis brackets."""
    keywords: List[str] = []
    seen: Set[str] = set()
    for pref in implicit_prefs or []:
        if not isinstance(pref, dict):
            continue
        if 'audio' not in _pref_sources(pref):
            continue
        for line in _pref_analysis_lines(pref):
            keyword = _extract_leading_audio_keyword(line)
            if keyword and keyword not in seen:
                seen.add(keyword)
                keywords.append(keyword)
    return keywords


def _format_implicit_audio_prefs(implicit_prefs):
    """Render the implicit preferences whose sources include 'audio'
    into bullet lines that the LLM can ground its background_audio
    choices on (a clue together with its analysis/rationale snippets)."""
    lines = []
    for p in implicit_prefs or []:
        if not isinstance(p, dict):
            continue
        if 'audio' not in _pref_sources(p):
            continue
        cat = p.get('category', '')
        content = p.get('content', '')
        lines.append(f"- [{cat}] {content}")
        for r in _pref_analysis_lines(p):
            lines.append(f"    证据: {r}")
    return "\n".join(lines) if lines else "（无含 audio 模态的隐式偏好）"


def validate_background_audio_keyword_coverage(
    dialog_messages: List[Dict[str, Any]],
    required_keywords: List[str],
    e_id: str,
) -> Tuple[bool, List[str]]:
    """Check required analysis keywords are present in written background_audio."""
    if not required_keywords:
        return True, []
    written = [
        str(m.get('background_audio', '') or '').strip()
        for m in dialog_messages
        if isinstance(m, dict) and str(m.get('background_audio', '') or '').strip()
    ]
    joined = "\n".join(written)
    missing = [keyword for keyword in required_keywords if keyword not in joined]
    if missing:
        _log(
            "WARN",
            e_id,
            "background_audio 缺少 analysis 音频关键词: " + "、".join(missing),
        )
        return False, missing
    _log("OK", e_id, "background_audio analysis 音频关键词覆盖通过: " + "、".join(required_keywords))
    return True, []


def assign_background_audio_by_llm(
    dialog_messages,
    background_audio_info,
    scene_description,
    implicit_prefs,
    e_id,
):
    """使用第二次 LLM 处理来决定哪些 **用户回合** 应带有
    ``background_audio`` 标签，以及每个标签应具体显示什么内容。

    返回一个字典列表 ``[{“turn_index”: int, “background_audio”: str}, ...]``。
    若处理失败，或输入内容不支持添加背景音频，则返回
    一个空列表，调用方应保持对话内容不变。
    """
    if not dialog_messages:
        return []
    if not background_audio_info or str(background_audio_info).strip().lower() == 'none':
        return []

    user_turns = _user_turn_indices(dialog_messages)
    if not user_turns:
        return []

    dialog_text_lines = []
    for i, msg in enumerate(dialog_messages):
        role = "用户" if msg.get('role') == 'user' else "助手"
        dialog_text_lines.append(f"[{i}] {role}: {msg.get('content', '')}")
    dialog_text = "\n".join(dialog_text_lines)

    implicit_audio_str = _format_implicit_audio_prefs(implicit_prefs)
    required_audio_keywords = extract_required_audio_keywords(implicit_prefs)
    required_audio_keywords_str = (
        "\n".join(f"- {kw}" for kw in required_audio_keywords)
        if required_audio_keywords else "（无必须逐字覆盖的 analysis 音频关键词）"
    )
    user_turn_str = ", ".join(str(i) for i in user_turns)

    if stop_event.is_set():
        return []

    retry_feedback = "无"
    last_results: List[Dict[str, Any]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        prompt = f"""请根据下面给出的对话、场景、隐式偏好和背景音概述，判断在哪几个**用户轮次**中插入 `background_audio` 标记最自然，并给出每条 background_audio 的具体内容。

[场景描述]
{scene_description or "（无）"}

[本事件的背景音概述（仅供你理解整体声学环境，禁止把它一字不漏地塞进 background_audio 字段）]
{background_audio_info}

[与音频相关的隐式偏好（必须从中提炼 background_audio 的含义；证据行来自 analysis/rationale）]
{implicit_audio_str}

[必须覆盖的 analysis 音频关键词]
{required_audio_keywords_str}

[完整对话（每行带轮次序号）]
{dialog_text}

[上一轮失败原因]
{retry_feedback}

[挑选规则]
1. 只能挑选 **role=用户** 的轮次，可选轮次集合（必须从中选）：[{user_turn_str}]。
2. 总共挑选 **{MIN_BG_AUDIO_TURNS} 到 {MAX_BG_AUDIO_TURNS}** 个轮次。
3. 每条 background_audio 必须满足：
   a. **简洁的描述**（≤{MAX_BG_AUDIO_LEN} 个汉字），形式如 "咖啡研磨声"、"切菜声"、"沸腾声"、"弹钢琴声"，禁止形容词堆砌或场景渲染；
   b. **只能描述一种声音**，不能用 "和"/"加上"/"&" 把多种声音并列；
   c. 必须**合理体现**[与音频相关的隐式偏好]，或属于[本事件的背景音概述]里出现的同类声音；不要出现与隐式偏好或显式对话主题冲突的声音。
4. 如果[必须覆盖的 analysis 音频关键词]非空，则每个关键词都必须在至少一条 background_audio 中**逐字出现**，例如关键词是“切菜声”，background_audio 就要包含“切菜声”，不要改写成“切菜的声音”。
5. 当所有必须关键词都出现以后，可以再自行生成其他符合 background_audio_info 的背景音。
6. **每种声音可以连续出现在 2-3 个相邻用户轮次上**（即同一个 background_audio 值在紧邻的用户轮次中重复标注），以充分体现对应的隐式偏好；
7. 不同种类的声音之间可以切换，但切换前后要保持基本的现实合理性（例如刚说"我去做饭"的下一秒不应该出现"打鼾声"）。
8. 不要选当前用户轮次内容明显与该背景音矛盾的位置（例如用户当前在谈"安静的午后"，就不要让那条加上"咖啡研磨声"）。
[输出格式]
**严格输出**如下 JSON 列表（按 turn_index 升序），不要附加任何解释文字：
```json
[
    {{"turn_index": <整数>, "background_audio": "<≤{MAX_BG_AUDIO_LEN}字的单一声音描述>"}},
    ...
]
```
"""

        if stop_event.is_set():
            return last_results

        try:
            client = openai_client(timeout=OPENAI_TIMEOUT)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.4,
            )
            raw = (response.choices[0].message.content or "").strip()
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(repair_json(cleaned))
        except Exception as e:
            _log("ERROR", e_id,
                 f"background_audio LLM 调用/解析异常 ({type(e).__name__}: {e})",
                 attempt=attempt)
            continue

        if not isinstance(parsed, list):
            _log("WARN", e_id,
                 f"background_audio LLM 输出非 list (type={type(parsed).__name__})",
                 attempt=attempt)
            retry_feedback = "上一轮输出不是 JSON list。请严格输出 JSON list。"
            continue

        user_turn_set = set(user_turns)
        seen_idx = set()
        results = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get('turn_index')
            text = item.get('background_audio')
            if not isinstance(idx, int) or idx in seen_idx or idx not in user_turn_set:
                continue
            if not isinstance(text, str):
                continue
            text = text.strip().strip('"').strip("'")
            if not text:
                continue
            if any(sep in text for sep in ['、', ',', '，', '+', '加', '和', '&']):
                _log("WARN", e_id,
                     f"background_audio 含多种声音被丢弃: idx={idx}, text={text!r}",
                     attempt=attempt)
                continue
            if len(text) > MAX_BG_AUDIO_LEN:
                text = text[:MAX_BG_AUDIO_LEN]
            seen_idx.add(idx)
            results.append({"turn_index": idx, "background_audio": text})

        results.sort(key=lambda x: x['turn_index'])

        if len(results) > MAX_BG_AUDIO_TURNS:
            results = results[:MAX_BG_AUDIO_TURNS]

        if len(results) < MIN_BG_AUDIO_TURNS:
            _log("WARN", e_id,
                 f"background_audio 命中数量过少 ({len(results)} < {MIN_BG_AUDIO_TURNS})，"
                 f"原始 LLM 输出={parsed}",
                 attempt=attempt)

        last_results = results
        assigned_text = "\n".join(str(item.get("background_audio", "")) for item in results)
        missing = [kw for kw in required_audio_keywords if kw not in assigned_text]
        if missing:
            retry_feedback = (
                "上一轮 background_audio 没有逐字覆盖这些 analysis 音频关键词："
                + "、".join(missing)
                + "。请在输出的 background_audio 字段中逐字写出这些关键词。"
            )
            _log("WARN", e_id, retry_feedback, attempt=attempt)
            continue

        return results

    if required_audio_keywords:
        _log(
            "WARN",
            e_id,
            "background_audio 多次重试后仍可能缺少关键词，将返回最后一次可解析结果: "
            + "、".join(required_audio_keywords),
        )
    return last_results


def attach_background_audio(dialog_messages, assignments, e_id):
    """Mutate ``dialog_messages`` in-place: write the chosen
    ``background_audio`` strings into the corresponding turns. Returns the
    list of (turn_index, background_audio) actually written, sorted."""
    written = []
    for item in assignments:
        idx = item.get('turn_index')
        text = item.get('background_audio')
        if not isinstance(idx, int) or not isinstance(text, str):
            continue
        if not (0 <= idx < len(dialog_messages)):
            continue
        msg = dialog_messages[idx]
        if not isinstance(msg, dict):
            continue
        if msg.get('role') != 'user':
            continue
        msg['background_audio'] = text
        written.append((idx, text))
    written.sort(key=lambda t: t[0])
    if written:
        _log("OK", e_id,
             "background_audio 已嵌入轮次: " +
             "; ".join(f"[{i}]={t}" for i, t in written))
    return written


def session_dialog(
    e_id,
    profile_str,
    event,
    explicit_prefs: list = None,
    all_implicit_prefs: list = None,
):
    _log("INFO", e_id, "开始生成对话任务")

    # prompt 内容在多次重试之间不变，提到循环外构造一次即可，
    # 这样无论成功/失败分支都能把同一份 prompt 作为返回值带回主流程，便于事后回溯。
    event_str, explicit_prefs_str = get_event_str(event, explicit_prefs)
    all_implicit_prefs = all_implicit_prefs or []
    retry_feedback = "无"
    prompt = prompt_dialog.format(
        profile_str=profile_str,
        scene_str=event_str,
        explicit_prefs_str=explicit_prefs_str,
        all_implicit_prefs_str=_format_implicit_prefs_forbidden(all_implicit_prefs),
        retry_feedback=retry_feedback,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        if stop_event.is_set():
            _log("WARN", e_id, "收到停止信号，放弃后续重试", attempt=attempt)
            return e_id, [], 0, 0, prompt

        usage_info = None
        prompt_tokens = 0
        completion_tokens = 0
        prompt = prompt_dialog.format(
            profile_str=profile_str,
            scene_str=event_str,
            explicit_prefs_str=explicit_prefs_str,
            all_implicit_prefs_str=_format_implicit_prefs_forbidden(all_implicit_prefs),
            retry_feedback=retry_feedback,
        )
        client = openai_client(timeout=OPENAI_TIMEOUT)

        try:
            _log("INFO", e_id, f"调用 LLM 生成对话 (model={MODEL}, prompt={len(prompt)} 字符)",
                 attempt=attempt)
            t_start = time.time()
            apiRes = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.8,
                top_p=0.9,
                stream=True,
                stream_options={"include_usage": True},
            )
            result = ""
            interrupted = False
            for chunk in apiRes:
                # 流式读取里穿插检查停止信号，确保 Ctrl+C 后 worker 能尽快退出
                if stop_event.is_set():
                    interrupted = True
                    try:
                        apiRes.close()
                    except Exception:
                        pass
                    break

                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content

                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            if interrupted:
                _log("WARN", e_id, "stream 读取过程中收到停止信号，放弃本次", attempt=attempt)
                return e_id, [], 0, 0, prompt

            response = result.strip().replace("```json", "").replace("```", "").strip()
            elapsed = time.time() - t_start
            _log("DEBUG", e_id,
                 f"LLM 返回完毕: 耗时 {elapsed:.1f}s, 返回 {len(response)} 字符",
                 attempt=attempt)

            # Parse JSON format
            try:
                dialog_messages = json.loads(repair_json(response))
            except Exception as parse_err:
                snippet = response[:200].replace("\n", " ")
                _log("WARN", e_id,
                     f"JSON 解析失败 ({type(parse_err).__name__}: {parse_err}); "
                     f"响应前 200 字符: {snippet!r}",
                     attempt=attempt)
                continue

            # Validate format: must be list of dicts with role and content
            if not isinstance(dialog_messages, list) or len(dialog_messages) == 0:
                type_name = type(dialog_messages).__name__
                length = len(dialog_messages) if hasattr(dialog_messages, '__len__') else 'N/A'
                _log("WARN", e_id,
                     f"格式非法: 顶层非列表或为空 (type={type_name}, len={length})",
                     attempt=attempt)
                continue

            valid = True
            invalid_reason = ""
            user_count = 0
            assistant_count = 0
            overlong_messages: List[str] = []
            for idx, msg in enumerate(dialog_messages):
                if not isinstance(msg, dict):
                    valid = False
                    invalid_reason = f"第 {idx} 条不是 dict (type={type(msg).__name__})"
                    break
                if msg.get('role') not in ['user', 'assistant']:
                    valid = False
                    invalid_reason = f"第 {idx} 条 role 非法 (role={msg.get('role')!r})"
                    break
                if 'content' not in msg:
                    valid = False
                    invalid_reason = f"第 {idx} 条缺少 content 字段 (keys={list(msg.keys())})"
                    break
                content_text = str(msg.get('content', '') or '').strip()
                content_len = len(content_text)
                if content_len > MAX_MESSAGE_CHARS:
                    role_label = "用户" if msg.get('role') == 'user' else "助手"
                    overlong_messages.append(
                        f"第 {idx} 条（{role_label}）{content_len} 字，超过 {MAX_MESSAGE_CHARS} 字："
                        f"{content_text[:60]}"
                    )
                if msg['role'] == 'user':
                    user_count += 1
                else:
                    assistant_count += 1

            if not valid:
                _log("WARN", e_id,
                     f"消息格式非法 -> {invalid_reason} (总 {len(dialog_messages)} 条)",
                     attempt=attempt)
                continue

            if overlong_messages:
                retry_feedback = (
                    f"上一轮 dialogue 有消息超过 {MAX_MESSAGE_CHARS} 字。"
                    f"请重写整段对话，确保每一条 user/assistant 消息都不超过 {MAX_MESSAGE_CHARS} 个中文字符；"
                    "可以拆短句、减少解释，但不要减少总轮次要求。\n"
                    + "\n".join(f"- {issue}" for issue in overlong_messages[:12])
                )
                _log(
                    "WARN",
                    e_id,
                    f"消息长度超限: {len(overlong_messages)} 条超过 {MAX_MESSAGE_CHARS} 字",
                    attempt=attempt,
                )
                continue

            # Check minimum turns
            if user_count < MIN_USER_TURNS or assistant_count < MIN_ASSISTANT_TURNS:
                _log("WARN", e_id,
                     f"轮次不足: 用户 {user_count}/{MIN_USER_TURNS}, "
                     f"助手 {assistant_count}/{MIN_ASSISTANT_TURNS} "
                     f"(总 {len(dialog_messages)} 条)",
                     attempt=attempt)
                continue

            if usage_info is not None:
                prompt_tokens += usage_info.prompt_tokens
                completion_tokens += usage_info.completion_tokens

            leakage_ok, leakage_issues, v_pt, v_ct = validate_dialogue_no_implicit_leakage(
                dialog_messages,
                all_implicit_prefs,
                e_id,
            )
            prompt_tokens += v_pt
            completion_tokens += v_ct
            if not leakage_ok:
                retry_feedback = (
                    "上一轮 dialogue 被判定泄露了隐式偏好。请重写整段对话，"
                    "保留显式偏好和场景主题，但完全删除/改写以下泄露点：\n"
                    + "\n".join(f"- {issue}" for issue in leakage_issues[:12])
                )
                _log("WARN", e_id, "因隐式偏好泄露而重试生成对话", attempt=attempt)
                continue

            _log("OK", e_id,
                 f"对话生成成功: 用户 {user_count} 轮 / 助手 {assistant_count} 轮, "
                 f"共 {len(dialog_messages)} 条; "
                 f"tokens prompt={prompt_tokens}, completion={completion_tokens}; "
                 f"耗时 {elapsed:.1f}s",
                 attempt=attempt)
            return e_id, dialog_messages, prompt_tokens, completion_tokens, prompt

        except Exception as e:
            if stop_event.is_set():
                _log("WARN", e_id,
                     f"API 调用异常 ({type(e).__name__}: {e}) 且收到停止信号，放弃重试",
                     attempt=attempt)
                return e_id, [], 0, 0, prompt
            _log("ERROR", e_id,
                 f"API 调用异常 ({type(e).__name__}: {e})；5 秒后重试",
                 attempt=attempt)
            if _interruptible_sleep(5):
                _log("WARN", e_id, "重试等待期间收到停止信号，放弃后续尝试", attempt=attempt)
                return e_id, [], 0, 0, prompt

    _log("FAIL", e_id, f"连续 {MAX_RETRIES} 次尝试均失败，放弃生成")
    return e_id, [], 0, 0, prompt


def process_event(t_id, profile_str, event, implicit_prefs,
                  existing_dialog: List[Dict] = None,
                  explicit_prefs: list = None,
                  all_implicit_prefs: list = None):
    """完整处理一个事件：对话生成 → 并发图片/背景音标注。

    若传入 existing_dialog，则跳过 Phase 1（对话生成），直接用已有对话跑 Phase 2。
    返回一个 dict，包含所有需要回填到 events_data 的字段，
    主线程只需原样写入，不再做任何 LLM 调用。
    """
    # ── Phase 1: 对话生成（续跑时可跳过）────────────────────────────────────
    if existing_dialog is not None:
        _log("INFO", t_id, "Phase 1 已完成（使用已保存对话），直接进入 Phase 2")
        dialog          = existing_dialog
        prompt_tokens   = 0
        completion_tokens = 0
        dl_prompt       = event.get("dialog_prompt", "")
        e_id            = t_id
    else:
        e_id, dialog, prompt_tokens, completion_tokens, dl_prompt = session_dialog(
            t_id, profile_str, event, explicit_prefs, all_implicit_prefs
        )

    out = {
        'e_id': e_id,
        'dialog': dialog,
        'dl_prompt': dl_prompt,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'image_turn_indices': [],
        'tts_turn_indices': [],
        'phase2_complete': False,
    }

    if not dialog:
        return out

    has_image = event.get('user_shared_image_description', 'none') != 'none'
    has_implicit_audio = event.get('background_audio_info', 'none') != 'none'
    has_explicit_audio = event.get('human_speech_content', 'none') != 'none'

    # explicit voice → tts_turn_indices（无需 LLM，直接确定）
    if has_explicit_audio:
        first_user_idx = next(
            (i for i, m in enumerate(dialog) if m.get('role') == 'user'), -1
        )
        if first_user_idx >= 0:
            out['tts_turn_indices'] = [first_user_idx]
            _log("INFO", e_id,
                 f"显式人声 TTS 轮次锚定到首个用户轮次: tts_turn_indices=[{first_user_idx}]")
        else:
            _log("WARN", e_id, "存在 human_speech_content 但对话内没有用户轮次，TTS 跳过")

    # ── Phase 2: 图片轮次 & 背景音标注并发执行 ───────────────────────────────
    img_future = None
    bg_future = None
    bg_audio_valid = True

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pp:
        if has_image and not stop_event.is_set():
            _log("INFO", e_id, "检测到图片，开始识别最佳插入轮次...")
            img_future = pp.submit(find_image_turn_by_llm, dialog, e_id)

        if has_implicit_audio and not stop_event.is_set():
            _log("INFO", e_id, "存在 background_audio_info，调用 LLM 决定背景音轮次...")
            bg_future = pp.submit(
                assign_background_audio_by_llm,
                dialog,
                event.get('background_audio_info', ''),
                event.get('scene_description', ''),
                implicit_prefs,
                e_id,
            )
    # with 块退出时会等待两个子任务都完成，之后再读取结果

    if img_future is not None:
        try:
            image_idx = img_future.result()
            out['image_turn_indices'] = [image_idx] if image_idx >= 0 else []
        except Exception as exc:
            _log("ERROR", e_id, f"图片轮次检测异常: {exc}")

    if bg_future is not None:
        try:
            bg_assignments = bg_future.result()
            attach_background_audio(dialog, bg_assignments, e_id)
            bg_audio_valid, _ = validate_background_audio_keyword_coverage(
                dialog,
                extract_required_audio_keywords(implicit_prefs),
                e_id,
            )
        except Exception as exc:
            _log("ERROR", e_id, f"背景音标注异常: {exc}")
            bg_audio_valid = False

    out['phase2_complete'] = bool(bg_audio_valid)

    _log("OK", e_id,
         f"事件处理完成: image_turns={out['image_turn_indices']}, "
         f"tts_turns={out['tts_turn_indices']}, "
         f"bg_audio_turns={[i for i, m in enumerate(dialog) if isinstance(m, dict) and 'background_audio' in m]}, "
         f"phase2_complete={out['phase2_complete']}")

    return out


def _flush_results(events_data, save_path):
    """把当前内存中的 events_data 完整写出到 save_path（带容错）。"""
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(events_data, f, indent=4, ensure_ascii=False)
        _log("INFO", "MAIN", f"已落盘当前进度到 {save_path}")
    except Exception as exc:
        _log("ERROR", "MAIN", f"落盘失败 ({type(exc).__name__}: {exc})")


def build_all_implicit_prefs_by_profile(events_data: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """Collect every implicit preference seen for each profile across all event records."""
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    seen: Dict[Any, Set[Tuple[str, str]]] = defaultdict(set)
    for item in events_data:
        p_id = item.get("p_id")
        if p_id is None:
            continue
        for pref in item.get("implicit_preferences", []) or []:
            if not isinstance(pref, dict):
                continue
            category = str(pref.get("category", "") or "").strip()
            content = str(pref.get("content", "") or "").strip()
            key = (category, content)
            if not category and not content:
                continue
            if key in seen[p_id]:
                continue
            seen[p_id].add(key)
            grouped[p_id].append(pref)
    return grouped


def _parse_args():
    parser = argparse.ArgumentParser(description="对话生成脚本")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate mode: only re-generate specified task IDs (Phase 1 + Phase 2)",
    )
    parser.add_argument(
        "--task-ids", nargs="+", metavar="TASK_ID",
        help="Task IDs to regenerate (used with --regenerate). "
             "If omitted, you will be prompted to enter them interactively.",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=0,
        help="只处理前 N 个 profile，用于小批量测试；0 表示处理全部 profile。",
    )
    parser.add_argument(
        "--only_profile_ids",
        nargs="*",
        default=None,
        help="只处理指定 p_id，支持逗号或空格分隔，例如：--only_profile_ids 5,6,7 或 --only_profile_ids 5 6 7。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="兼容参数：若输出文件已存在则沿用已有断点续跑逻辑，跳过已完成任务。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="并发 worker 数；默认 50。",
    )
    parser.add_argument("--input", default=str(EVENT_PATH), help="event JSON/JSONL input")
    parser.add_argument("--output", default=str(SAVE_PATH), help="dialogue output")
    parser.add_argument("--log", default=str(LOG_PATH), help="run log path")
    parser.add_argument("--model", default=MODEL, help="LLM model name")
    return parser.parse_args()


def _parse_profile_ids(raw_values: Optional[List[str]]) -> Optional[Set[int]]:
    if raw_values is None:
        return None
    ids: Set[int] = set()
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            ids.add(int(part))
    return ids


if __name__ == "__main__":
    _install_sigint_handler()
    args = _parse_args()

    EVENT_PATH = resolve_path(args.input)
    SAVE_PATH = resolve_path(args.output)
    LOG_PATH = resolve_path(args.log)
    MODEL = args.model

    t_overall_start = time.time()
    _log("INFO", "MAIN", f"日志文件: {LOG_PATH}")
    _log("INFO", "MAIN", f"读取事件文件: {EVENT_PATH}")
    with open(EVENT_PATH, "r") as f:
        events_data = json.load(f)

    _log("INFO", "MAIN", f"已加载 {len(events_data)} 个事件组")
    if args.max_profiles < 0:
        raise ValueError("--max-profiles must be >= 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")

    all_profile_ids = []
    seen_profile_ids = set()
    for item in events_data:
        p_id = item.get("p_id")
        if p_id is None or p_id in seen_profile_ids:
            continue
        seen_profile_ids.add(p_id)
        all_profile_ids.append(p_id)

    only_profile_ids = _parse_profile_ids(args.only_profile_ids)
    if only_profile_ids is not None:
        selected_profile_ids = only_profile_ids
        missing_profile_ids = sorted(pid for pid in selected_profile_ids if pid not in set(all_profile_ids))
        _log(
            "INFO",
            "MAIN",
            f"Profile 选择: 只处理指定 p_id: {sorted(selected_profile_ids)}",
        )
        if missing_profile_ids:
            _log("WARN", "MAIN", f"指定的 p_id 不存在于事件文件中: {missing_profile_ids}")
    elif args.max_profiles > 0:
        selected_profile_ids = set(all_profile_ids[:args.max_profiles])
        _log(
            "INFO",
            "MAIN",
            f"小批量测试模式: 只处理前 {args.max_profiles} 个 profile: {sorted(selected_profile_ids)}",
        )
    else:
        selected_profile_ids = set(all_profile_ids)
        _log("INFO", "MAIN", "Profile 选择: 全量处理")

    def _is_selected_profile(item: Dict[str, Any]) -> bool:
        return item.get("p_id") in selected_profile_ids

    all_implicit_prefs_by_pid = build_all_implicit_prefs_by_profile(events_data)
    _log(
        "INFO",
        "MAIN",
        "已按人物汇总全部隐式偏好: "
        + ", ".join(
            f"p_id={pid}: {len(prefs)}"
            for pid, prefs in sorted(all_implicit_prefs_by_pid.items(), key=lambda x: x[0])
        )
    )

    # ── 断点续跑：从已有输出文件中恢复已完成的对话 ──────────────────────────────
    # done_task_ids   : Phase 1 + Phase 2 均已完成，完全跳过
    # phase2only_dialogs : Phase 1 完成但 Phase 2 未完成，只重跑 Phase 2
    done_task_ids: Set[str] = set()
    phase2only_dialogs: Dict[str, List[Dict]] = {}   # task_id → existing dialog

    # ── Regenerate mode: collect target task IDs before loading saved state ──
    regen_task_ids: Set[str] = set()
    if args.regenerate:
        if args.task_ids:
            regen_task_ids = set(args.task_ids)
        else:
            print("请输入要重新生成的 task_id，多个用空格或逗号分隔，输入完毕后按 Enter：")
            raw_input = input().strip()
            # support both space- and comma-separated IDs
            regen_task_ids = {t.strip() for t in raw_input.replace(",", " ").split() if t.strip()}
        if not regen_task_ids:
            print("未输入任何 task_id，退出。")
            sys.exit(1)
        _log("INFO", "MAIN",
             f"Regenerate mode 已激活，目标 task_ids ({len(regen_task_ids)} 个): "
             f"{sorted(regen_task_ids)}")

    if os.path.exists(SAVE_PATH):
        try:
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            # 建立 task_id → saved_result 映射
            saved_map: Dict[str, Any] = {}
            for item in saved_data:
                tid = item.get("task_id", "")
                if tid and item.get("event", {}).get("dialog"):
                    saved_map[tid] = item
            # 把已有结果回填到 events_data，并按完成程度分类
            restored_full = 0
            restored_p2 = 0
            for item in events_data:
                if not _is_selected_profile(item):
                    continue
                tid = item.get("task_id", "")
                if tid not in saved_map:
                    continue
                saved_event = saved_map[tid].get("event", {})
                # 无论如何先把已有字段写回，防止覆盖
                item["event"]["dialog"]             = saved_event.get("dialog", [])
                item["event"]["dialog_prompt"]      = saved_event.get("dialog_prompt", "")
                item["event"]["image_turn_indices"] = saved_event.get("image_turn_indices", [])
                item["event"]["tts_turn_indices"]   = saved_event.get("tts_turn_indices", [])
                if args.regenerate and tid in regen_task_ids:
                    # Force re-generation: do NOT mark as done even if previously complete.
                    # Clear stale fields so old data doesn't linger on failure.
                    item["event"]["dialog"]             = []
                    item["event"]["dialog_prompt"]      = ""
                    item["event"]["image_turn_indices"] = []
                    item["event"]["tts_turn_indices"]   = []
                    item["phase2_complete"]             = False
                elif saved_map[tid].get("phase2_complete"):
                    # Phase 1 + Phase 2 均完成，完全跳过
                    done_task_ids.add(tid)
                    restored_full += 1
                else:
                    # Phase 1 完成，Phase 2 需要重跑
                    phase2only_dialogs[tid] = saved_event.get("dialog", [])
                    restored_p2 += 1
            _log("INFO", "MAIN",
                 f"从 {SAVE_PATH} 恢复: "
                 f"{restored_full} 条完全完成（跳过），"
                 f"{restored_p2} 条需补跑 Phase 2（图片轮次/背景音）"
                 + (f"，{len(regen_task_ids)} 条强制重生成" if args.regenerate else ""))
        except Exception as exc:
            _log("WARN", "MAIN", f"读取已有输出文件失败 ({exc})，将从头生成全部任务")

    if args.regenerate:
        # Verify all requested task IDs actually exist in events_data
        all_event_ids = {item.get("task_id", "") for item in events_data}
        missing = regen_task_ids - all_event_ids
        if missing:
            _log("WARN", "MAIN",
                 f"以下 task_id 在事件文件中不存在，将被忽略: {sorted(missing)}")

    # 用 task_id 作为每个事件的唯一 ID 并建立索引，保证回填准确。
    # 注意：同一个用户的多个事件共享 p_id，因此不能用 p_id 做唯一键，
    # 否则会出现所有任务的结果都覆盖到同一个事件的 bug。
    task_index = {}
    dup_task_ids = []
    for idx, result in enumerate(events_data):
        t_id = result.get('task_id', '')
        if not t_id:
            _log("WARN", "MAIN", f"第 {idx} 个事件缺少 task_id，将使用索引 'idx-{idx}' 代替")
            t_id = f"idx-{idx}"
            result['task_id'] = t_id
        if t_id in task_index:
            dup_task_ids.append(t_id)
        task_index[t_id] = idx
    if dup_task_ids:
        _log("WARN", "MAIN",
             f"检测到重复 task_id: {dup_task_ids}，后出现者将覆盖前者的回填位置")

    TOTAL_PROMPT_TOKENS = 0
    TOTAL_COMPLETION_TOKENS = 0
    SUCCESS_COUNT = len(done_task_ids)   # 已恢复的完全完成条目计入成功数
    FAIL_COUNT = 0
    interrupted = False

    # 显式管理 executor，便于在 KeyboardInterrupt 时立刻取消未启动任务。
    # 不使用 with 是因为 ThreadPoolExecutor.__exit__ 会硬等所有任务完成（shutdown(wait=True)），
    # Ctrl+C 后会被卡在那里，与我们想"立即停"的目标矛盾。
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    tasks = []
    try:
        for result in events_data:
            if not _is_selected_profile(result):
                continue
            t_id = result.get('task_id', '')
            # Regenerate mode: only process explicitly requested task IDs
            if args.regenerate and t_id not in regen_task_ids:
                continue
            # Normal mode: Phase 1 + Phase 2 均已完成，完全跳过
            if not args.regenerate and t_id in done_task_ids:
                continue
            profile_str    = result.get('profile_str', '')
            event          = result.get('event', {})
            implicit_prefs = result.get('implicit_preferences', []) or []
            explicit_prefs = result.get('explicit_preferences', []) or []
            all_implicit_prefs = all_implicit_prefs_by_pid.get(result.get('p_id'), implicit_prefs)
            if event:
                # Regenerate mode always runs Phase 1 + Phase 2 from scratch.
                # Normal mode: Phase 1 已完成但 Phase 2 未完成时，传入已有对话，只跑 Phase 2
                existing_dlg = None if args.regenerate else phase2only_dialogs.get(t_id, None)
                tasks.append(
                    executor.submit(
                        process_event, t_id, profile_str, event, implicit_prefs,
                        existing_dlg, explicit_prefs, all_implicit_prefs,
                    )
                )

        _log("INFO", "MAIN",
             f"已提交 {len(tasks)} 个并发任务 "
             f"(完全跳过 {len(done_task_ids)} 个, "
             f"仅跑 Phase2 {len(phase2only_dialogs)} 个, "
             f"max_workers={args.workers})")

        # 为什么不用 `for future in as_completed(tasks):`？
        # 在 Windows 上，as_completed 内部使用 threading.Condition.wait() 无限阻塞，
        # 这种 native 等待不会被 SIGINT (Ctrl+C) 打断，导致 signal handler 永远不会被调用，
        # stop_event 也永远不会被置位，整个进程"按了没反应"。
        # 改用 wait(timeout=0.5) 轮询，每 0.5 秒返回一次，让解释器有机会处理 Ctrl+C。
        pending = set(tasks)
        total_count = len(tasks) + len(done_task_ids)
        pbar = tqdm(total=total_count, initial=len(done_task_ids), desc="Generating dialogues...")
        try:
            while pending:
                if stop_event.is_set():
                    break
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    try:
                        out = future.result()
                    except concurrent.futures.CancelledError:
                        pbar.update(1)
                        continue

                    e_id = out['e_id']
                    dialog = out['dialog']
                    TOTAL_PROMPT_TOKENS += out['prompt_tokens']
                    TOTAL_COMPLETION_TOKENS += out['completion_tokens']

                    # 通过 task_id 索引回填（O(1) 查找）
                    target_idx = task_index.get(e_id)
                    if target_idx is None:
                        _log("ERROR", e_id,
                             f"未找到对应的事件索引，无法回填 (task_id={e_id!r})")
                        pbar.update(1)
                        continue

                    event = events_data[target_idx].get('event', {})
                    if not event:
                        _log("WARN", e_id, "对应位置上没有 event 字段，跳过回填")
                        pbar.update(1)
                        continue

                    event['dialog'] = dialog
                    event['dialog_prompt'] = out['dl_prompt']
                    event['image_turn_indices'] = out['image_turn_indices']
                    event['tts_turn_indices'] = out['tts_turn_indices']
                    # phase2_complete 写到顶层 result 而非 event 内（与 event 字段分离）
                    events_data[target_idx]['phase2_complete'] = out.get('phase2_complete', False)

                    total_count_display = len(tasks) + len(done_task_ids)
                    if not dialog:
                        FAIL_COUNT += 1
                        _log("WARN", e_id,
                             "对话为空；"
                             f"进度: 成功 {SUCCESS_COUNT} / 失败 {FAIL_COUNT} / 总 {total_count_display}")
                    else:
                        SUCCESS_COUNT += 1
                        _log("INFO", e_id,
                             f"回填完成; 进度: 成功 {SUCCESS_COUNT} / 失败 {FAIL_COUNT} / 总 {total_count_display}")

                    _flush_results(events_data, SAVE_PATH)
                    pbar.update(1)

                    if stop_event.is_set():
                        break
        finally:
            pbar.close()

        if stop_event.is_set():
            interrupted = True
            _log("WARN", "MAIN",
                 "检测到停止信号，正在取消未启动任务并尽快收尾，"
                 "再按一次 Ctrl+C 可强制立即退出")
            # cancel_futures=True 会丢弃所有还没开始执行的任务；
            # wait=False 让主线程不阻塞等待正在跑的 worker，
            # worker 内部会因 stop_event 与 OPENAI_TIMEOUT 在数秒内自行返回。
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
    finally:
        # 不论是否中断，都把当前已经回填的结果落盘
        _flush_results(events_data, SAVE_PATH)

        total_elapsed = time.time() - t_overall_start
        total_cost = TOTAL_PROMPT_TOKENS * 1 * 0.000001 + TOTAL_COMPLETION_TOKENS * 3 * 0.000001
        _log_raw("=" * 80)
        _log("INFO", "MAIN", "执行结束（被中断）" if interrupted else "全部任务完成")
        _log_raw(f"  - 任务总数:      {len(tasks)}")
        _log_raw(f"  - 成功:          {SUCCESS_COUNT}")
        _log_raw(f"  - 失败:          {FAIL_COUNT}")
        _log_raw(f"  - 输出文件:      {SAVE_PATH}")
        _log_raw(f"  - 日志文件:      {LOG_PATH}")
        _log_raw(f"  - 总耗时:        {total_elapsed:.1f}s "
                 f"({total_elapsed / 60:.1f} 分钟)")
        _log_raw(f"  - Prompt Tokens: {TOTAL_PROMPT_TOKENS}")
        _log_raw(f"  - Completion:    {TOTAL_COMPLETION_TOKENS}")
        _log_raw(f"  - 预估花费:      ${total_cost:.4f}")
        _log_raw("=" * 80)

        _close_log_file()

        if interrupted:
            sys.exit(130)
