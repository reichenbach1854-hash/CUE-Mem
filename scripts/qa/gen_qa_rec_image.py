"""根据用户已有偏好生成"个性化推荐"类图片选择题——四个选项均为图片。

与 gen_qa_pref_image.py 的区别:
- pref_image: "以下哪张图最符合该用户的偏好？"（识别已有偏好）
- rec_image:  "以下哪张图是最适合推荐给该用户的？"（推荐新事物）

推荐逻辑: 用户喜欢 A → 推荐与 A 相似的新事物 B（用户从未直接提及）

Pipeline（三阶段）:
--------
1. 加载 profiles，提取每条偏好。
2. **Stage 1（LLM·描述生成）**：对每条偏好调用 LLM，输出
   - 一道推荐场景的中文题干 Q
   - 四个选项的 image_prompt（描述推荐的新事物场景）
   - 正确答案字母 A
   - 三个错误选项的错误原因
3. **Stage 2（图像生成）**：调用 AIFast Gemini 生成 A/B/C/D 四张 PNG。
4. **Stage 3（LLM·memory clue）**：从对话历史提取支撑推荐依据的证据。
5. 保存至 ./qa/qa_rec_image_mcq.json，图片存至
   ./qa/rec_images/。
"""

import base64
import json
import os
import re
import concurrent.futures
import threading
from pathlib import Path

from tqdm import tqdm
from json_repair import repair_json

from scripts.common.images import generate_aifast_image_to_path
from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.qa.config import profile_path, qa_path

# ========================= 路径 & 模型配置 =========================

PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
FORMATTED_DIALOG_PATH = qa_path("qa_formatted_data_000_019.json")
OUTPUT_PATH = qa_path("qa_rec_image_mcq.json")
IMAGE_DIR = qa_path("rec_images")

CHECKPOINT_EVERY = 5

# ---- LLM（文本）----
LLM_MODEL = env_value("CUE_MEM_LLM_MODEL", "gpt-5.5")

# ---- 图像生成（AIFast Gemini 原生 generateContent）----
AIFAST_IMAGE_MODEL = env_value("CUE_MEM_IMAGE_AIFAST_MODEL", "gemini-3.1-flash-image-preview")

MAX_WORKERS_LLM = 8
MAX_WORKERS_IMG = 4
LLM_RETRIES = 3
IMG_RETRIES = 4
STAGE1_REPAIR_RETRIES = 2
TEMPERATURE_GEN = 0.9
TEMPERATURE_ANS = 0.2

CATEGORIES = [
    "FoodAndDrink",
    "HomeAndSpace",
    "BodyAndHealth",
    "HobbiesAndEntertainment",
    "WorkAndLearning",
    "MobilityAndTravel",
    "Pets",
]

# ========================= Prompts =========================

# ---------------------------------------------------------------------------
# Stage 1 — 根据偏好生成 1 道推荐类图片选择题（题干 + 4 个 image_prompt）
# ---------------------------------------------------------------------------
prompt_gen_image_mcq = '''你需要根据用户偏好信息，设计 **1 道个性化推荐类图片选择题**。该题将用于评估智能体能否基于用户已有偏好，通过视觉场景为用户推荐**可能感兴趣的新事物**。

**核心区别**：这不是在考察"用户已经喜欢什么"，而是考察"基于用户喜欢的 A，能否推荐与 A 相似/契合的新事物 B"。
正确选项的图片应描绘一个**用户从未直接提及**但与其偏好在风格、理念、场景或需求上高度契合的新事物/新场景。

**关键：你需要同时输出一道中文题干，以及 A/B/C/D 四个选项各自对应的"图片生成描述"（英文），以便后续调用图像生成模型分别画出四张照片作为四个选项。**

[输入信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}（explicit = 显式表达；implicit = 间接体现）
证据来源: {evidence_sources}
分析依据:
{analysis}

[设计原则]

1. **题干 (Q)**：
   - 简短中文问句，采用**推荐/建议**的口吻，询问"应该向该用户推荐什么"。
   - 示例："如果要为该用户推荐一款新饮品，以下哪张图中的最可能受到欢迎？"
   - 示例："向该用户推荐一个周末去处，以下哪张图最契合其偏好？"
   - **不要**在题干中泄露偏好描述的关键词。

2. **四个选项的 image_prompt（核心）**：
   - 每个 image_prompt 是一段**中文**的场景描述（60-100 词），描绘一张"用户用手机随手拍下的第一人称视角日常照片"。
   - **正确选项**的 image_prompt 描绘一个**用户从未直接提及的新事物/新场景**，但该事物与用户偏好在核心特征上高度契合。
     - 推荐逻辑示例：用户偏好"在家手冲咖啡" → 正确推荐可以是"虹吸壶咖啡套装"的场景（同为居家精品咖啡器具，但用户未提及）。
     - 推荐逻辑示例：用户偏好"喜欢有设计感的小馆子" → 正确推荐可以是"一家新开的日式庭院风小餐厅"的场景。
   - **3 个干扰选项**的 image_prompt 描绘的推荐事物也属于**同一生活领域**，表面上看起来也可能适合该用户，但在**关键维度**上与用户偏好不契合：
     a. 至少 2 个"强干扰"：与正确选项的推荐方向高度相似，仅 1-2 个关键要素不同（比如不同风格/不同使用场景/不同理念），这 1-2 个关键要素的差异必须足以让模型判断出正确答案。
     b. 第 3 个干扰也必须在同领域内合理可信。
   - 每个 image_prompt 都必须描述**一个完整、具体、可拍照的场景**，包含：
     a. 核心物品/活动的具体外观（颜色、材质、形状、品牌风格等）
     b. 环境/背景（桌面、厨房、户外等）
     c. 光线/时段（自然光、灯光、早晨、傍晚等）
     d. 拍摄视角（俯拍桌面、平视前方等第一人称视角）
   - 四个 image_prompt 应在场景结构和长度上保持**平行**，避免正确选项在描述上显得更具体或更长。
   - **绝对禁止**在 image_prompt 中出现任何文字标签、品牌 logo 文字、选项字母等。

3. **正确答案位置随机化**：正确选项应随机分布在 A/B/C/D 中。

4. **image_prompt 中禁止出现的内容**：
   - 不要出现人脸或完整人像（第一人称视角，拍摄者不入镜）
   - 不要出现任何中文或英文文字在画面中
   - 不要出现 watermark、logo

[示例——仅供理解格式，请勿照抄]
偏好描述: "习惯在家自制咖啡，偏好在开始工作前先泡一杯"
推荐思路：用户喜欢居家精品咖啡 → 推荐同类新器具

可能的 image_prompt 设计思路（正确选项——推荐新事物）:
"用户用手机随手拍下的第一人称视角日常照片，画面主体是一台全新的虹吸壶咖啡套装摆在用户家厨房的木质台面上，虹吸壶由玻璃和金属支架组成，旁边有一小袋精品咖啡豆和一个手摇磨豆机，窗外晨光照进来，台面整洁，氛围安静而精致"

强干扰 1（同为新咖啡器具，但偏商用/社交场景，不符合居家独处偏好）:
"用户用手机随手拍下的第一人称视角日常照片，画面主体是一台自动意式咖啡机摆在开放式厨房的大理石岛台上，旁边有多个咖啡杯和奶缸，像是准备招待朋友，背景是宽敞明亮的客厅，氛围热闹社交"

强干扰 2（同为居家咖啡器具但偏速溶/便捷方向，不符合精品手冲偏好）:
"用户用手机随手拍下的第一人称视角日常照片，画面主体是一台胶囊咖啡机摆在用户家厨房台面上，旁边整齐排列着彩色咖啡胶囊，一杯已经冲好的咖啡放在马克杯里，窗外晨光柔和，氛围简洁高效"

干扰 3（完全不同品类——茶而非咖啡）:
"用户用手机随手拍下的第一人称视角日常照片，画面主体是一套日式茶具摆在用户家的茶几上，包括一把侧把壶和两个小茶杯，旁边有一罐茶叶，晨光从窗帘缝隙中照进来，氛围宁静雅致"

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
- "A_false_reason" / "B_false_reason" / "C_false_reason" / "D_false_reason"：对于**3 个错误选项**，分别用一句简短中文说明该推荐**为何不契合用户偏好**（即与用户真实偏好的关键差异）；**正确选项**对应的字段填空字符串 ""。
```json
{{
    "Q": "中文题干文本",
    "options": {{
        "A": "选项 A 的 image_prompt（中文场景描述，60-100 词）",
        "B": "选项 B 的 image_prompt（中文场景描述，60-100 词）",
        "C": "选项 C 的 image_prompt（中文场景描述，60-100 词）",
        "D": "选项 D 的 image_prompt（中文场景描述，60-100 词）"
    }},
    "A": "正确答案字母（A/B/C/D 之一）",
    "A_false_reason": "选项 A 错误原因（若 A 为正确答案则填空字符串）",
    "B_false_reason": "选项 B 错误原因（若 B 为正确答案则填空字符串）",
    "C_false_reason": "选项 C 错误原因（若 C 为正确答案则填空字符串）",
    "D_false_reason": "选项 D 错误原因（若 D 为正确答案则填空字符串）"
}}
```
'''

# ---------------------------------------------------------------------------
# Stage 3 — 根据对话历史提取 memory clue
# ---------------------------------------------------------------------------
prompt_memory_clue = '''你是一个精确的答题系统。你会收到：
1. 【偏好信息】—— 该用户的**真实偏好描述**，这是判定推荐是否契合的权威依据。
2. 【相关对话历史】—— 与该偏好相关的对话记录（含文字、图像描述、音频描述），用于定位 memory clue。
3. 一道已经确定了正确答案的**个性化推荐类**选择题信息。

你的任务：
- 从【相关对话历史】中找出所有能支撑"为何这个推荐最契合该用户"的证据，作为 memory clue 返回。

[输入说明]
【相关对话历史】是与该偏好相关的若干 session 拼接：
- 每条用户/助手消息以 [Dxx:NN] 表示其轮次编号；
- 用户在某轮次分享的图像描述以 "图像[Dxx-NNN.png]:" 出现；
- 该 session 的背景音频或用户语音消息描述以 "音频[Dxx-NNN.wav]:" 出现。

[偏好信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}
证据来源: {evidence_sources}
分析依据: {analysis}

[已确定的正确答案]
题干: {question}
正确答案: {answer_letter}
正确选项描述: {answer_desc}

[判定原则]
1. **遍历**【相关对话历史】中的全部信息，把**所有**能够支撑"该推荐与用户偏好契合"的证据都列入 `memory clue`。
2. 从三类线索中选用证据：(a) 对话文字、(b) 图像描述、(c) 音频/语音描述。**任何一类**都可独立支撑。
3. **禁止**把与推荐依据无直接关系的轮次/图像/音频塞进 `memory clue`。
4. memory clue 元素格式：
   - 对话证据: "Dxx:NN"
   - 图像证据: "Dxx-NNN.png"
   - 音频证据: "Dxx-NNN.wav"
   - 每一条 clue 都必须**真实出现**在【相关对话历史】中。
   - 如果无相关证据，返回空列表 []。

[相关对话历史]
{dialog_str}

[输出格式]
**严格输出**如下 JSON 结构，不要添加任何额外说明文字：
```json
{{
    "memory clue": ["D01:03", "D01-001.png"]
}}
```
'''


# ========================= Utility functions =========================

def load_json_or_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    return load_records(path)


def call_llm(prompt: str, temperature: float = 0.7) -> tuple:
    client = openai_client(
        api_key_env="CUE_MEM_LLM_API_KEY",
        base_url_env="CUE_MEM_LLM_BASE_URL",
    )
    last_err = None
    last_text = None

    for _ in range(LLM_RETRIES):
        try:
            api_res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            text = ""
            usage_info = None
            for chunk in api_res:
                if chunk.choices:
                    text += message_content_to_text(chunk.choices[0].delta.content)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            last_text = text
            cleaned = (
                text.strip().replace("```json", "").replace("```", "").strip()
            )
            data = json.loads(repair_json(cleaned))
            tokens_in = usage_value(usage_info, "prompt_tokens")
            tokens_out = usage_value(usage_info, "completion_tokens")
            return data, tokens_in, tokens_out
        except Exception as e:
            last_err = e
            print(f"[call_llm] Error: {e}; raw_tail={(last_text or '')[-200:]}")

    raise RuntimeError(f"call_llm failed after {LLM_RETRIES} attempts: {last_err}")


def collect_preference_events(formatted_profile: dict, pref_code: str) -> list:
    matched = []
    for event in formatted_profile.get("events", []) or []:
        codes = []
        codes += event.get("explicit_preferences_reflected", []) or []
        codes += event.get("implicit_preferences_reflected", []) or []
        if pref_code in codes:
            matched.append(event)
    return matched


def format_events_for_prompt(events: list) -> str:
    if not events:
        return "（无相关对话历史）"

    sections = []
    seen = set()
    for event in events:
        session_id = event.get("session_id", "?")
        if session_id in seen:
            continue
        seen.add(session_id)

        scene = (event.get("scene_description") or "").strip()
        img_desc = (event.get("user_shared_image_description") or "").strip()
        bg_audio = (event.get("background_audio_info") or "").strip()
        speech = (event.get("human_speech_content") or "").strip()

        section = [f"=== Session {session_id} ==="]
        if scene:
            section.append(f"场景: {scene}")
        if img_desc and img_desc.lower() != "none":
            section.append(f"图像总览: {img_desc}")
        if bg_audio and bg_audio.lower() != "none":
            section.append(f"背景音频: {bg_audio}")
        if speech and speech.lower() != "none":
            section.append(f"用户语音: {speech}")

        bg_audio_map: dict = {}
        user_turn_idx = 0
        dialog_list = event.get("dialog_list", []) or []
        for dt in event.get("dialog", []) or []:
            if dt.get("role") != "user":
                continue
            if user_turn_idx < len(dialog_list):
                rid = dialog_list[user_turn_idx].get("round", "")
                ba = (dt.get("background_audio") or "").strip()
                if ba and ba.lower() != "none":
                    bg_audio_map[rid] = ba
            user_turn_idx += 1

        section.append("对话:")
        for turn in dialog_list:
            round_id = turn.get("round", "?")
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            section.append(f"  [{round_id}] User: {user}")
            if round_id in bg_audio_map:
                section.append(f"    └─ 背景音频: {bg_audio_map[round_id]}")
            for k, v in turn.items():
                if k in ("round", "user", "assistant"):
                    continue
                if k.endswith(".png"):
                    section.append(f"    └─ 图像[{k}]: {v}")
                elif k.endswith(".wav"):
                    section.append(f"    └─ 音频[{k}]: {v}")
            if assistant:
                section.append(f"  [{round_id}] Assistant: {assistant}")
        sections.append("\n".join(section))

    return "\n\n".join(sections)


def collect_valid_clue_keys(events: list) -> set:
    valid: set = set()
    for event in events:
        for turn in event.get("dialog_list", []) or []:
            round_id = turn.get("round")
            if isinstance(round_id, str) and round_id:
                valid.add(round_id)
            for k in turn:
                if k in ("round", "user", "assistant"):
                    continue
                if k.endswith(".png") or k.endswith(".wav"):
                    valid.add(k)
    return valid


def filter_memory_clues(clues, valid_keys: set) -> list:
    if not isinstance(clues, list):
        return []
    seen = []
    for c in clues:
        if isinstance(c, str) and c.strip() and c.strip() in valid_keys and c.strip() not in seen:
            seen.append(c.strip())
    return seen


def validate_image_mcq_errors(item: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["root must be a JSON object"]

    if not isinstance(item.get("Q"), str) or not item["Q"].strip():
        errors.append("Q must be a non-empty string")

    options = item.get("options")
    if not isinstance(options, dict):
        errors.append("options must be an object with A/B/C/D")
        options = {}
    for letter in ["A", "B", "C", "D"]:
        value = options.get(letter)
        text = str(value or "").strip()
        if not text:
            errors.append(f"options.{letter} must be a non-empty image prompt")
        elif text.upper() in {"A", "B", "C", "D"} or len(text) < 20:
            errors.append(
                f"options.{letter} looks invalid; it must be a full visual scene description, not an answer letter or short phrase"
            )

    answer = (item.get("A") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        errors.append("A must be exactly one answer letter among A/B/C/D")

    for letter in ["A", "B", "C", "D"]:
        reason = item.get(f"{letter}_false_reason")
        if reason is None:
            errors.append(f"{letter}_false_reason is missing")
            continue
        reason_text = str(reason).strip()
        if answer in {"A", "B", "C", "D"} and letter == answer and reason_text:
            errors.append(f"{letter}_false_reason must be an empty string because {letter} is the correct option")
        if letter != answer and not reason_text:
            errors.append(f"{letter}_false_reason must explain why option {letter} is wrong")

    return errors


def validate_image_mcq(item: dict) -> bool:
    return not validate_image_mcq_errors(item)


def build_stage1_repair_prompt(original_prompt: str, bad_mcq, errors: list[str]) -> str:
    return (
        f"{original_prompt}\n\n"
        "【上一轮输出不合法，需要修复】\n"
        "请根据下面的错误列表，重新输出一个完整、合法的 JSON 对象。只输出 JSON，不要解释，不要 markdown。\n"
        "必须满足：\n"
        "- options.A/B/C/D 每一项都必须是完整、具体、可拍照的中文 image_prompt，不能是空字符串，不能只是 A/B/C/D 字母。\n"
        "- 字段 A 只能是正确答案字母 A/B/C/D 之一。\n"
        "- 正确选项对应的 *_false_reason 必须是空字符串 \"\"。\n"
        "- 其他三个错误选项对应的 *_false_reason 必须非空，并说明为什么不契合用户偏好。\n"
        "- 保持推荐题逻辑：正确选项是用户未直接提及但高度契合的新推荐，干扰项同领域但关键维度不契合。\n\n"
        "错误列表：\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\n\n上一轮不合法 JSON：\n"
        + json.dumps(bad_mcq, ensure_ascii=False, indent=2)
    )


# ========================= Image generation =========================

def _save_from_data_url(data_url: str, save_path: Path) -> bool:
    try:
        if "," not in data_url or not data_url.strip().lower().startswith("data:"):
            return False
        _, b64_part = data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as exc:
        print(f"    [DATA_URL SAVE ERR] {exc}")
        return False


def _download_image(url: str, save_path: Path) -> bool:
    try:
        import requests
        save_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [DOWNLOAD ERR] {exc}")
        return False


def _save_image_from_url_or_dataurl(src: str, save_path: Path) -> bool:
    s = (src or "").strip()
    if not s:
        return False
    if s.lower().startswith("data:"):
        return _save_from_data_url(s, save_path)
    if s.startswith("http://") or s.startswith("https://"):
        return _download_image(s, save_path)
    print(f"    [SAVE ERR] unsupported image source: {s[:80]!r}...")
    return False


def generate_option_image(image_prompt: str, save_path: Path) -> bool:
    full_prompt = (
        "Generate one highly realistic and clear photographic image, "
        "captured from a first-person perspective as if taken by a user with their smartphone. "
        "Subject in sharp focus. Natural bright lighting. No text, watermark, or logo. "
        "No human face or full body in the frame. "
        f"{image_prompt} "
        "Output one high-quality, sharp, clear image."
    )
    return generate_aifast_image_to_path(
        full_prompt,
        save_path,
        model=AIFAST_IMAGE_MODEL,
        retries=IMG_RETRIES,
    )


def _extract_urls_from_message(message) -> list:
    urls = []
    if message is None:
        return urls
    if hasattr(message, "model_dump"):
        try:
            data = message.model_dump()
        except Exception:
            data = {}
    elif isinstance(message, dict):
        data = message
    else:
        data = {}

    images = data.get("images")
    if images is None:
        images = getattr(message, "images", None)
    if not images:
        return urls

    for img in images:
        if isinstance(img, dict):
            iu = img.get("image_url")
            if isinstance(iu, dict):
                u = iu.get("url")
            elif isinstance(iu, str):
                u = iu
            else:
                u = None
            if not u:
                u = img.get("url")
        else:
            iu = getattr(img, "image_url", None)
            u = getattr(iu, "url", None) if iu is not None else getattr(img, "url", None)
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


# ========================= Per-preference helpers =========================

def _pref_common_fields(pref: dict) -> tuple:
    subcategory = pref.get("subcategory", "")
    preference = pref.get("preference", "")
    expression_type = pref.get("expression_type", "")
    evidence_sources = pref.get("evidence_sources", [])
    analysis = pref.get("analysis", [])
    evidence_str = ", ".join(evidence_sources) if isinstance(evidence_sources, list) else str(evidence_sources)
    analysis_str = "\n".join(f"  - {a}" for a in analysis) if isinstance(analysis, list) else str(analysis)
    return subcategory, preference, expression_type, evidence_sources, evidence_str, analysis, analysis_str


# ========================= Stage 1: LLM gen descriptions =========================

def run_stage1_one(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    pref_code = f"{category}-{pref_idx}"
    qa_id = f"{p_id}-{category}-{pref_idx}"
    subcategory, preference, expression_type, evidence_sources, evidence_str, analysis, analysis_str = _pref_common_fields(pref)

    events = collect_preference_events(formatted_profile, pref_code)
    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    prompt1 = prompt_gen_image_mcq.format(
        category=category,
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
    )

    tokens_in, tokens_out = 0, 0
    mcq = None
    current_prompt = prompt1
    errors: list[str] = []

    for attempt in range(STAGE1_REPAIR_RETRIES + 1):
        try:
            mcq, tin, tout = call_llm(
                current_prompt,
                temperature=TEMPERATURE_GEN if attempt == 0 else TEMPERATURE_ANS,
            )
            tokens_in += tin
            tokens_out += tout
        except Exception as e:
            print(f"[{qa_id}] Stage 1 (gen descriptions) failed on attempt {attempt + 1}: {e}")
            if attempt >= STAGE1_REPAIR_RETRIES:
                return None, tokens_in, tokens_out
            errors = [f"LLM call or JSON parse failed: {e}"]
            current_prompt = build_stage1_repair_prompt(prompt1, mcq or {}, errors)
            continue

        errors = validate_image_mcq_errors(mcq)
        if not errors:
            if attempt > 0:
                print(f"[{qa_id}] Stage 1 repaired after {attempt} retry attempt(s).")
            break

        print(
            f"[{qa_id}] Stage 1 invalid MCQ on attempt {attempt + 1}: "
            + "; ".join(errors[:6])
            + (f"; ... {len(errors) - 6} more" if len(errors) > 6 else "")
        )
        if attempt >= STAGE1_REPAIR_RETRIES:
            print(f"[{qa_id}] Stage 1 returned invalid MCQ after repair attempts: {mcq}")
            return None, tokens_in, tokens_out
        current_prompt = build_stage1_repair_prompt(prompt1, mcq, errors)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "category": category,
        "subcategory": subcategory,
        "preference": preference,
        "expression_type": expression_type,
        "evidence_sources": evidence_sources,
        "Q": mcq["Q"].strip(),
        "question_image_descriptions": {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]},
        "option_images": {"A": "", "B": "", "C": "", "D": ""},
        "A": mcq["A"].strip().upper(),
        "A_false_reason": str(mcq.get("A_false_reason", "")).strip(),
        "B_false_reason": str(mcq.get("B_false_reason", "")).strip(),
        "C_false_reason": str(mcq.get("C_false_reason", "")).strip(),
        "D_false_reason": str(mcq.get("D_false_reason", "")).strip(),
        "memory clue": [],
        "matched_session_ids": matched_sessions,
        "type": "recommendation_image_mcq",
    }
    return record, tokens_in, tokens_out


# ========================= Stage 2: generate images =========================

def run_stage2_one(record: dict) -> tuple:
    qa_id = record["qa_id"]
    image_prompts = record.get("question_image_descriptions", {})
    if not all(image_prompts.get(k) for k in ["A", "B", "C", "D"]):
        print(f"[{qa_id}] Stage 2 skipped: missing image_prompts")
        return record, 0, 0

    img_dir = Path(IMAGE_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = dict(record.get("option_images", {}))
    all_ok = True
    for letter in ["A", "B", "C", "D"]:
        if image_paths.get(letter) and os.path.exists(image_paths[letter]):
            continue

        safe_qid = re.sub(r"[^A-Za-z0-9_\-]", "_", qa_id)
        img_path = img_dir / f"{safe_qid}_{letter}.png"

        if img_path.exists():
            image_paths[letter] = str(img_path)
            continue

        print(f"  [{qa_id}] generating image {letter} ...")
        ok = generate_option_image(image_prompts[letter], img_path)
        if ok:
            image_paths[letter] = str(img_path)
        else:
            print(f"  [{qa_id}] FAILED to generate image {letter}")
            image_paths[letter] = ""
            all_ok = False

    if not all_ok:
        print(f"[{qa_id}] some images failed, record saved with partial paths")

    record["option_images"] = image_paths
    return record, 0, 0


# ========================= Stage 3: memory clue =========================

def run_stage3_one(record: dict, formatted_profile: dict) -> tuple:
    qa_id = record["qa_id"]
    subcategory = record.get("subcategory", "")
    preference = record.get("preference", "")
    expression_type = record.get("expression_type", "")
    evidence_sources = record.get("evidence_sources", [])
    evidence_str = ", ".join(evidence_sources) if isinstance(evidence_sources, list) else str(evidence_sources)
    analysis_raw = record.get("analysis", [])
    analysis_str = "\n".join(f"  - {a}" for a in analysis_raw) if isinstance(analysis_raw, list) else str(analysis_raw)

    parts = qa_id.split("-", 1)
    pref_code = parts[1] if len(parts) == 2 else ""

    events = collect_preference_events(formatted_profile, pref_code)
    dialog_str = format_events_for_prompt(events)
    valid_keys = collect_valid_clue_keys(events)

    question = record.get("Q", "")
    answer_letter = record.get("A", "")
    image_prompts = record.get("question_image_descriptions", {})
    answer_desc = image_prompts.get(answer_letter, "")

    prompt3 = prompt_memory_clue.format(
        category=record.get("category", ""),
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
        question=question,
        answer_letter=answer_letter,
        answer_desc=answer_desc,
        dialog_str=dialog_str,
    )

    tokens_in, tokens_out = 0, 0
    memory_clue = []
    try:
        ans, tin, tout = call_llm(prompt3, temperature=TEMPERATURE_ANS)
        tokens_in += tin
        tokens_out += tout
        raw_clues = ans.get("memory clue")
        if raw_clues is None:
            raw_clues = ans.get("memory_clue")
        memory_clue = filter_memory_clues(raw_clues, valid_keys)
    except Exception as e:
        print(f"[{qa_id}] Stage 3 (memory clue) failed: {e}")

    record["memory clue"] = memory_clue
    return record, tokens_in, tokens_out


# ========================= Combined (all stages) =========================

def build_rec_image_qa(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    record, tin1, tout1 = run_stage1_one(p_id, category, pref_idx, pref, formatted_profile)
    if record is None:
        return None, tin1, tout1
    tokens_in, tokens_out = tin1, tout1

    record, _, _ = run_stage2_one(record)

    record, tin3, tout3 = run_stage3_one(record, formatted_profile)
    tokens_in += tin3
    tokens_out += tout3

    return record, tokens_in, tokens_out


# ========================= I/O helpers =========================

def _sort_key(r):
    return (
        r["p_id"],
        CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else 99,
        r["qa_id"],
    )


def _save_checkpoint(records: list, path: str):
    sorted_records = sorted(records, key=_sort_key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, ensure_ascii=False, indent=4)
    try:
        os.replace(tmp, path)
    except PermissionError:
        import shutil
        shutil.copy2(tmp, path)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_existing(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["qa_id"]: r for r in records}
    except Exception as e:
        print(f"WARN: failed to load {path} for resume: {e}")
        return {}


def _is_record_complete(record: dict) -> bool:
    paths = record.get("option_images", {})
    return all(
        isinstance(paths.get(k), str) and paths[k] and os.path.exists(paths[k])
        for k in ["A", "B", "C", "D"]
    )


# ========================= Stage runners =========================

def _load_common():
    profiles = load_json_or_jsonl(PROFILE_PATH)
    if not profiles:
        print(f"ERROR: cannot load {PROFILE_PATH}.")
        return None, None, None, None
    print(f"Loaded {len(profiles)} profile(s) from {PROFILE_PATH}.")

    formatted_profiles = load_json_or_jsonl(FORMATTED_DIALOG_PATH)
    if not formatted_profiles:
        print(f"ERROR: cannot load {FORMATTED_DIALOG_PATH}.")
        return None, None, None, None
    total_events = sum(len(p.get("events", []) or []) for p in formatted_profiles)
    print(
        f"Loaded {len(formatted_profiles)} formatted profile(s) "
        f"({total_events} events) from {FORMATTED_DIALOG_PATH}."
    )
    formatted_by_pid = {fp.get("p_id", i): fp for i, fp in enumerate(formatted_profiles)}

    existing = _load_existing(OUTPUT_PATH)
    if existing:
        print(f"[resume] 已有 {len(existing)} 条记录。")

    return profiles, formatted_by_pid, existing, formatted_profiles


def parse_profile_id_filter(raw_values: list[str] | None) -> set[int] | None:
    if not raw_values:
        return None
    ids: set[int] = set()
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            if part.strip():
                ids.add(int(part))
    return ids


def profile_allowed(p_id: int, max_profiles: int | None = None, only_profile_ids: set[int] | None = None) -> bool:
    if max_profiles is not None and max_profiles > 0 and p_id >= max_profiles:
        return False
    if only_profile_ids is not None and p_id not in only_profile_ids:
        return False
    return True


def _print_summary(qa_map: dict, n_existing: int, total_in: int, total_out: int):
    all_records = list(qa_map.values())
    by_cat: dict = {}
    by_type = {"explicit": 0, "implicit": 0}
    complete = 0
    for r in all_records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        et = r.get("expression_type", "")
        by_type[et] = by_type.get(et, 0) + 1
        if _is_record_complete(r):
            complete += 1

    print(f"\n总记录数: {len(all_records)} (本次新增/更新 {len(all_records) - n_existing})")
    print(f"图片完整: {complete} / {len(all_records)}")
    print("By category:")
    for c in CATEGORIES:
        if c in by_cat:
            print(f"  - {c}: {by_cat[c]}")
    print(f"By expression_type: {by_type}")
    if total_in or total_out:
        print(f"Tokens — prompt: {total_in}, completion: {total_out}")
        print(
            f"Estimated cost (input $1/M, output $3/M): "
            f"${(total_in * 1e-6 + total_out * 3e-6):.4f}"
        )
    print(f"Output -> {OUTPUT_PATH}")
    print(f"Images -> {IMAGE_DIR}/")


# ---------------------------------------------------------------------------
# Stage 1 only
# ---------------------------------------------------------------------------
def main_stage1(sample=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, _ = _load_common()
    if profiles is None:
        return

    all_tasks = []
    for p_idx, profile in enumerate(profiles):
        p_id = profile.get("id", p_idx)
        if not profile_allowed(p_id, max_profiles, only_profile_ids):
            continue
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        for category in CATEGORIES:
            for pref_idx, pref in enumerate(profile.get(category, []) or []):
                qa_id = f"{p_id}-{category}-{pref_idx}"
                rec = existing.get(qa_id)
                if rec and rec.get("Q") and rec.get("question_image_descriptions"):
                    descs = rec["question_image_descriptions"]
                    if all(descs.get(k) for k in ["A", "B", "C", "D"]):
                        continue
                all_tasks.append((p_id, category, pref_idx, pref, fp))

    if sample is not None:
        all_tasks = all_tasks[:sample]

    if not all_tasks:
        print("Stage 1: 所有 image_prompt 均已生成，无需重新运行。")
        return
    print(f"Stage 1: 待生成 {len(all_tasks)} 条 (已有 {len(existing)} 条记录)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM) as executor:
        for p_id, category, pref_idx, pref, fp in all_tasks:
            futures[executor.submit(run_stage1_one, p_id, category, pref_idx, pref, fp)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 1: gen descriptions"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ---------------------------------------------------------------------------
# Stage 2 only
# ---------------------------------------------------------------------------
def main_stage2(sample=None, regen_ids=None, max_profiles=None, only_profile_ids=None):
    _, _, existing, _ = _load_common()
    if existing is None:
        return

    # --regen-ids 模式：删除指定 qa_id 的已有图片，强制重新生成
    if regen_ids:
        regen_set = set(regen_ids)
        not_found = regen_set - set(existing.keys())
        if not_found:
            print(f"WARN: 以下 qa_id 在已有记录中不存在，将跳过: {not_found}")
        tasks = []
        for qid in regen_ids:
            rec = existing.get(qid)
            if rec is None:
                continue
            if not profile_allowed(int(rec.get("p_id", 0) or 0), max_profiles, only_profile_ids):
                continue
            descs = rec.get("question_image_descriptions", {})
            if not all(descs.get(k) for k in ["A", "B", "C", "D"]):
                print(f"WARN: [{qid}] 缺少 image_prompt，无法生成图片，跳过")
                continue
            # 删除已有图片文件，清空路径，强制重新生成
            for letter in "ABCD":
                old_path = rec.get("option_images", {}).get(letter, "")
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"  [{qid}] deleted old image {letter}: {old_path}")
            rec["option_images"] = {"A": "", "B": "", "C": "", "D": ""}
            tasks.append(rec)
        if not tasks:
            print("Stage 2 regen: 没有有效的 qa_id 需要重新生成。")
            return
        print(f"Stage 2 regen: 将重新生成 {len(tasks)} 条记录的图片")
    else:
        tasks = []
        for qa_id, rec in existing.items():
            if not profile_allowed(int(rec.get("p_id", 0) or 0), max_profiles, only_profile_ids):
                continue
            descs = rec.get("question_image_descriptions", {})
            if not all(descs.get(k) for k in ["A", "B", "C", "D"]):
                continue
            if _is_record_complete(rec):
                continue
            tasks.append(rec)

        if sample is not None:
            tasks = tasks[:sample]

        if not tasks:
            print("Stage 2: 所有图片均已生成，无需重新运行。")
            return
        print(f"Stage 2: 待生成图片 {len(tasks)} 条记录")

    qa_map = dict(existing)
    lock = threading.Lock()
    new_since_cp = 0

    def _on(record):
        nonlocal new_since_cp
        qa_map[record["qa_id"]] = record
        new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_IMG) as executor:
        for rec in tasks:
            futures[executor.submit(run_stage2_one, rec)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 2: gen images"):
            rec, _, _ = future.result()
            with lock:
                _on(rec)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), 0, 0)


# ---------------------------------------------------------------------------
# Stage 3 only
# ---------------------------------------------------------------------------
def main_stage3(sample=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, _ = _load_common()
    if profiles is None:
        return

    tasks = []
    for qa_id, rec in existing.items():
        if not rec.get("Q") or not rec.get("A"):
            continue
        if rec.get("memory clue"):
            continue
        p_id = rec.get("p_id", 0)
        if not profile_allowed(int(p_id or 0), max_profiles, only_profile_ids):
            continue
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        tasks.append((rec, fp))

    if sample is not None:
        tasks = tasks[:sample]

    if not tasks:
        print("Stage 3: 所有 memory clue 均已生成，无需重新运行。")
        return
    print(f"Stage 3: 待提取 memory clue {len(tasks)} 条记录")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        qa_map[record["qa_id"]] = record
        new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM) as executor:
        for rec, fp in tasks:
            futures[executor.submit(run_stage3_one, rec, fp)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 3: memory clue"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ---------------------------------------------------------------------------
# All stages (default)
# ---------------------------------------------------------------------------
def main_all(sample=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, _ = _load_common()
    if profiles is None:
        return

    all_tasks = []
    for p_idx, profile in enumerate(profiles):
        p_id = profile.get("id", p_idx)
        if not profile_allowed(p_id, max_profiles, only_profile_ids):
            continue
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        for category in CATEGORIES:
            for pref_idx, pref in enumerate(profile.get(category, []) or []):
                qa_id = f"{p_id}-{category}-{pref_idx}"
                if qa_id in existing and _is_record_complete(existing[qa_id]):
                    continue
                all_tasks.append((p_id, category, pref_idx, pref, fp))

    if sample is not None:
        all_tasks = all_tasks[:sample]

    if not all_tasks:
        print("所有 recommendation image QA 均已生成，无需重新运行。")
        return
    print(f"待生成: {len(all_tasks)} 条推荐图片 QA (已有 {len(existing)} 条完整记录)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_IMG) as executor:
        for p_id, category, pref_idx, pref, fp in all_tasks:
            futures[executor.submit(build_rec_image_qa, p_id, category, pref_idx, pref, fp)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Building recommendation image MCQs"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ========================= Entry point =========================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="个性化推荐图片选择题生成（支持分阶段运行）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "只运行指定阶段:\n"
            "  1 = LLM 生成题干 + 4 个 image_prompt（保存到 JSON）\n"
            "  2 = 根据已有 image_prompt 调用图像生成模型生成 4 张图片\n"
            "  3 = 根据已有 Q + answer 从对话历史提取 memory clue\n"
            "不指定则按 1→2→3 顺序全部运行"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="只处理前 N 条任务，用于小批量测试",
    )
    parser.add_argument(
        "--max-profiles",
        "--max_profiles",
        type=int,
        default=None,
        help="只处理 p_id < N 的 profile，例如 --max-profiles 5 只处理 p_id=0..4",
    )
    parser.add_argument(
        "--only_profile_ids",
        "--only-profile-ids",
        nargs="*",
        default=None,
        metavar="P_ID",
        help="只处理指定 p_id，支持空格或逗号分隔，例如 --only_profile_ids 0 或 --only_profile_ids 0,1,2",
    )
    parser.add_argument(
        "--regen-ids",
        type=str,
        nargs="+",
        default=None,
        metavar="QA_ID",
        help=(
            "Stage 2 专用：指定需要重新生成图片的 qa_id 列表。\n"
            "会删除已有图片并强制重新调用图像生成模型。\n"
            "示例: --regen-ids 0-FoodAndDrink-0 0-Pets-1"
        ),
    )
    args = parser.parse_args()
    only_profile_ids = parse_profile_id_filter(args.only_profile_ids)

    if args.max_profiles is not None and args.max_profiles < 1:
        print("ERROR: --max-profiles 必须为正整数")
        return

    if args.max_profiles:
        print(f"[filter] max_profiles={args.max_profiles} -> only p_id < {args.max_profiles}")
    if only_profile_ids is not None:
        print(f"[filter] only_profile_ids={sorted(only_profile_ids)}")

    if args.regen_ids and args.stage not in (None, 2):
        print("ERROR: --regen-ids 仅可与 --stage 2 搭配使用")
        return

    if args.regen_ids and args.stage is None:
        print("=" * 60)
        print("Running Stage 2 (regen mode): regenerate specified images")
        print("=" * 60)
        main_stage2(
            regen_ids=args.regen_ids,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
        return

    if args.stage == 1:
        print("=" * 60)
        print("Running Stage 1 only: LLM gen Q + image_prompts")
        print("=" * 60)
        main_stage1(
            sample=args.sample,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    elif args.stage == 2:
        print("=" * 60)
        print("Running Stage 2 only: generate option images")
        print("=" * 60)
        main_stage2(
            sample=args.sample,
            regen_ids=args.regen_ids,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    elif args.stage == 3:
        print("=" * 60)
        print("Running Stage 3 only: LLM extract memory clue")
        print("=" * 60)
        main_stage3(
            sample=args.sample,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    else:
        print("=" * 60)
        print("Running all stages: 1 → 2 → 3")
        print("=" * 60)
        main_all(
            sample=args.sample,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )


if __name__ == "__main__":
    main()
