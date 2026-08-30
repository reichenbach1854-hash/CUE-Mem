"""根据用户已有偏好生成"个性化推荐"类单选题（A/B/C/D）。

与 preference 类题目的区别:
- preference: "用户喜欢什么？"（识别已有偏好）
- recommendation: "用户可能喜欢什么？"（推荐新事物）

推荐逻辑: 用户喜欢 A → 推荐与 A 相似的 B

Pipeline（两阶段 LLM 调用）:
--------
1. 加载 profiles_000_002_with_anchors.jsonl，提取每个角色的 7 大类偏好。
2. 对每条偏好调用 LLM（Stage 1），生成 1 道推荐类 MCQ（题干 + 4 个选项）。
   正确选项是一个**用户从未提及但与其偏好高度契合的新事物**。
3. 在 qa_formatted_data_000_002.json 中查找相关事件，渲染为对话上下文。
4. 对每道 MCQ 调用 LLM（Stage 2），仅基于对话历史提取支撑已知正确答案的
   memory clue 列表。
5. 保存至 ./qa/qa_recommendation_mcq_000_002.json。

评估维度: 显式推荐准确率、隐式推荐准确率、整体准确率。
"""

import json
import os
import concurrent.futures
import threading

from tqdm import tqdm
from json_repair import repair_json

from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.qa.config import profile_path, qa_path

PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
FORMATTED_DIALOG_PATH = qa_path("qa_formatted_data_000_019.json")
OUTPUT_PATH = qa_path("qa_recommendation_mcq.json")

CHECKPOINT_EVERY = 10

MODEL = env_value("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")

MAX_WORKERS = 4
LLM_RETRIES = 3
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

# ---------------------------------------------------------------------------
# Prompt: Stage 1 — 根据单条偏好生成 1 道推荐类 MCQ
# ---------------------------------------------------------------------------
prompt_gen_mcq = '''请根据给定的【用户偏好信息】，设计 **1 道个性化推荐类单项选择题（MCQ）**，用于评估智能体能否基于用户已有偏好，为其推荐用户**可能感兴趣的新事物**。

**核心区别**：这不是在考察"用户已经喜欢什么"，而是考察"基于用户喜欢的 A，能否推荐与 A 相似/契合的新事物 B"。

[输入信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}（explicit = 用户在对话/图像主体/语音中直接表达；implicit = 通过图像背景/环境音等间接体现）
证据来源: {evidence_sources}
分析依据: {analysis}

[出题要求]
1. **题干**必须是推荐/建议场景的中文问句，询问的是"应该向该用户推荐什么**新事物**"，而非"用户已有什么偏好"。
   - 题干示例：
     - "如果向该用户推荐一款新饮品，以下哪个最可能受到欢迎？"
     - "为这位用户挑选一本书作为礼物，以下哪个最合适？"
     - "向该用户推荐一个周末活动，以下哪个最契合其兴趣？"
   - 题干中**不得**出现偏好描述的原文关键词，不要泄露偏好。
2. 给出 **4 个选项 A/B/C/D**，每个选项是简短中文短语（≤30 汉字）。
   - 每个选项都应该是一个**具体的新事物/新建议**（如一款产品、一种活动、一个目的地、一道菜、一本书等），而非对已有偏好的直接复述。
3. **正确选项**：一个用户**从未直接提及**，但与其偏好在风格、理念、场景或需求上高度契合的新推荐。
   - 推荐逻辑：用户喜欢 A → 推荐与 A 在核心特征上相似的 B。
   - 例如：用户偏好"在家手冲咖啡" → 可推荐"虹吸壶咖啡套装"（同为居家精品咖啡）；而非推荐"速溶咖啡"或"奶茶"。
4. **高迷惑性干扰项设计原则（核心）**：
   a. 4 个选项必须**结构平行**、**互斥**、**同维度同粒度**，长度和信息密度均衡。
   b. **至少 2 个"强干扰项"**：与正确推荐属于同一大类，表面上也可能适合该用户，但在关键维度上与用户偏好不契合。
      - 示例：若用户偏好"安静的独处型运动"，正确推荐"室内攀岩"，强干扰可以是"羽毛球双打"（运动但社交型）或"户外夜跑团"（独处但高强度社交场景），而非"看电视"。
   c. 其余干扰项也须在同领域内合理可信，不能一眼排除。
   d. 正确选项**不得**因措辞更具体或更长而在形式上显得突出。
   e. 不要使用"以上都是/都不是""不确定""无法判断"等元选项。
5. **正确答案位置随机化**：正确选项应随机分布在 A/B/C/D 中。
6. 不要在题面或选项中原样复述偏好描述文本。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "Q": "题干文本（中文）",
    "options": {{
        "A": "选项 A 文本",
        "B": "选项 B 文本",
        "C": "选项 C 文本",
        "D": "选项 D 文本"
    }},
    "answer": "A"
}}
```
'''

# ---------------------------------------------------------------------------
# Prompt: Stage 2 — 基于已知正确答案提取 memory clue
# ---------------------------------------------------------------------------
prompt_answer_clue = '''你是一个精确的答题系统。你会收到：
1. 【偏好信息】—— 该用户的真实偏好描述，用于理解为什么给定推荐是正确的。
2. 【已知正确答案】—— 这道题的正确选项字母，以及对应的选项文本。
3. 【相关对话历史】—— 与该偏好相关的对话记录（含文字、图像描述、音频描述），用于定位 memory clue。
4. 一道**个性化推荐类**单项选择题（A/B/C/D），每个选项是一个向用户推荐的新事物。

你的任务：
- 不需要重新判断答案，正确答案已经给出。
- 你只需要从【相关对话历史】中找出所有能支撑该推荐判断的证据，作为 memory clue 返回。

[输入说明]
【相关对话历史】是与该偏好相关的若干 session 拼接，每条 session 内部按轮次给出对话，并标注了同一轮次中出现的图像/音频证据：
- 每条用户/助手消息以 [Dxx:NN] 表示其轮次编号，其中 Dxx 是 session_id，NN 是该 session 内的轮次序号；
- 用户在某轮次分享的图像描述以 "图像[Dxx-NNN.png]:" 出现；
- 该 session 的背景音频或用户语音消息描述以 "音频[Dxx-NNN.wav]:" 出现。

[偏好信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}
证据来源: {evidence_sources}
分析依据: {analysis}

[已知正确答案]
正确选项: {answer_letter}
正确选项文本: {answer_text}

[提取原则]
1. 以【已知正确答案】为目标，从【相关对话历史】中找出所有能直接支撑该推荐判断的证据。
2. 从三类线索中选用证据：(a) 对话文字、(b) 图像描述、(c) 音频/语音描述。任何一类都可独立支撑答案。
3. **遍历**【相关对话历史】中的全部信息，把**所有**能够作为推荐依据的证据都列入 `memory clue`，不要因为已有一条就提前停止。
4. **禁止**把与答案无直接关系的轮次/图像/音频塞进 `memory clue`；只列真正能支撑结论的证据。
5. memory clue 元素格式：
   - 对话证据使用 "Dxx:NN" 格式（必须严格匹配实际出现的轮次编号）；
   - 图像证据使用 "Dxx-NNN.png" 格式；
   - 音频证据使用 "Dxx-NNN.wav" 格式；
   - 每一条 clue 都必须**真实出现**在【相关对话历史】中，禁止编造；
   - 如果【相关对话历史】中没有任何相关证据，可返回空列表 []。

[相关对话历史]
{dialog_str}

[选择题]
题目：{question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

[输出格式]
**严格输出**如下 JSON 结构，不要添加任何额外说明文字（注意 "memory clue" 中间是空格）：
```json
{{
    "memory clue": ["D01:03", "D01-001.png"]
}}
```
'''


# ========================= Utility functions =========================

def load_json_or_jsonl(path: str) -> list:
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
                model=MODEL,
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


def validate_mcq(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("Q"), str) or not item["Q"].strip():
        return False
    options = item.get("options")
    if not isinstance(options, dict):
        return False
    return all(k in options and str(options[k]).strip() for k in ["A", "B", "C", "D"])


def extract_answer_letter(item: dict) -> str:
    answer = str(item.get("answer", "")).strip().upper()
    return answer if answer in {"A", "B", "C", "D"} else ""


# ========================= Core build function =========================

def build_recommendation_qa(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    """Two-stage LLM pipeline for one preference → one recommendation MCQ with memory clue."""
    pref_code = f"{category}-{pref_idx}"
    qa_id = f"{p_id}-{category}-{pref_idx}"

    subcategory = pref.get("subcategory", "")
    preference = pref.get("preference", "")
    expression_type = pref.get("expression_type", "")
    evidence_sources = pref.get("evidence_sources", [])
    analysis = pref.get("analysis", [])

    evidence_str = ", ".join(evidence_sources) if isinstance(evidence_sources, list) else str(evidence_sources)
    analysis_str = "\n".join(f"  - {a}" for a in analysis) if isinstance(analysis, list) else str(analysis)

    events = collect_preference_events(formatted_profile, pref_code)
    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    tokens_in, tokens_out = 0, 0

    # ---- Stage 1: generate recommendation MCQ ----
    prompt1 = prompt_gen_mcq.format(
        category=category,
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
    )

    try:
        mcq, tin, tout = call_llm(prompt1, temperature=TEMPERATURE_GEN)
        tokens_in += tin
        tokens_out += tout
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (gen) failed: {e}")
        return None, 0, 0

    if not validate_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid MCQ: {mcq}")
        return None, tokens_in, tokens_out
    answer_letter = extract_answer_letter(mcq)
    if not answer_letter:
        print(f"[{qa_id}] Stage 1 returned invalid answer: {mcq}")
        return None, tokens_in, tokens_out

    options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
    question = mcq["Q"].strip()
    answer_text = options[answer_letter]

    # ---- Stage 2: memory clue only ----
    dialog_str = format_events_for_prompt(events)
    valid_keys = collect_valid_clue_keys(events)

    prompt2 = prompt_answer_clue.format(
        category=category,
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
        dialog_str=dialog_str,
        question=question,
        opt_a=options["A"],
        opt_b=options["B"],
        opt_c=options["C"],
        opt_d=options["D"],
        answer_letter=answer_letter,
        answer_text=answer_text,
    )

    try:
        ans, tin2, tout2 = call_llm(prompt2, temperature=TEMPERATURE_ANS)
        tokens_in += tin2
        tokens_out += tout2
    except Exception as e:
        print(f"[{qa_id}] Stage 2 (clue) failed: {e}")
        return None, tokens_in, tokens_out

    raw_clues = ans.get("memory clue")
    if raw_clues is None:
        raw_clues = ans.get("memory_clue")
    memory_clue = filter_memory_clues(raw_clues, valid_keys)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "category": category,
        "subcategory": subcategory,
        "preference": preference,
        "expression_type": expression_type,
        "evidence_sources": evidence_sources,
        "Q": question,
        "options": options,
        "A": answer_letter,
        "memory clue": memory_clue,
        "matched_session_ids": matched_sessions,
        "type": "recommendation_mcq",
    }
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


# ========================= Main =========================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="个性化推荐类选择题生成",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="只处理前 N 条任务，用于小批量测试",
    )
    args = parser.parse_args()

    profiles = load_json_or_jsonl(PROFILE_PATH)
    if not profiles:
        print(f"ERROR: cannot load {PROFILE_PATH}.")
        return
    print(f"Loaded {len(profiles)} profile(s) from {PROFILE_PATH}.")

    formatted_profiles = load_json_or_jsonl(FORMATTED_DIALOG_PATH)
    if not formatted_profiles:
        print(f"ERROR: cannot load {FORMATTED_DIALOG_PATH}.")
        return
    total_events = sum(len(p.get("events", []) or []) for p in formatted_profiles)
    print(
        f"Loaded {len(formatted_profiles)} formatted profile(s) "
        f"({total_events} events) from {FORMATTED_DIALOG_PATH}."
    )
    formatted_by_pid = {fp.get("p_id", i): fp for i, fp in enumerate(formatted_profiles)}

    existing = _load_existing(OUTPUT_PATH)
    if existing:
        print(f"[resume] 已有 {len(existing)} 条记录，将跳过已完成的 qa_id。")

    all_tasks = []
    for p_idx, profile in enumerate(profiles):
        p_id = profile.get("id", p_idx)
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            print(f"[p_id={p_id}] no formatted profile found, skip.")
            continue

        for category in CATEGORIES:
            pref_list = profile.get(category, []) or []
            for pref_idx, pref in enumerate(pref_list):
                qa_id = f"{p_id}-{category}-{pref_idx}"
                if qa_id in existing:
                    continue
                all_tasks.append((p_id, category, pref_idx, pref, fp))

    if args.sample is not None:
        all_tasks = all_tasks[:args.sample]

    if not all_tasks:
        print("所有 recommendation QA 均已生成，无需重新运行。")
        return

    print(f"待生成: {len(all_tasks)} 条推荐 QA (已有 {len(existing)} 条记录)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = 0
    total_out = 0
    new_since_checkpoint = 0

    def _on_result(record, tin, tout):
        nonlocal total_in, total_out, new_since_checkpoint
        total_in += tin
        total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_checkpoint += 1
        if new_since_checkpoint >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_checkpoint = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for p_id, category, pref_idx, pref, fp in all_tasks:
            fut = executor.submit(
                build_recommendation_qa, p_id, category, pref_idx, pref, fp,
            )
            futures[fut] = None

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Building recommendation MCQs",
        ):
            record, tin, tout = future.result()
            with lock:
                _on_result(record, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)

    all_records = list(qa_map.values())
    by_cat = {}
    by_type = {"explicit": 0, "implicit": 0}
    for r in all_records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        et = r.get("expression_type", "")
        by_type[et] = by_type.get(et, 0) + 1

    print(f"\n总记录数: {len(all_records)} (本次新增 {len(all_records) - len(existing)})")
    print("By category:")
    for c in CATEGORIES:
        if c in by_cat:
            print(f"  - {c}: {by_cat[c]}")
    print(f"By expression_type: {by_type}")
    print(f"Tokens — prompt: {total_in}, completion: {total_out}")
    print(
        f"Estimated cost (input $1/M, output $3/M): "
        f"${(total_in * 1e-6 + total_out * 3e-6):.4f}"
    )
    print(f"Output -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
