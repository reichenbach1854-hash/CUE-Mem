"""
Oracle Evidence 评测脚本。

不使用任何记忆系统，直接将 QA 对应的 memory clue / matched session 的原始对话
内容（scene_description、user_shared_image_description、background_audio_info、
完整 dialogue 含 background_audio）作为上下文喂给 LLM，评测模型在拥有"完美证据"
时的表现上界。

数据来源：
  - QA 题目：benchmark/data/dialog/base/history_with_qa_p{0,1,2}.json
  - 对话历史：qa/qa_formatted_data_000_002.json（按 p_id + session_id 索引）

用法:
    python -m scripts.RQ1_RQ2.benchmark.run.run_bench_oracle_evidence
    python run_bench_oracle_evidence.py \
        --llm_name gpt-5.4-mini \
        --all_datasets --save_results --max_workers 8

    # 只评测特定 point 类型
    python run_bench_oracle_evidence.py \
        --llm_name gpt-5.4-mini \
        --all_datasets --save_results --qa_category pref_text entity_text
"""

import json, random, re, sys, os, time, threading, argparse, warnings
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import concurrent.futures
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np

from benchmark.paths import (
    BENCHMARK_ROOT,
    DATA_ROOT,
    DIALOG_ROOT,
    ORACLE_RESULT_ROOT,
    PROMPT_ROOT,
    QA_ROOT,
    REPOSITORY_ROOT,
)
from benchmark.security import redact_runtime_text, safe_runtime_error
from scripts.common.llm import message_content_to_text, openai_client

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(BENCHMARK_ROOT)
REPO_ROOT = str(REPOSITORY_ROOT)

DATA_DIR = str(DATA_ROOT)
DIALOG_DIR = os.path.join(DATA_DIR, "dialog")
RESULT_DIR = str(ORACLE_RESULT_ROOT)

FORMATTED_DIALOG_PATH = os.path.join(str(QA_ROOT), "qa_formatted_data_000_019.json")

PROMPT_DIR = str(PROMPT_ROOT)

BASE_DIALOG_DIR = os.path.join(DATA_DIR, "dialog", "base")

IMG_POINT_TYPES = {"pref_img", "entity_img", "rec_img"}


def load_base_img_captions():
    """从 base 数据集的 history_with_qa_p*.json 中提取图像类题目的 option_captions，
    返回 {(qa_id, point): {"A": "...", "B": "...", ...}}。"""
    index = {}
    for fn in sorted(os.listdir(BASE_DIALOG_DIR)):
        if not fn.startswith("history_with_qa_p") or not fn.endswith(".json"):
            continue
        path = os.path.join(BASE_DIALOG_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {path}: {redact_runtime_text(e)}")
            continue
        for qa in data.get("human-annotated QAs", []):
            point = qa.get("point", "")
            if point not in IMG_POINT_TYPES:
                continue
            qa_id = qa.get("qa_id", "")
            captions = qa.get("option_captions")
            if qa_id and captions and isinstance(captions, dict):
                index[(qa_id, point)] = captions
    return index

# ---------------------------------------------------------------------------
# Prompt 加载（与 run_bench.py 保持一致）
# ---------------------------------------------------------------------------
_PROMPT_CACHE = {}

def load_prompt_file(filename):
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    prompt_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            _PROMPT_CACHE[filename] = content
            return content
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_path}")
        _PROMPT_CACHE[filename] = ""
        return ""
    except Exception as e:
        print(f"Error loading prompt file {filename}: {redact_runtime_text(e)}")
        _PROMPT_CACHE[filename] = ""
        return ""

SystemPrompt = load_prompt_file("sys_prompt.txt")
if not SystemPrompt:
    SystemPrompt = """Your task is to answer questions in a concise manner with the help of memory content.
When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"Mental health.\", as this is more concise.
"""

TextMsgPrompt = """
The retrieved memory contents are as follows:

{memory_context}
"""

DialogueAgentPrompt = """
Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} in a concise manner with the help of memory content.
Please only provide the content of the answer, without including introductory phrases like 'answer:'.
For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.
Generate answers primarily concise, yet complete enough to accurately answer the questions.

The current question is as follows:
{observation} {format_constraint}
"""

_FC_MAP = {
    "ENTITY": "entity_prompt.txt", "PREFERENCE": "preference_prompt.txt",
    "PREFERENCE_SAME_CATEGORY": "preference_prompt.txt",
    "PREFERENCE_CROSS_CATEGORY": "preference_prompt.txt",
    "RECOMMENDATION": "recommendation_prompt.txt",
    "RECOMMENDATION_SAME_CATEGORY": "recommendation_prompt.txt",
    "RECOMMENDATION_CROSS_CATEGORY": "recommendation_prompt.txt",
    "REFUSAL": "refusal_prompt.txt",
    "PREF_IMG": "preference_prompt.txt",
    "PREF_TEXT": "preference_prompt.txt",
    "REC_IMG": "recommendation_prompt.txt",
    "REC_TEXT": "recommendation_prompt.txt",
    "ENTITY_IMG": "entity_prompt.txt",
    "ENTITY_TEXT": "entity_prompt.txt",
    "REFUSAL_TEXT": "refusal_prompt.txt",
    "AUDIO_CONTEXT": "preference_prompt.txt",
}


def get_format_constraint(category):
    """根据 category 加载对应的 format constraint（与 run_bench.py 一致）。"""
    if not category:
        return None
    fc_file = _FC_MAP.get(category.upper())
    if fc_file:
        return load_prompt_file(fc_file)
    return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    print(f"Global seed set to {seed}")


def get_available_datasets(dialog_dir):
    datasets = []
    if os.path.exists(dialog_dir):
        for fn in os.listdir(dialog_dir):
            if fn.endswith(".json") and "_results_" not in fn and "_evaluate_result_" not in fn:
                datasets.append(fn.replace(".json", ""))
    return sorted(datasets)


def _extract_choice(answer: str) -> str:
    if not answer:
        return ""
    s = answer.strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = re.search(r"\b([A-D])\b", s.upper())
    return m.group(1) if m else s.upper()


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _split_answer_reasoning(raw):
    """将 LLM 原始输出拆分为选项字母、推理依据和依据 session 日期列表。"""
    if not raw or not raw.strip():
        return raw, "", []
    lines = raw.strip().splitlines()
    choice_line = lines[0].strip()

    reasoning_sessions = []
    reasoning_lines = []
    for line in lines[1:]:
        stripped = line.strip()
        for prefix in ("依据session：", "依据session:", "依据 session：", "依据 session:"):
            if stripped.lower().startswith(prefix.lower()):
                session_str = stripped[len(prefix):].strip()
                reasoning_sessions = [s.strip() for s in re.split(r'[,，;；\s]+', session_str) if s.strip()]
                break
        else:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    for prefix in ("依据：", "依据:", "判断依据：", "判断依据:"):
        if reasoning.startswith(prefix):
            reasoning = reasoning[len(prefix):].strip()
            break
    return choice_line, reasoning, reasoning_sessions


# ---------------------------------------------------------------------------
# 加载 formatted dialog 并建索引
# ---------------------------------------------------------------------------
def load_formatted_dialog(path):
    """加载 qa_formatted_data_000_002.json，返回 {p_id: {session_id: event}}。"""
    with open(path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    index = {}
    for profile in profiles:
        p_id = profile.get("p_id", 0)
        session_map = {}
        for event in profile.get("events", []):
            sid = event.get("session_id", "")
            if sid:
                session_map[sid] = event
        index[p_id] = session_map
    return index


# ---------------------------------------------------------------------------
# 从 QA 的 clue + session_id 字段提取所需的 session IDs
# ---------------------------------------------------------------------------
def derive_session_ids(qa):
    """从 clue 和 session_id 字段导出关联的 session ID 集合。"""
    sessions = set()
    # 1) 从 clue 中提取 (如 "D15:00" → "D15", "D15-001.wav" → "D15")
    for c in qa.get("clue", []) or []:
        m = re.match(r"(D\d+)", c)
        if m:
            sessions.add(m.group(1))
    # 2) 从 matched_session_ids 字段（如果存在）
    for sid in qa.get("matched_session_ids", []) or []:
        if isinstance(sid, str) and sid:
            sessions.add(sid)
    # 3) 从 session_id 字段
    sid = qa.get("session_id", "")
    if isinstance(sid, str) and sid:
        sessions.add(sid)
    return sorted(sessions)


# ---------------------------------------------------------------------------
# 将 session event 渲染为结构化文本（复用 gen_qa 的格式）
# ---------------------------------------------------------------------------
def format_session_text(event):
    """将一个 event（session）渲染为包含完整对话的结构化文本。"""
    session_id = event.get("session_id", "?")
    scene = (event.get("scene_description") or "").strip()
    img_desc = (event.get("user_shared_image_description") or "").strip()
    bg_audio_info = (event.get("background_audio_info") or "").strip()
    speech = (event.get("human_speech_content") or "").strip()

    lines = [f"=== Session {session_id} ==="]
    if scene:
        lines.append(f"场景: {scene}")
    if img_desc and img_desc.lower() != "none":
        lines.append(f"图像总览: {img_desc}")
    if bg_audio_info and bg_audio_info.lower() != "none":
        lines.append(f"背景音频: {bg_audio_info}")
    if speech and speech.lower() != "none":
        lines.append(f"用户语音: {speech}")

    # 建 round → background_audio 映射
    bg_audio_map = {}
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

    lines.append("对话:")
    for turn in dialog_list:
        round_id = turn.get("round", "?")
        user_text = (turn.get("user") or "").strip()
        assistant_text = (turn.get("assistant") or "").strip()
        lines.append(f"  [{round_id}] User: {user_text}")
        if round_id in bg_audio_map:
            lines.append(f"    └─ 背景音频: {bg_audio_map[round_id]}")
        for k, v in turn.items():
            if k in ("round", "user", "assistant"):
                continue
            if k.endswith(".png"):
                lines.append(f"    └─ 图像[{k}]: {v}")
            elif k.endswith(".wav"):
                lines.append(f"    └─ 音频[{k}]: {v}")
        if assistant_text:
            lines.append(f"  [{round_id}] Assistant: {assistant_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 构建 oracle evidence prompt
# ---------------------------------------------------------------------------
def build_oracle_messages(qa, session_index, speaker_a, speaker_b, image_option_mode="caption", with_reasoning=False, img_captions_index=None):
    """构建完整的 messages list，格式与 run_bench.py 一致。"""
    p_id = qa.get("p_id", 0)
    session_ids = derive_session_ids(qa)

    # 渲染相关 session 的对话文本
    evidence_parts = []
    pid_sessions = session_index.get(p_id, {})
    for sid in session_ids:
        event = pid_sessions.get(sid)
        if event:
            evidence_parts.append(format_session_text(event))

    if evidence_parts:
        evidence_text = "\n\n".join(evidence_parts)
    else:
        evidence_text = "（无相关对话历史）"

    # 构建问题文本（含图像选项 caption 注入，与 run_bench.py 一致）
    question = qa.get("question", "")
    # 优先从 base 数据集获取图像选项描述
    qa_id = qa.get("qa_id", "")
    point = qa.get("point", "")
    option_captions = None
    if img_captions_index and (qa_id, point) in img_captions_index:
        option_captions = img_captions_index[(qa_id, point)]
    if not option_captions:
        option_captions = qa.get("option_captions") or qa.get("question_image_descriptions")
    if option_captions and isinstance(option_captions, dict) and any(option_captions.get(l) for l in "ABCD"):
        if image_option_mode == "caption":
            desc_lines = []
            for letter in "ABCD":
                desc = option_captions.get(letter, "")
                if desc:
                    desc_lines.append(f"{letter}. {desc}")
            if desc_lines:
                insert_text = "\n".join(desc_lines)
                suffix = "请在 A/B/C/D 中选择最符合的选项。"
                q_stripped = question.rstrip()
                if q_stripped.endswith(suffix):
                    question = q_stripped[: -len(suffix)].rstrip() + "\n\n" + insert_text + "\n\n" + suffix
                else:
                    question = q_stripped + "\n\n" + insert_text

    # 加载 category 对应的 format constraint（与 run_bench.py 一致）
    category = qa.get("point", "")
    format_constraint = get_format_constraint(category)

    if with_reasoning and format_constraint:
        format_constraint = (
            format_constraint.rstrip()
            + "\n\n"
            + "## 输出格式\n"
            + "第一行输出你选择的选项字母（A、B、C 或 D）。\n"
            + "第二行输出「依据session：」后跟你做出判断所依赖的所有记忆的 session 日期（即记忆中 timestamp 字段的值），用逗号分隔。\n"
            + "第三行起输出你的判断依据：结合上述 session 日期，说明你是如何从对话文本、图片描述还是语音消息中推断出用户的偏好或习惯，从而选出该选项的。\n"
            + "示例：\n"
            + "A\n"
            + "依据session：2024-03-15, 2024-04-02\n"
            + "依据：在 2024-03-15 的对话中，用户通过语音消息提到喜欢清淡口味；在 2024-04-02 的对话中，用户分享了一张日式料理的图片并表达了喜爱，因此选择A最符合用户的饮食偏好。"
        )

    format_constraint_str = ""
    if format_constraint:
        format_constraint_str = "\n\n" + format_constraint

    # 构建 evidence prompt（对应 run_bench.py 的 TextMsgPrompt）
    evidence_prompt = TextMsgPrompt.format(memory_context=evidence_text)

    # 构建 query prompt（对应 run_bench.py 的 DialogueAgentPrompt）
    query_prompt = DialogueAgentPrompt.format(
        observation=question,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint_str,
    )

    messages = [
        {"role": "system", "content": SystemPrompt},
        {"role": "user", "content": [
            {"type": "text", "text": evidence_prompt},
            {"type": "text", "text": query_prompt},
        ]},
    ]
    return messages, len(session_ids), len(evidence_parts), question, session_ids, evidence_parts, evidence_text


# ---------------------------------------------------------------------------
# 正确率统计
# ---------------------------------------------------------------------------
def compute_accuracy_summary(results):
    ENTITY_EXPLICIT_TYPES = {"Relationship", "Pets"}

    def get_explicitness(item):
        cat = (item.get("category") or "").lower()
        qa_type = item.get("qa_type", "")
        if cat in ("entity", "entity_img", "entity_text"):
            if qa_type in ENTITY_EXPLICIT_TYPES:
                return "explicit"
            if qa_type == "Items":
                return item.get("entity_explicitness") or "unknown"
            return "unknown"
        if qa_type in ("explicit", "implicit"):
            return qa_type
        return "unknown"

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for item in results:
        cat = (item.get("category") or "").lower()
        pred = _extract_choice(item.get("system_answer", ""))
        gold = _extract_choice(item.get("original_answer", ""))
        correct = 1 if pred == gold else 0
        explicitness = get_explicitness(item)
        stats[cat]["overall"][0] += correct
        stats[cat]["overall"][1] += 1
        stats[cat][explicitness][0] += correct
        stats[cat][explicitness][1] += 1

    all_correct = sum(v["overall"][0] for v in stats.values())
    all_total = sum(v["overall"][1] for v in stats.values())

    def make_entry(c, t):
        return {"correct": c, "total": t, "accuracy": round(c / t * 100, 2) if t > 0 else 0.0}

    summary = {}
    for cat in sorted(stats):
        cat_entry = {}
        for split in ["overall", "explicit", "implicit", "mixed", "unknown"]:
            c, t = stats[cat][split]
            if t > 0:
                cat_entry[split] = make_entry(c, t)
        summary[cat] = cat_entry
    summary["__overall__"] = {"overall": make_entry(all_correct, all_total)}
    return summary


def print_accuracy_summary(summary):
    W = 72
    print(f"\n{'=' * W}")
    print("Accuracy Summary (Oracle Evidence)")
    print(f"{'=' * W}")
    print(f"{'Category':<20} {'Split':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-' * W}")
    for cat in sorted(k for k in summary if k != "__overall__"):
        first = True
        for split in ["overall", "explicit", "implicit", "mixed", "unknown"]:
            if split not in summary[cat]:
                continue
            d = summary[cat][split]
            label = cat if first else ""
            first = False
            print(f"{label:<20} {split:<12} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>9.2f}%")
        print(f"{'-' * W}")
    if "__overall__" in summary:
        d = summary["__overall__"]["overall"]
        print(f"{'TOTAL':<20} {'overall':<12} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>9.2f}%")
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# 增量保存 / 断点续跑
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 10


def _atomic_save_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_completed_results(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        completed = {}
        skipped_empty = 0
        for r in records:
            if "qa_id" not in r:
                continue
            if not r.get("system_answer", "").strip():
                skipped_empty += 1
                continue
            completed[r["qa_id"]] = r
        if skipped_empty:
            print(f"[resume] {skipped_empty} items have empty system_answer, will re-evaluate them.")
        return completed
    except Exception as e:
        print(f"WARN: failed to load checkpoint {path}: {redact_runtime_text(e)}")
        return {}


# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, api_key, api_base, model_name, temperature=0.0):
        self.client = openai_client(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.temperature = temperature

    def call(self, messages):
        """调用 LLM，接受完整的 messages list（与 run_bench.py 的 VLMAgent.run 一致）。"""
        api_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        max_retries = 5
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(**api_params)
                return message_content_to_text(resp.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  API retry {attempt + 1}/{max_retries}: {redact_runtime_text(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise safe_runtime_error("Oracle evidence API request failed", e) from None


# ---------------------------------------------------------------------------
# 单数据集评测
# ---------------------------------------------------------------------------
def run_oracle_evidence(
    llm_client,
    llm_name,
    data_name,
    dialog_dir,
    result_dir,
    session_index,
    save_results=True,
    seed=42,
    sample=None,
    max_workers=8,
    qa_category=None,
    image_option_mode="caption",
    with_reasoning=False,
    img_captions_index=None,
):
    sample_suffix = f"_sample{sample}" if sample else ""
    category_suffix = f"_{qa_category}" if qa_category else ""
    output_file = os.path.join(
        result_dir, llm_name,
        f"{data_name}{sample_suffix}{category_suffix}_results.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    json_path = os.path.join(dialog_dir, f"{data_name}.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {data_name} from {json_path}")
    except FileNotFoundError:
        print(f"Not found: {json_path}")
        return
    except Exception as e:
        print(f"Error: {redact_runtime_text(e)}")
        return

    character_profile = dataset.get("character_profile", {})
    sample_id = character_profile.get("name", data_name)

    # speaker 名称（与 run_bench.py 一致）
    if character_profile and character_profile.get("name"):
        speaker_a = f"user ({character_profile.get('name')})"
    else:
        speaker_a = "user"
    speaker_b = "assistant"

    qa_pairs = dataset.get("human-annotated QAs", [])
    if not qa_pairs:
        print(f"  No QA items in {data_name}, skip")
        return

    if qa_category:
        qa_cat_lower = qa_category.lower()
        original = len(qa_pairs)
        qa_pairs = [q for q in qa_pairs if (q.get("point", "") or "").lower().startswith(qa_cat_lower)]
        print(f"  Category filter '{qa_category}': {len(qa_pairs)}/{original}")
        if not qa_pairs:
            return

    if sample and sample > 0:
        counts = {}
        sampled = []
        for qa in qa_pairs:
            cat = qa.get("point", "unknown")
            if counts.get(cat, 0) >= sample:
                continue
            sampled.append(qa)
            counts[cat] = counts.get(cat, 0) + 1
        print(f"  Sampled: {len(sampled)}/{len(qa_pairs)}")
        qa_pairs = sampled

    completed_map = _load_completed_results(output_file) if save_results else {}
    if completed_map:
        print(f"  [resume] {len(completed_map)} items already done")

    all_qa_ids = {qa.get("qa_id", f"qa_{i}") for i, qa in enumerate(qa_pairs)}
    if save_results and all_qa_ids and all_qa_ids.issubset(completed_map.keys()):
        print(f"  [skip] All {len(all_qa_ids)} items done, skipping.")
        results = list(completed_map.values())
        accuracy_summary = compute_accuracy_summary(results)
        print_accuracy_summary(accuracy_summary)
        return

    # 构建任务
    tasks = []
    no_evidence_count = 0
    for i, qa in enumerate(qa_pairs):
        qa_id = qa.get("qa_id", f"qa_{i}")
        if qa_id in completed_map:
            continue
        messages, n_session_ids, n_matched, enriched_question, evidence_session_ids, evidence_parts, evidence_text = build_oracle_messages(
            qa, session_index, speaker_a, speaker_b, image_option_mode,
            with_reasoning=with_reasoning, img_captions_index=img_captions_index)
        if n_matched == 0:
            no_evidence_count += 1
        tasks.append({
            "qa_id": qa_id,
            "messages": messages,
            "question": enriched_question,
            "original_answer": qa.get("answer", ""),
            "category": qa.get("point", ""),
            "qa_type": qa.get("qa_type", ""),
            "entity_explicitness": qa.get("entity_explicitness", ""),
            "entity_source_refs": qa.get("entity_source_refs", []),
            "entity_name": qa.get("entity_name", ""),
            "entity_anchor_lookup_name": qa.get("entity_anchor_lookup_name", ""),
            "sample_id": sample_id,
            "n_evidence_sessions": n_matched,
            "evidence_session_ids": evidence_session_ids,
            "related_history": evidence_parts,
            "related_history_text": evidence_text,
        })

    print(f"  {len(tasks)} items to evaluate, {len(completed_map)} cached")
    if no_evidence_count:
        print(f"  WARNING: {no_evidence_count} items have no matching evidence sessions")

    def _call_one(task):
        try:
            raw = llm_client.call(task["messages"])
            answer = raw.strip() if raw else ""
        except Exception as e:
            print(f"  Error [{task['qa_id']}]: {redact_runtime_text(e)}")
            answer = ""

        reasoning = ""
        reasoning_sessions = []
        if with_reasoning and answer:
            answer, reasoning, reasoning_sessions = _split_answer_reasoning(answer)

        result_item = {
            "qa_id": task["qa_id"],
            "sample_id": task["sample_id"],
            "question": task["question"],
            "system_answer": answer,
            "original_answer": task["original_answer"],
            "category": task["category"],
            "qa_type": task["qa_type"],
            "entity_explicitness": task.get("entity_explicitness", ""),
            "entity_source_refs": task.get("entity_source_refs", []),
            "entity_name": task.get("entity_name", ""),
            "entity_anchor_lookup_name": task.get("entity_anchor_lookup_name", ""),
            "n_evidence_sessions": task["n_evidence_sessions"],
            "evidence_session_ids": task["evidence_session_ids"],
            "related_history": task["related_history"],
            "related_history_text": task["related_history_text"],
            "timestamp": get_timestamp(),
        }
        if with_reasoning:
            result_item["reasoning"] = reasoning
            result_item["reasoning_sessions"] = reasoning_sessions
        return result_item

    new_results = []
    lock = threading.Lock()
    t0 = time.time()

    if tasks:
        actual_workers = min(max_workers, len(tasks))
        print(f"  LLM calls: {len(tasks)} items × {actual_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {pool.submit(_call_one, t): t for t in tasks}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(tasks),
                desc=f"  OracleEvidence ({actual_workers}w)",
            ):
                result = future.result()
                with lock:
                    new_results.append(result)
                    if save_results and len(new_results) % CHECKPOINT_EVERY == 0:
                        all_so_far = list(completed_map.values()) + new_results
                        _atomic_save_json(all_so_far, output_file)

    elapsed = time.time() - t0
    results = list(completed_map.values()) + new_results

    if save_results and results:
        _atomic_save_json(results, output_file)
        print(f"  Saved {len(results)} results to {output_file}")

    if results:
        accuracy_summary = compute_accuracy_summary(results)
        print_accuracy_summary(accuracy_summary)

        if save_results:
            acc_file = output_file.replace("_results.json", "_accuracy.json")
            with open(acc_file, "w", encoding="utf-8") as f:
                json.dump(accuracy_summary, f, ensure_ascii=False, indent=2)

    print(f"  Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# LLM 配置映射
# ---------------------------------------------------------------------------
def _get_llm_config(llm_name):
    supported = {
        "qwen2-5-7b", "qwen2-5-vl-3b", "qwen2-5-vl-7b", "qwen2-5-vl-32b",
        "gpt-5.4-mini", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        "qwen3.6-35b-a3b",
    }
    if llm_name not in supported:
        raise ValueError(f"Unsupported LLM: {llm_name}")
    api_key = os.environ.get("CUE_MEM_VLLM_API_KEY") or os.environ.get("CUE_MEM_LLM_API_KEY")
    api_base = os.environ.get("CUE_MEM_VLLM_BASE_URL") or os.environ.get("CUE_MEM_LLM_BASE_URL")
    model_name = os.environ.get("CUE_MEM_LLM_MODEL", "").strip() or llm_name
    if llm_name == "qwen3.6-35b-a3b" and not os.environ.get("CUE_MEM_LLM_MODEL"):
        model_name = "qwen36-35b-a3b-fp8"
    return api_key, api_base, model_name


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Oracle Evidence 评测（注入 memory clue 对应的完整 session 对话）")
    parser.add_argument(
        "--llm_name", default=["gpt-5.4-mini"], nargs="+",
        choices=[
            "qwen2-5-7b", "qwen2-5-vl-3b", "qwen2-5-vl-7b", "qwen2-5-vl-32b",
            "gpt-5.4-mini", "gemini-2.5-flash", "gemini-2.5-flash-lite", "qwen3.6-35b-a3b",
        ],
    )
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--all_datasets", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument(
        "--qa_category", type=str, nargs="+", default=None,
        help="只评测特定 point 类型，如 pref_text entity_img",
    )
    parser.add_argument(
        "--image_option_mode", type=str, default="caption", choices=["caption"],
    )
    parser.add_argument(
        "--formatted_dialog", type=str, default=FORMATTED_DIALOG_PATH,
        help="qa_formatted_data_000_019.json 路径",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
    )
    parser.add_argument(
        "--with_reasoning", action="store_true",
        help="让 LLM 同时输出判断依据，保存到结果文件的 reasoning 字段",
    )
    parser.add_argument("--api_key", default=None,
                        help="Runtime API key override; prefer CUE_MEM_LLM_API_KEY")
    parser.add_argument("--base_url", default=None,
                        help="Runtime API endpoint override; prefer CUE_MEM_LLM_BASE_URL")
    args = parser.parse_args()

    if args.api_key:
        os.environ["CUE_MEM_LLM_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["CUE_MEM_LLM_BASE_URL"] = args.base_url

    seed_everything(args.seed)

    if not args.all_datasets and args.data_name is None:
        print("Error: --data_name or --all_datasets required")
        sys.exit(1)

    # 加载 formatted dialog 并建立 session 索引
    print(f"Loading formatted dialog from {args.formatted_dialog} ...")
    session_index = load_formatted_dialog(args.formatted_dialog)
    total_sessions = sum(len(v) for v in session_index.values())
    print(f"  {len(session_index)} profiles, {total_sessions} sessions indexed")

    # 从 base 数据集加载图像选项 caption（pref_img / entity_img / rec_img）
    print("Loading image option captions from base dataset ...")
    img_captions_index = load_base_img_captions()
    print(f"  Total: {len(img_captions_index)} qa_ids with image option captions")

    dialog_dir = os.path.join(DATA_DIR, "dialog", "base")
    result_dir = os.path.join(RESULT_DIR, "base")

    if args.all_datasets:
        datasets = get_available_datasets(dialog_dir)
        if not datasets:
            print(f"  No datasets in {dialog_dir}")
            sys.exit(1)
        print(f"  Found {len(datasets)} datasets")
    else:
        datasets = [args.data_name]

    qa_categories = args.qa_category or [None]
    combo_idx = 0

    for llm_name in args.llm_name:
        api_key, api_base, model_name = _get_llm_config(llm_name)
        llm_client = LLMClient(api_key, api_base, model_name, temperature=args.temperature)

        for qa_category in qa_categories:
            for data_name in datasets:
                combo_idx += 1
                cat_label = qa_category or "all"
                print(f"\n{'=' * 70}")
                print(f"[{combo_idx}] llm={llm_name}  cat={cat_label}  data={data_name}")
                print(f"{'=' * 70}")
                try:
                    run_oracle_evidence(
                        llm_client=llm_client,
                        llm_name=llm_name,
                        data_name=data_name,
                        dialog_dir=dialog_dir,
                        result_dir=result_dir,
                        session_index=session_index,
                        save_results=args.save_results,
                        seed=args.seed,
                        sample=args.sample,
                        max_workers=args.max_workers,
                        qa_category=qa_category,
                        image_option_mode=args.image_option_mode,
                        with_reasoning=args.with_reasoning,
                        img_captions_index=img_captions_index,
                    )
                except Exception as e:
                    print(f"Error: {data_name}: {redact_runtime_text(e)}")

    print(f"\n{'=' * 70}")
    print(f"Oracle Evidence evaluation complete: {combo_idx} task(s)")
    print(f"{'=' * 70}")
