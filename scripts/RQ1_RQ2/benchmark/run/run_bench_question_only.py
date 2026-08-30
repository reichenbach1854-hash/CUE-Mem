"""
Question-Only baseline 评测脚本（支持多次运行 + 交叉比对）。

只给 LLM 提供问题和选项，不注入任何历史对话记忆，用于检测选项是否存在偏置。

Stage 1 — 多次运行:
    每个模型跑 N 次（--num_runs），结果分别保存到 run1/ run2/ run3/ 子目录。
    若检测到旧版（无 runX/ 子目录）的结果，会自动迁移为 run1/。

用法:
    python -m scripts.RQ1_RQ2.benchmark.run.run_bench_question_only

    # 两个模型各跑 3 次（已有结果自动跳过）
    python run_bench_question_only.py \
        --llm_name gpt-5.4-mini qwen3.6-35b-a3b \
        --num_runs 3 --all_datasets --save_results

    # 只跑某个模型的第 2 次
    python run_bench_question_only.py \
        --llm_name qwen3.6-35b-a3b \
        --run_ids 2 --all_datasets --save_results

    # 单次运行（默认读取 data/dialog/base，结果写入 result_question_only/base）
    python run_bench_question_only.py \
        --llm_name gpt-5.4-mini \
        --data_name history_with_qa_p0 --save_results
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

from benchmark.paths import BENCHMARK_ROOT, DATA_ROOT, QUESTION_ONLY_RESULT_ROOT
from benchmark.security import redact_runtime_text, safe_runtime_error
from scripts.common.llm import message_content_to_text, openai_client

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(BENCHMARK_ROOT)

DATA_DIR = str(DATA_ROOT)
DIALOG_DIR = os.path.join(DATA_DIR, "dialog")
RESULT_DIR = str(QUESTION_ONLY_RESULT_ROOT)

BASE_DIALOG_DIR = os.path.join(DATA_DIR, "dialog", "base")

IMG_POINT_TYPES = {"pref_img", "entity_img", "rec_img"}
EXCLUDED_POINT_TYPES = {"adversarial_text"}


def _qa_point(qa):
    return str(qa.get("point") or qa.get("category") or "").lower()


def _filter_excluded_qas(qa_pairs, adversarial_only=False):
    if adversarial_only:
        kept = [qa for qa in qa_pairs if _qa_point(qa) in EXCLUDED_POINT_TYPES]
        skipped = len(qa_pairs) - len(kept)
        if skipped:
            print(f"  Kept {len(kept)} adversarial QA item(s); skipped {skipped} non-adversarial item(s).")
        return kept
    kept = [qa for qa in qa_pairs if _qa_point(qa) not in EXCLUDED_POINT_TYPES]
    skipped = len(qa_pairs) - len(kept)
    if skipped:
        print(f"  Excluded {skipped} adversarial QA item(s).")
    return kept


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
# System Prompt — 不提及任何记忆内容
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
你是一个选择题作答助手。你将看到一道单项选择题（A/B/C/D），请根据题目本身的信息作答。
必须从 A、B、C、D 中选择且只选择一个选项。
只输出选项字母（A、B、C 或 D），不要输出任何解释或额外文本。"""

FORMAT_CONSTRAINT = """\
请从 A、B、C、D 中选择一个最合理的选项。只输出选项字母，不要输出解释或额外文本。"""

FORMAT_CONSTRAINT_WITH_REASONING = """\
请从 A、B、C、D 中选择一个最合理的选项，并给出简要判断依据。
严格按以下格式输出：
第一行：只输出选项字母（A、B、C 或 D）。
第二行起：输出「依据：」后跟你的判断依据。

注意：这是 question-only baseline，你不能声称看过历史对话、图片或音频；只能基于题干和选项本身解释为什么该选项更可能被选中。"""


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


def _split_answer_reasoning(raw: str) -> tuple[str, str]:
    """Split a model response into final option and visible explanation."""
    if not raw or not raw.strip():
        return "", ""
    text = raw.strip()
    choice = _extract_choice(text)
    lines = text.splitlines()

    reasoning_lines = []
    first_choice_consumed = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not first_choice_consumed and _extract_choice(stripped) == choice:
            first_choice_consumed = True
            remainder = re.sub(r"^\s*([A-D])[\.\)、:：]?\s*", "", stripped, flags=re.IGNORECASE).strip()
            if remainder:
                reasoning_lines.append(remainder)
            continue
        reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()
    for prefix in ("依据：", "依据:", "理由：", "理由:", "判断依据：", "判断依据:"):
        if reasoning.startswith(prefix):
            reasoning = reasoning[len(prefix):].strip()
            break
    return choice, reasoning


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ---------------------------------------------------------------------------
# 正确率统计（复用原脚本逻辑）
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


def compute_answer_distribution(results):
    """统计模型回答的 A/B/C/D 分布。"""
    by_point = defaultdict(Counter)
    global_counter = Counter()
    for item in results:
        pred = _extract_choice(item.get("system_answer", ""))
        if pred in "ABCD":
            pt = item.get("category", "unknown")
            by_point[pt][pred] += 1
            global_counter[pred] += 1
    return global_counter, by_point


def print_accuracy_summary(summary):
    W = 72
    print(f"\n{'=' * W}")
    print("Accuracy Summary (Question-Only Baseline)")
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
    print("  (随机猜测基线 = 25.00%，显著高于此值说明选项存在偏置)")


def print_answer_distribution(global_counter, by_point):
    print(f"\n{'=' * 60}")
    print("模型回答分布 (Question-Only)")
    print(f"{'=' * 60}")
    for pt in sorted(by_point):
        c = by_point[pt]
        total = sum(c.values())
        dist = "  ".join(f"{o}:{c.get(o, 0):3d}" for o in "ABCD")
        print(f"  {pt:<20s}  n={total:4d}  {dist}")
    total = sum(global_counter.values())
    dist = "  ".join(f"{o}:{global_counter.get(o, 0):3d}" for o in "ABCD")
    print(f"  {'TOTAL':<20s}  n={total:4d}  {dist}")


# ---------------------------------------------------------------------------
# 增量保存 / 断点续跑
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 10

# ---------------------------------------------------------------------------
# 旧版结果 → run1/ 自动迁移
# ---------------------------------------------------------------------------
def _migrate_flat_results_to_run1(model_result_dir):
    """如果 model_result_dir 下直接存在 *_results.json（旧版平铺），
    将它们全部移入 run1/ 子目录。"""
    import shutil
    flat_files = [
        f for f in os.listdir(model_result_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(model_result_dir, f))
    ]
    if not flat_files:
        return
    run1_dir = os.path.join(model_result_dir, "run1")
    if os.path.isdir(run1_dir) and os.listdir(run1_dir):
        return
    os.makedirs(run1_dir, exist_ok=True)
    for f in flat_files:
        src = os.path.join(model_result_dir, f)
        dst = os.path.join(run1_dir, f)
        shutil.move(src, dst)
    print(f"  [migrate] moved {len(flat_files)} files → {run1_dir}")


def _atomic_save_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _qa_resume_key_from_result(record: dict) -> tuple[str, str]:
    return (
        str(record.get("qa_id", "")),
        str(record.get("category") or record.get("point") or ""),
    )


def _qa_resume_key_from_input(qa: dict, fallback_idx: int) -> tuple[str, str]:
    return (
        str(qa.get("qa_id", f"qa_{fallback_idx}")),
        str(qa.get("point") or qa.get("category") or ""),
    )



def _load_completed_results(path, require_reasoning=False):
    """加载已完成的结果用于断点续跑。

    只有 system_answer 非空的记录才视为"已完成"；
    API 调用失败导致 system_answer 为空的条目会被重新评测。
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        completed = {}
        skipped_empty = 0
        skipped_no_reasoning = 0
        skipped_no_category = 0
        for r in records:
            if "qa_id" not in r:
                continue
            if not r.get("system_answer", "").strip():
                skipped_empty += 1
                continue
            if require_reasoning and not str(r.get("reasoning", "")).strip():
                skipped_no_reasoning += 1
                continue
            key = _qa_resume_key_from_result(r)
            if not key[1]:
                skipped_no_category += 1
                continue
            completed[key] = r
        if skipped_empty:
            print(f"[resume] {skipped_empty} items have empty system_answer, will re-evaluate them.")
        if skipped_no_reasoning:
            print(f"[resume] {skipped_no_reasoning} items have no reasoning, will re-evaluate them.")
        if skipped_no_category:
            print(f"[resume] {skipped_no_category} items have no category/point, will re-evaluate them.")
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

    def call(self, question_prompt, with_reasoning=False):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT if not with_reasoning else (
                "你是一个选择题作答助手。你将看到一道单项选择题（A/B/C/D）。"
                "请先选出一个选项，并给出简要、可读的判断依据。"
                "不要声称你看过历史对话、图片或音频；这是 question-only baseline。"
            )},
            {"role": "user", "content": question_prompt},
        ]
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
                    raise safe_runtime_error("Question-only API request failed", e) from None


# ---------------------------------------------------------------------------
# 构建纯问题 prompt（无记忆注入）
# ---------------------------------------------------------------------------
def build_question_prompt(qa, image_option_mode="caption", img_captions_index=None, with_reasoning=False):
    """将 QA 条目转为纯问题 prompt，包含选项文本。"""
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

    prompt = question + "\n\n" + (FORMAT_CONSTRAINT_WITH_REASONING if with_reasoning else FORMAT_CONSTRAINT)
    return prompt, question


# ---------------------------------------------------------------------------
# 单数据集评测
# ---------------------------------------------------------------------------
def run_question_only(
    llm_client,
    llm_name,
    data_name,
    dialog_dir,
    result_dir,
    save_results=True,
    seed=42,
    sample=None,
    max_workers=8,
    qa_category=None,
    image_option_mode="caption",
    run_id=None,
    img_captions_index=None,
    with_reasoning=False,
    adversarial_only=False,
    result_subdir=None,
):
    sample_suffix = f"_sample{sample}" if sample else ""
    category_suffix = f"_{qa_category}" if qa_category else ""
    if run_id is not None:
        model_dir = os.path.join(result_dir, llm_name, f"run{run_id}")
    else:
        model_dir = os.path.join(result_dir, llm_name, result_subdir) if result_subdir else os.path.join(result_dir, llm_name)
    output_file = os.path.join(
        model_dir,
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

    qa_pairs = _filter_excluded_qas(dataset.get("human-annotated QAs", []), adversarial_only=adversarial_only)
    if not qa_pairs:
        label = "adversarial" if adversarial_only else "non-adversarial"
        print(f"  No {label} QA items in {data_name}, skip")
        return

    if qa_category:
        qa_cat_lower = qa_category.lower()
        original = len(qa_pairs)
        qa_pairs = [q for q in qa_pairs if _qa_point(q).startswith(qa_cat_lower)]
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

    all_qa_keys = {_qa_resume_key_from_input(qa, i) for i, qa in enumerate(qa_pairs)}
    completed_map = _load_completed_results(output_file, require_reasoning=with_reasoning) if save_results else {}
    if completed_map:
        before_filter = len(completed_map)
        completed_map = {k: v for k, v in completed_map.items() if k in all_qa_keys}
        removed_cached = before_filter - len(completed_map)
        if removed_cached:
            print(f"  [resume] ignored {removed_cached} cached adversarial/out-of-scope item(s).")
        print(f"  [resume] {len(completed_map)} items already done")

    if save_results and all_qa_keys and all_qa_keys.issubset(completed_map.keys()):
        print(f"  [skip] All {len(all_qa_keys)} items done, skipping.")
        results = list(completed_map.values())
        accuracy_summary = compute_accuracy_summary(results)
        print_accuracy_summary(accuracy_summary)
        _atomic_save_json(results, output_file)
        acc_file = output_file.replace("_results.json", "_accuracy.json")
        with open(acc_file, "w", encoding="utf-8") as f:
            json.dump(accuracy_summary, f, ensure_ascii=False, indent=2)
        return

    tasks = []
    for i, qa in enumerate(qa_pairs):
        qa_id = qa.get("qa_id", f"qa_{i}")
        resume_key = _qa_resume_key_from_input(qa, i)
        if resume_key in completed_map:
            continue
        prompt, enriched_question = build_question_prompt(
            qa,
            image_option_mode=image_option_mode,
            img_captions_index=img_captions_index,
            with_reasoning=with_reasoning,
        )
        tasks.append({
            "qa_id": qa_id,
            "prompt": prompt,
            "question": enriched_question,
            "original_answer": qa.get("answer", ""),
            "category": qa.get("point", ""),
            "qa_type": qa.get("qa_type", ""),
            "entity_explicitness": qa.get("entity_explicitness", ""),
            "entity_source_refs": qa.get("entity_source_refs", []),
            "entity_name": qa.get("entity_name", ""),
            "entity_anchor_lookup_name": qa.get("entity_anchor_lookup_name", ""),
            "sample_id": sample_id,
        })

    print(f"  {len(tasks)} items to evaluate, {len(completed_map)} cached")

    def _call_one(task):
        try:
            raw = llm_client.call(task["prompt"], with_reasoning=with_reasoning)
            if with_reasoning:
                answer, reasoning = _split_answer_reasoning(raw or "")
            else:
                answer = raw.strip() if raw else ""
                reasoning = ""
        except Exception as e:
            print(f"  Error [{task['qa_id']}]: {redact_runtime_text(e)}")
            answer = ""
            reasoning = ""
            raw = ""
        result = {
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
            "timestamp": get_timestamp(),
        }
        if with_reasoning:
            result["reasoning"] = reasoning
            result["raw_response"] = raw or ""
        return result

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
                desc=f"  Question-Only ({actual_workers}w)",
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

        global_dist, by_point_dist = compute_answer_distribution(results)
        print_answer_distribution(global_dist, by_point_dist)

        if save_results:
            acc_file = output_file.replace("_results.json", "_accuracy.json")
            with open(acc_file, "w", encoding="utf-8") as f:
                json.dump(accuracy_summary, f, ensure_ascii=False, indent=2)

    print(f"  Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# LLM 配置映射（复用原脚本）
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
    parser = argparse.ArgumentParser(description="Question-Only baseline（无记忆注入，支持多次运行）")
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
        help="图像选项处理方式：caption（注入文字描述）",
    )
    parser.add_argument(
        "--caption_category", type=str, default=None, nargs="+",
        choices=["base", "brief", "medium", "detailed"],
    )
    parser.add_argument("--audio_caption", type=str, default=None, nargs="+")
    # Stage 1: 多次运行
    parser.add_argument(
        "--num_runs", type=int, default=1,
        help="每个模型运行的次数（默认 1）。>1 时结果保存到 run1/ run2/ ... 子目录",
    )
    parser.add_argument(
        "--run_ids", type=int, nargs="+", default=None,
        help="只运行指定的 run ID（如 --run_ids 2 3）。不指定则运行 1..num_runs",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="LLM 采样温度（默认 0.0）。多次运行时建议设为 >0 以获得不同结果",
    )
    parser.add_argument(
        "--with_reasoning", "--with-reasoning",
        action="store_true",
        help="让 LLM 输出选项后附带简要判断依据，并保存到 results.json 的 reasoning/raw_response 字段",
    )
    parser.add_argument(
        "--adversarial_only",
        action="store_true",
        help="只评测 adversarial_text。用于补测拒答类 question-only 结果。",
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default=None,
        help="单次运行时写入模型目录下的子目录，例如 --result_subdir final。",
    )
    parser.add_argument("--api_key", default=None,
                        help="Runtime API key override; prefer CUE_MEM_LLM_API_KEY")
    parser.add_argument("--base_url", default=None,
                        help="Runtime API endpoint override; prefer CUE_MEM_LLM_BASE_URL")
    args = parser.parse_args()

    seed_everything(args.seed)

    if not args.all_datasets and args.data_name is None:
        print("Error: --data_name or --all_datasets required")
        sys.exit(1)

    # 从 base 数据集加载图像选项 caption（pref_img / entity_img / rec_img）
    print("Loading image option captions from base dataset ...")
    img_captions_index = load_base_img_captions()
    print(f"  Total: {len(img_captions_index)} qa_ids with image option captions")

    # 确定 run_id 列表
    if args.num_runs > 1:
        run_ids = args.run_ids or list(range(1, args.num_runs + 1))
        if args.temperature == 0.0:
            print("WARNING: temperature=0 且 num_runs>1，多次运行结果可能完全相同。"
                  "建议使用 --temperature 0.3 或更高。")
    else:
        run_ids = args.run_ids or [None]

    # 构建 (dialog_dir, result_dir, label) 列表
    all_configs = []
    for cap_cat in args.caption_category or []:
        all_configs.append((
            os.path.join(DATA_DIR, "dialog", cap_cat),
            os.path.join(RESULT_DIR, cap_cat),
            f"caption={cap_cat}",
        ))
    for audio_cap in args.audio_caption or []:
        all_configs.append((
            os.path.join(DATA_DIR, "dialog", "audio_caption", audio_cap),
            os.path.join(RESULT_DIR, "audio_caption", audio_cap),
            f"audio={audio_cap}",
        ))
    if not all_configs:
        all_configs.append((
            os.path.join(DATA_DIR, "dialog", "base"),
            os.path.join(RESULT_DIR, "base"),
            "base",
        ))

    qa_categories = args.qa_category or [None]
    combo_idx = 0

    for llm_name in args.llm_name:
        if args.api_key:
            os.environ["CUE_MEM_LLM_API_KEY"] = args.api_key
        if args.base_url:
            os.environ["CUE_MEM_LLM_BASE_URL"] = args.base_url
        api_key, api_base, model_name = _get_llm_config(llm_name)
        llm_client = LLMClient(api_key, api_base, model_name, temperature=args.temperature)

        for dialog_dir, result_dir, config_label in all_configs:
            print(f"\n[{config_label}] dialog_dir={dialog_dir}")

            # 多次运行时，自动迁移旧版平铺结果到 run1/
            if any(r is not None for r in run_ids):
                model_result_dir = os.path.join(result_dir, llm_name)
                if os.path.isdir(model_result_dir):
                    _migrate_flat_results_to_run1(model_result_dir)

            if args.all_datasets:
                datasets = get_available_datasets(dialog_dir)
                if not datasets:
                    print(f"  No datasets in {dialog_dir}")
                    continue
                print(f"  Found {len(datasets)} datasets")
            else:
                datasets = [args.data_name]

            for run_id in run_ids:
                run_label = f"run{run_id}" if run_id is not None else "single"
                for qa_category in qa_categories:
                    for data_name in datasets:
                        combo_idx += 1
                        cat_label = qa_category or "all"
                        print(f"\n{'=' * 70}")
                        print(f"[{combo_idx}] llm={llm_name}  {run_label}  "
                              f"config={config_label}  cat={cat_label}  data={data_name}")
                        print(f"{'=' * 70}")
                        try:
                            run_question_only(
                                llm_client=llm_client,
                                llm_name=llm_name,
                                data_name=data_name,
                                dialog_dir=dialog_dir,
                                result_dir=result_dir,
                                save_results=args.save_results,
                                seed=args.seed,
                                sample=args.sample,
                                max_workers=args.max_workers,
                                qa_category=qa_category,
                                image_option_mode=args.image_option_mode,
                                run_id=run_id,
                                img_captions_index=img_captions_index,
                                with_reasoning=args.with_reasoning,
                                adversarial_only=args.adversarial_only,
                                result_subdir=args.result_subdir,
                            )
                        except Exception as e:
                            print(f"Error: {data_name}: {redact_runtime_text(e)}")

    print(f"\n{'=' * 70}")
    print(f"Question-Only evaluation complete: {combo_idx} task(s)")
    print(f"{'=' * 70}")

