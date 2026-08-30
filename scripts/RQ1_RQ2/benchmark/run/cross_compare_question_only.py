"""Question-only 结果交叉比较与 retention 集合分析。

默认模式（``cross-compare``）扫描
``result_question_only/base/{model}/run{N}/`` 下的 ``*_results.json``，按
``profile + category + qa_id`` 汇总每道题在各次运行中的正误，并生成
``correct_0.json`` 至 ``correct_N.json`` 以及 ``summary.json``。

``retention`` 模式读取上述 ``cross_compare`` 目录，逐步累加
``correct_0..correct_k``，计算每个 retention 集合的分类准确率、显式/隐式
拆分和文本/图像合并结果，输出终端表格并可保存 JSON 报告。

用法（从仓库根目录执行）：
    python scripts/RQ1_RQ2/benchmark/run/cross_compare_question_only.py
    python scripts/RQ1_RQ2/benchmark/run/cross_compare_question_only.py \
        --mode cross-compare --result_dir benchmark/result_question_only/base
    python scripts/RQ1_RQ2/benchmark/run/cross_compare_question_only.py \
        --mode retention --cross_compare_dir \
        benchmark/result_question_only/base/cross_compare \
        --output_json benchmark/result_question_only/base/cross_compare/retention_accuracy.json

``cross-compare`` 是默认模式，因此原交叉比较脚本的参数和输出保持兼容；
``retention`` 模式兼容原 retention 分析工具的参数。
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.paths import QUESTION_ONLY_RESULT_ROOT
from benchmark.security import redact_runtime_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(QUESTION_ONLY_RESULT_ROOT.parent)
DEFAULT_RESULT_DIR = os.path.join(str(QUESTION_ONLY_RESULT_ROOT), "base")
DEFAULT_CROSS_COMPARE_DIR = os.path.join(DEFAULT_RESULT_DIR, "cross_compare")

_CATEGORY_ALIASES = {
    "preference_same_category": "pref_img",
    "recommendation_same_category": "rec_img",
    "entity": "entity_img",
}


def _extract_choice(answer: str) -> str:
    if not answer:
        return ""
    s = answer.strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = re.search(r"\b([A-D])\b", s.upper())
    return m.group(1) if m else s.upper()


def discover_runs(result_dir, models=None):
    """扫描 result_dir 下的 model/runN/ 结构，返回 [(model, run_id, run_dir)] 列表。"""
    runs = []
    if not os.path.isdir(result_dir):
        return runs
    for model_name in sorted(os.listdir(result_dir)):
        if models and model_name not in models:
            continue
        model_dir = os.path.join(result_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for entry in sorted(os.listdir(model_dir)):
            m = re.match(r"^run(\d+)$", entry)
            if not m:
                continue
            run_id = int(m.group(1))
            run_dir = os.path.join(model_dir, entry)
            if os.path.isdir(run_dir):
                runs.append((model_name, run_id, run_dir))
    return runs


def _source_name_from_result_file(fname):
    """history_with_qa_p0_pref_img_results.json -> history_with_qa_p0_pref_img"""
    suffix = "_results.json"
    return fname[:-len(suffix)] if fname.endswith(suffix) else os.path.splitext(fname)[0]


def _profile_from_source_name(source_name):
    m = re.match(r"^(history_with_qa_p\d+)", source_name)
    return m.group(1) if m else source_name


def _normalize_category(category):
    return _CATEGORY_ALIASES.get(category, category)


def _make_compare_key(profile_name, category, qa_id):
    return f"{profile_name}::{category}::{qa_id}"


def load_run_results(run_dir):
    """加载一个 runN/ 目录下所有 _results.json，返回 {compare_key: record}。"""
    qa_map = {}
    duplicate_keys = 0
    for fname in sorted(os.listdir(run_dir)):
        if not fname.endswith("_results.json"):
            continue
        source_name = _source_name_from_result_file(fname)
        profile_name = _profile_from_source_name(source_name)
        fpath = os.path.join(run_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                qa_id = r.get("qa_id")
                if qa_id:
                    category = _normalize_category(r.get("category", "") or r.get("point", ""))
                    compare_key = _make_compare_key(profile_name, category, qa_id)
                    if compare_key in qa_map:
                        duplicate_keys += 1
                        # 同一个 run 中可能同时存在按类别拆分文件和旧的全量结果文件。
                        # 保留先读到的分类结果，避免旧文件覆盖已拆分结果。
                        continue
                    enriched = dict(r)
                    enriched["_compare_key"] = compare_key
                    enriched["_source_file"] = fname
                    enriched["_source_name"] = source_name
                    enriched["_profile_name"] = profile_name
                    enriched["category"] = category
                    qa_map[compare_key] = enriched
        except (OSError, TypeError, ValueError) as e:
            print(f"WARN: skip {fpath}: {redact_runtime_text(e)}")
    if duplicate_keys:
        print(f"WARN: {run_dir} has {duplicate_keys} duplicated compare keys")
    return qa_map


def cross_compare(result_dir, models=None, num_runs=None, include_missing=False):
    runs = discover_runs(result_dir, models)
    if num_runs:
        runs = [(m, r, d) for m, r, d in runs if r <= num_runs]

    if not runs:
        print("No runs found. Check --result_dir and run directories.")
        sys.exit(1)

    run_labels = [f"{m}/run{r}" for m, r, _ in runs]
    total_runs = len(runs)
    print(f"Found {total_runs} runs:")
    for label in run_labels:
        print(f"  - {label}")

    # 加载所有 run 的结果
    all_results = []
    for model, run_id, run_dir in runs:
        qa_map = load_run_results(run_dir)
        print(f"  {model}/run{run_id}: {len(qa_map)} QAs loaded")
        all_results.append((model, run_id, qa_map))

    # 收集 compare_key。qa_id 在不同 point/profile 中会复用，不能单独作为唯一键。
    key_sets = [set(qa_map.keys()) for _, _, qa_map in all_results]
    if include_missing:
        all_qa_keys = set().union(*key_sets)
        print("\nCompare scope: union of all runs (missing results count as incorrect)")
    else:
        all_qa_keys = set.intersection(*key_sets)
        dropped = len(set().union(*key_sets)) - len(all_qa_keys)
        print("\nCompare scope: intersection of all runs")
        if dropped:
            print(f"  Dropped {dropped} QA records missing from at least one run")
    all_qa_keys = sorted(all_qa_keys)
    print(f"\nTotal unique QA records: {len(all_qa_keys)}")

    # 逐题汇总
    qa_summary = {}
    for qa_key in all_qa_keys:
        per_run = []
        correct_count = 0
        qa_id = ""
        source_file = ""
        source_name = ""
        profile_name = ""
        question = ""
        gold_answer = ""
        category = ""
        qa_type = ""
        entity_explicitness = ""
        entity_source_refs = []
        entity_name = ""
        entity_anchor_lookup_name = ""

        for model, run_id, qa_map in all_results:
            record = qa_map.get(qa_key)
            if record is None:
                per_run.append({
                    "model": model,
                    "run": run_id,
                    "answer": "",
                    "correct": False,
                    "missing": True,
                })
                continue

            if not question:
                qa_id = record.get("qa_id", "")
                source_file = record.get("_source_file", "")
                source_name = record.get("_source_name", "")
                profile_name = record.get("_profile_name", "")
                question = record.get("question", "")
                gold_answer = _extract_choice(record.get("original_answer", ""))
                category = record.get("category", "")
                qa_type = record.get("qa_type", "")
                entity_explicitness = record.get("entity_explicitness", "")
                entity_source_refs = record.get("entity_source_refs", [])
                entity_name = record.get("entity_name", "")
                entity_anchor_lookup_name = record.get("entity_anchor_lookup_name", "")

            pred = _extract_choice(record.get("system_answer", ""))
            is_correct = pred == gold_answer and pred != ""
            if is_correct:
                correct_count += 1
            per_run.append({
                "model": model,
                "run": run_id,
                "answer": pred,
                "correct": is_correct,
            })

        qa_summary[qa_key] = {
            "compare_key": qa_key,
            "qa_id": qa_id,
            "source_file": source_file,
            "source_name": source_name,
            "profile_name": profile_name,
            "question": question,
            "gold_answer": gold_answer,
            "category": category,
            "qa_type": qa_type,
            "entity_explicitness": entity_explicitness,
            "entity_source_refs": entity_source_refs,
            "entity_name": entity_name,
            "entity_anchor_lookup_name": entity_anchor_lookup_name,
            "correct_count": correct_count,
            "total_runs": total_runs,
            "per_run": per_run,
        }

    return qa_summary, total_runs, run_labels


def output_by_correct_count(qa_summary, total_runs, run_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 按 correct_count 分组
    by_count = defaultdict(list)
    for qa_key in sorted(qa_summary):
        entry = qa_summary[qa_key]
        by_count[entry["correct_count"]].append(entry)

    # 按 category 统计分布
    count_by_category = defaultdict(lambda: defaultdict(int))
    for qa_key, entry in qa_summary.items():
        count_by_category[entry["correct_count"]][entry["category"]] += 1

    # 输出每个分组的文件
    W = 80
    print(f"\n{'=' * W}")
    print(f"  Cross-Compare Summary  ({total_runs} runs: {', '.join(run_labels)})")
    print(f"{'=' * W}")
    print(f"  {'Correct Count':<16} {'QA Count':>10} {'Percent':>10}")
    print(f"  {'-' * 50}")

    total_qas = len(qa_summary)
    summary_data = {
        "total_runs": total_runs,
        "run_labels": run_labels,
        "total_qas": total_qas,
        "distribution": {},
    }

    for count in range(total_runs, -1, -1):
        items = by_count.get(count, [])
        n = len(items)
        pct = n / total_qas * 100 if total_qas > 0 else 0

        label = f"{count}/{total_runs}"
        if count == total_runs:
            label += " (all correct)"
        elif count == 0:
            label += " (all wrong)"

        print(f"  {label:<16} {n:>10} {pct:>9.1f}%")

        # 按 category 细分
        cat_dist = count_by_category.get(count, {})
        if cat_dist:
            cats_str = ", ".join(f"{c}:{v}" for c, v in sorted(cat_dist.items()))
            print(f"    breakdown: {cats_str}")

        summary_data["distribution"][str(count)] = {
            "count": n,
            "percent": round(pct, 2),
            "by_category": dict(sorted(cat_dist.items())),
        }

        # 写分组文件
        out_file = os.path.join(output_dir, f"correct_{count}.json")
        output_items = []
        for entry in items:
            compact = {
                "compare_key": entry["compare_key"],
                "qa_id": entry["qa_id"],
                "source_file": entry["source_file"],
                "source_name": entry["source_name"],
                "profile_name": entry["profile_name"],
                "gold_answer": entry["gold_answer"],
                "category": entry["category"],
                "qa_type": entry["qa_type"],
                "entity_explicitness": entry.get("entity_explicitness", ""),
                "entity_source_refs": entry.get("entity_source_refs", []),
                "entity_name": entry.get("entity_name", ""),
                "entity_anchor_lookup_name": entry.get("entity_anchor_lookup_name", ""),
                "question": entry["question"],
                "per_run": [
                    {
                        "model": r["model"],
                        "run": r["run"],
                        "answer": r["answer"],
                        "correct": r["correct"],
                    }
                    for r in entry["per_run"]
                ],
            }
            output_items.append(compact)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_items, f, ensure_ascii=False, indent=2)
        if n > 0:
            print(f"    → {out_file}")

    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<16} {total_qas:>10}")

    # 难度分级汇总
    easy = sum(len(by_count.get(c, [])) for c in range(total_runs, total_runs * 2 // 3, -1))
    medium = sum(len(by_count.get(c, [])) for c in range(total_runs * 2 // 3, total_runs // 3, -1))
    hard = sum(len(by_count.get(c, [])) for c in range(total_runs // 3, -1, -1))
    print("\n  Difficulty tiers (by correct ratio):")
    print(f"    Easy   (>{total_runs*2//3}/{total_runs}): {easy} QAs")
    print(f"    Medium ({total_runs//3+1}-{total_runs*2//3}/{total_runs}): {medium} QAs")
    print(f"    Hard   (≤{total_runs//3}/{total_runs}): {hard} QAs")

    summary_data["difficulty_tiers"] = {
        "easy": easy, "medium": medium, "hard": hard,
    }

    # 模型间一致性：两个模型各自 3 次全对但对方全错的题
    model_names = sorted({r["model"] for entry in qa_summary.values() for r in entry["per_run"]})
    if len(model_names) == 2:
        m1, m2 = model_names
        only_m1, only_m2, both_all = 0, 0, 0
        for entry in qa_summary.values():
            m1_correct = sum(1 for r in entry["per_run"] if r["model"] == m1 and r["correct"])
            m2_correct = sum(1 for r in entry["per_run"] if r["model"] == m2 and r["correct"])
            m1_total = sum(1 for r in entry["per_run"] if r["model"] == m1)
            m2_total = sum(1 for r in entry["per_run"] if r["model"] == m2)
            if m1_correct == m1_total and m2_correct == 0:
                only_m1 += 1
            elif m2_correct == m2_total and m1_correct == 0:
                only_m2 += 1
            elif m1_correct == m1_total and m2_correct == m2_total:
                both_all += 1

        print(f"\n  Model agreement ({m1} vs {m2}):")
        print(f"    Both all-correct:     {both_all}")
        print(f"    Only {m1} all-correct: {only_m1}")
        print(f"    Only {m2} all-correct: {only_m2}")

        summary_data["model_agreement"] = {
            "models": model_names,
            "both_all_correct": both_all,
            f"only_{m1}_all_correct": only_m1,
            f"only_{m2}_all_correct": only_m2,
        }

    # 保存汇总
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary → {summary_path}")
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Retention analysis
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "pref_text",
    "pref_img",
    "rec_text",
    "rec_img",
    "entity_text",
    "entity_img",
    "refusal_text",
]

COMBINED_GROUPS = [
    ("pref_text+pref_img", ("pref_text", "pref_img")),
    ("rec_text+rec_img", ("rec_text", "rec_img")),
    ("entity_text+entity_img", ("entity_text", "entity_img")),
]

ENTITY_EXPLICIT_TYPES = {"Relationship", "Pets"}
SPLIT_ORDER = ["explicit", "implicit", "mixed", "unknown"]


def normalize_category(category):
    """Normalize category names used by different result-generation versions."""

    return _CATEGORY_ALIASES.get(category, category)


def infer_explicitness(record):
    """Infer explicit/implicit split using the benchmark's QA conventions."""

    category = normalize_category(record.get("category", "unknown") or "unknown")
    qa_type = record.get("qa_type", "")
    category_l = (category or "").lower()

    if category_l in {"entity", "entity_img", "entity_text"}:
        if qa_type in ENTITY_EXPLICIT_TYPES:
            return "explicit"
        if qa_type == "Items":
            return record.get("entity_explicitness") or "unknown"
        return "unknown"

    if qa_type in {"explicit", "implicit"}:
        return qa_type
    return "unknown"


def load_correct_bucket(cross_compare_dir, count):
    path = os.path.join(cross_compare_dir, f"correct_{count}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"{path} should contain a JSON list")
    return records


def infer_total_runs(cross_compare_dir):
    summary_path = os.path.join(cross_compare_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        total_runs = summary.get("total_runs")
        if isinstance(total_runs, int) and total_runs > 0:
            return total_runs

    max_count = -1
    for fname in os.listdir(cross_compare_dir):
        if not fname.startswith("correct_") or not fname.endswith(".json"):
            continue
        mid = fname[len("correct_") : -len(".json")]
        if mid.isdigit():
            max_count = max(max_count, int(mid))
    if max_count < 0:
        raise FileNotFoundError(f"No correct_*.json files found in {cross_compare_dir}")
    return max_count


def record_correct_and_total(record, fallback_correct_count, total_runs):
    per_run = record.get("per_run") or []
    if per_run:
        correct = sum(1 for run in per_run if run.get("correct") is True)
        return correct, len(per_run)
    return fallback_correct_count, total_runs


def summarize_records(records, total_runs):
    stats = defaultdict(lambda: {"qa_count": 0, "correct": 0, "trials": 0})
    for record, bucket_count in records:
        category = normalize_category(record.get("category", "unknown") or "unknown")
        correct, trials = record_correct_and_total(record, bucket_count, total_runs)
        stats[category]["qa_count"] += 1
        stats[category]["correct"] += correct
        stats[category]["trials"] += trials
    return stats


def summarize_explicitness(records, total_runs):
    stats = defaultdict(lambda: {"qa_count": 0, "correct": 0, "trials": 0})
    for record, bucket_count in records:
        split = infer_explicitness(record)
        correct, trials = record_correct_and_total(record, bucket_count, total_runs)
        stats[split]["qa_count"] += 1
        stats[split]["correct"] += correct
        stats[split]["trials"] += trials
    return stats


def merge_stats(stats, categories):
    merged = {"qa_count": 0, "correct": 0, "trials": 0}
    for category in categories:
        item = stats.get(category, {})
        merged["qa_count"] += item.get("qa_count", 0)
        merged["correct"] += item.get("correct", 0)
        merged["trials"] += item.get("trials", 0)
    return merged


def accuracy(item):
    trials = item.get("trials", 0)
    return item.get("correct", 0) / trials if trials else 0.0


def row_dict(name, item):
    return {
        "category": name,
        "qa_count": item.get("qa_count", 0),
        "correct": item.get("correct", 0),
        "trials": item.get("trials", 0),
        "accuracy": round(accuracy(item), 6),
    }


def build_retention_report(cross_compare_dir, start_count=1):
    total_runs = infer_total_runs(cross_compare_dir)
    buckets = {
        count: load_correct_bucket(cross_compare_dir, count)
        for count in range(total_runs + 1)
    }

    report = {
        "cross_compare_dir": os.path.abspath(cross_compare_dir),
        "total_runs": total_runs,
        "retention_sets": [],
    }

    retained = []
    for count in range(total_runs + 1):
        retained.extend((record, count) for record in buckets[count])
        if count < start_count:
            continue

        stats = summarize_records(retained, total_runs)
        explicitness_stats = summarize_explicitness(retained, total_runs)
        category_rows = [
            row_dict(category, stats[category])
            for category in CATEGORY_ORDER
            if category in stats
        ]
        explicitness_rows = [
            row_dict(split, explicitness_stats[split])
            for split in SPLIT_ORDER
            if explicitness_stats[split]["qa_count"] > 0
        ]
        combined_rows = [
            row_dict(name, merge_stats(stats, categories))
            for name, categories in COMBINED_GROUPS
        ]

        report["retention_sets"].append(
            {
                "kept_correct_range": f"correct_0..correct_{count}",
                "max_correct_count": count,
                "qa_count": sum(item["qa_count"] for item in stats.values()),
                "category_rows": category_rows,
                "explicitness_rows": explicitness_rows,
                "combined_rows": combined_rows,
            }
        )

    return report


def print_retention_table(report):
    total_runs = report["total_runs"]
    for item in report["retention_sets"]:
        title = item["kept_correct_range"]
        print("\n" + "=" * 88)
        print(
            f"Retained QA set: {title}  |  QA count: {item['qa_count']}  "
            f"|  total_runs: {total_runs}"
        )
        print("=" * 88)
        print(f"{'Category':<26} {'QA':>8} {'Correct':>10} {'Trials':>10} {'Accuracy':>10}")
        print("-" * 88)
        for row in item["category_rows"]:
            print(
                f"{row['category']:<26} {row['qa_count']:>8} "
                f"{row['correct']:>10} {row['trials']:>10} "
                f"{row['accuracy'] * 100:>9.2f}%"
            )
        print("-" * 88)
        for row in item.get("explicitness_rows", []):
            print(
                f"{('split:' + row['category']):<26} {row['qa_count']:>8} "
                f"{row['correct']:>10} {row['trials']:>10} "
                f"{row['accuracy'] * 100:>9.2f}%"
            )
        print("-" * 88)
        for row in item["combined_rows"]:
            print(
                f"{row['category']:<26} {row['qa_count']:>8} "
                f"{row['correct']:>10} {row['trials']:>10} "
                f"{row['accuracy'] * 100:>9.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare repeated question-only runs or analyze cumulative "
            "retention sets. Default mode: cross-compare."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("cross-compare", "retention"),
        default="cross-compare",
        help="cross-compare: generate correct_N buckets; retention: analyze them",
    )
    parser.add_argument(
        "--result_dir", "--result-dir", dest="result_dir", default=DEFAULT_RESULT_DIR,
        help="result_question_only/base 目录路径",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="限定比对的模型（默认全部）",
    )
    parser.add_argument(
        "--num_runs", "--num-runs", dest="num_runs", type=int, default=None,
        help="每个模型最多取几次 run（默认全部）",
    )
    parser.add_argument(
        "--output_dir", "--output-dir", dest="output_dir", default=None,
        help="输出目录（默认 result_dir/cross_compare）",
    )
    parser.add_argument(
        "--include_missing", action="store_true",
        help="使用所有 run 的并集；缺失结果按错误计入（默认只比较所有 run 都存在的题）",
    )
    parser.add_argument(
        "--cross_compare_dir",
        "--cross-compare-dir",
        dest="cross_compare_dir",
        default=DEFAULT_CROSS_COMPARE_DIR,
        help="retention 模式读取的 cross_compare 目录",
    )
    parser.add_argument(
        "--start_count",
        "--start-count",
        dest="start_count",
        type=int,
        default=1,
        help="retention 模式的首个累计桶（默认从 correct_0..correct_1 开始）",
    )
    parser.add_argument(
        "--output_json",
        "--output-json",
        dest="output_json",
        default=None,
        help="retention 模式保存报告 JSON 的路径",
    )
    args = parser.parse_args()

    if args.mode == "retention":
        report = build_retention_report(
            args.cross_compare_dir,
            start_count=args.start_count,
        )
        print_retention_table(report)
        if args.output_json:
            output_parent = os.path.dirname(os.path.abspath(args.output_json))
            if output_parent:
                os.makedirs(output_parent, exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nSaved JSON report to {args.output_json}")
        return

    qa_summary, total_runs, run_labels = cross_compare(
        args.result_dir,
        models=args.models,
        num_runs=args.num_runs,
        include_missing=args.include_missing,
    )
    output_dir = args.output_dir or os.path.join(args.result_dir, "cross_compare")
    output_by_correct_count(qa_summary, total_runs, run_labels, output_dir)


if __name__ == "__main__":
    main()
