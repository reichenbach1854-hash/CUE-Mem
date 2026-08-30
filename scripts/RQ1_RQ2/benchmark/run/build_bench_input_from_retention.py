"""
Build a benchmark input subset from question-only retention buckets.

The script copies each source history_with_qa_p*.json file and filters only
"human-annotated QAs":
  - keep QAs that appear in correct_0.json .. correct_N.json
  - always keep all adversarial_text QAs

Multi-session dialogues and character profiles are kept unchanged.

Usage:
    cd <repository-root>
    python build_bench_input_from_retention.py
    python build_bench_input_from_retention.py --max_correct 5
    python build_bench_input_from_retention.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.paths import DATA_ROOT, DIALOG_ROOT, QUESTION_ONLY_RESULT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = DATA_ROOT.parent

DEFAULT_INPUT_DIR = DIALOG_ROOT / "base"
DEFAULT_CROSS_COMPARE_DIR = QUESTION_ONLY_RESULT_ROOT / "base" / "cross_compare"
DEFAULT_OUTPUT_DIR = DIALOG_ROOT / "base_retention_correct0_5"

ADVERSARIAL_POINT = "adversarial_text"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_category(category: str) -> str:
    aliases = {
        "preference_same_category": "pref_img",
        "recommendation_same_category": "rec_img",
        "entity": "entity_img",
    }
    return aliases.get(category, category)


def profile_name_from_path(path: Path) -> str:
    # history_with_qa_p3.json -> history_with_qa_p3
    return path.stem


def profile_name_from_record(record: dict) -> str:
    profile_name = str(record.get("profile_name") or "").strip()
    if profile_name:
        return profile_name

    source_name = str(record.get("source_name") or "").strip()
    match = re.match(r"^(history_with_qa_p\d+)", source_name)
    if match:
        return match.group(1)

    source_file = str(record.get("source_file") or "").strip()
    match = re.match(r"^(history_with_qa_p\d+)", source_file)
    if match:
        return match.group(1)

    compare_key = str(record.get("compare_key") or "")
    if "::" in compare_key:
        return compare_key.split("::", 1)[0]

    return ""


def make_key(profile_name: str, category: str, qa_id: str) -> tuple[str, str, str]:
    return (
        profile_name,
        normalize_category(category or ""),
        qa_id or "",
    )


def qa_key(profile_name: str, qa: dict) -> tuple[str, str, str]:
    return make_key(
        profile_name,
        str(qa.get("point") or qa.get("category") or ""),
        str(qa.get("qa_id") or ""),
    )


def retention_record_key(record: dict) -> tuple[str, str, str]:
    return make_key(
        profile_name_from_record(record),
        str(record.get("category") or record.get("point") or ""),
        str(record.get("qa_id") or ""),
    )


def load_retained_keys(cross_compare_dir: Path, max_correct: int) -> tuple[set[tuple[str, str, str]], Counter]:
    retained: set[tuple[str, str, str]] = set()
    counts = Counter()

    for correct_count in range(max_correct + 1):
        path = cross_compare_dir / f"correct_{correct_count}.json"
        if not path.exists():
            print(f"WARN: missing retention bucket: {path}")
            continue
        records = load_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} should contain a JSON list")

        counts[f"correct_{correct_count}"] = len(records)
        for record in records:
            if not isinstance(record, dict):
                continue
            key = retention_record_key(record)
            if all(key):
                retained.add(key)
            else:
                counts["bad_retention_records"] += 1

    return retained, counts


def filter_profile_qas(
    profile_name: str,
    qas: list[dict],
    retained_keys: set[tuple[str, str, str]],
) -> tuple[list[dict], Counter]:
    kept = []
    stats = Counter()
    kept_seen: set[tuple[str, str, str]] = set()

    for qa in qas:
        point = str(qa.get("point") or qa.get("category") or "")
        key = qa_key(profile_name, qa)

        if point == ADVERSARIAL_POINT:
            kept.append(qa)
            stats["kept_adversarial"] += 1
            stats[f"kept_point::{point}"] += 1
            kept_seen.add(key)
            continue

        if key in retained_keys:
            kept.append(qa)
            stats["kept_retention"] += 1
            stats[f"kept_point::{point}"] += 1
            kept_seen.add(key)
        else:
            stats["dropped"] += 1
            stats[f"dropped_point::{point}"] += 1

    stats["kept_total"] = len(kept)
    stats["input_total"] = len(qas)
    stats["matched_unique_keys"] = len(kept_seen)
    return kept, stats


def build_subset(
    input_dir: Path,
    cross_compare_dir: Path,
    output_dir: Path,
    max_correct: int,
    dry_run: bool = False,
) -> dict:
    retained_keys, bucket_counts = load_retained_keys(cross_compare_dir, max_correct)
    if not retained_keys:
        raise ValueError(f"No retained QA keys loaded from {cross_compare_dir}")

    source_files = sorted(input_dir.glob("history_with_qa_p*.json"))
    if not source_files:
        raise FileNotFoundError(f"No history_with_qa_p*.json files found in {input_dir}")

    if output_dir.exists() and not dry_run:
        shutil.rmtree(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input_dir": str(input_dir.resolve()),
        "cross_compare_dir": str(cross_compare_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "kept_correct_range": f"correct_0..correct_{max_correct}",
        "retention_bucket_counts": dict(bucket_counts),
        "retained_unique_keys": len(retained_keys),
        "profiles": [],
    }

    total = Counter()
    by_point_total: dict[str, Counter] = defaultdict(Counter)

    for source_path in source_files:
        data = load_json(source_path)
        profile_name = profile_name_from_path(source_path)
        qas = data.get("human-annotated QAs", []) or []
        if not isinstance(qas, list):
            raise ValueError(f"{source_path} has non-list human-annotated QAs")

        kept_qas, stats = filter_profile_qas(profile_name, qas, retained_keys)
        data["human-annotated QAs"] = kept_qas

        out_path = output_dir / source_path.name
        if not dry_run:
            write_json(out_path, data)

        point_counts = Counter(str(q.get("point") or "unknown") for q in kept_qas)
        dropped_point_counts = Counter({
            key.split("::", 1)[1]: value
            for key, value in stats.items()
            if key.startswith("dropped_point::")
        })

        for key, value in stats.items():
            total[key] += value
        for point, count in point_counts.items():
            by_point_total["kept"][point] += count
        for point, count in dropped_point_counts.items():
            by_point_total["dropped"][point] += count

        public_stats = {
            key: value
            for key, value in stats.items()
            if not key.startswith("kept_point::") and not key.startswith("dropped_point::")
        }

        report["profiles"].append({
            "source_file": str(source_path),
            "output_file": str(out_path),
            "profile_name": profile_name,
            **dict(public_stats),
            "kept_by_point": dict(sorted(point_counts.items())),
            "dropped_by_point": dict(sorted(dropped_point_counts.items())),
        })

    report["total"] = dict(total)
    report["kept_by_point_total"] = dict(sorted(by_point_total["kept"].items()))
    report["dropped_by_point_total"] = dict(sorted(by_point_total["dropped"].items()))

    if not dry_run:
        write_json(output_dir / "build_retention_subset_report.json", report)

    return report


def print_report(report: dict) -> None:
    print("=" * 92)
    print("Build Benchmark Input From Retention")
    print("=" * 92)
    print(f"Input     : {report['input_dir']}")
    print(f"Retention : {report['cross_compare_dir']}")
    print(f"Output    : {report['output_dir']}")
    print(f"Keep      : {report['kept_correct_range']} + all {ADVERSARIAL_POINT}")
    print(f"Keys      : {report['retained_unique_keys']} retained unique QA keys")
    print("-" * 92)

    for item in report["profiles"]:
        print(
            f"{item['profile_name']:<20} "
            f"input={item['input_total']:>4} "
            f"kept={item['kept_total']:>4} "
            f"retention={item['kept_retention']:>4} "
            f"adversarial={item['kept_adversarial']:>4} "
            f"dropped={item['dropped']:>4}"
        )

    total = report["total"]
    print("-" * 92)
    print(
        f"{'TOTAL':<20} "
        f"input={total.get('input_total', 0):>4} "
        f"kept={total.get('kept_total', 0):>4} "
        f"retention={total.get('kept_retention', 0):>4} "
        f"adversarial={total.get('kept_adversarial', 0):>4} "
        f"dropped={total.get('dropped', 0):>4}"
    )

    print("\nKept by point:")
    for point, count in report["kept_by_point_total"].items():
        print(f"  {point:<20} {count}")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter benchmark input QAs by question-only retention buckets."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--cross-compare-dir", type=Path, default=DEFAULT_CROSS_COMPARE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-correct",
        type=int,
        default=5,
        help="Keep correct_0..correct_N. Default keeps correct_0..correct_5.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_correct < 0:
        raise ValueError("--max-correct must be >= 0")

    report = build_subset(
        input_dir=args.input_dir,
        cross_compare_dir=args.cross_compare_dir,
        output_dir=args.output_dir,
        max_correct=args.max_correct,
        dry_run=args.dry_run,
    )
    print_report(report)
    if args.dry_run:
        print("\nDry run only; no files were written.")
    else:
        print(f"\nSaved subset and report to {args.output_dir}")


if __name__ == "__main__":
    main()
