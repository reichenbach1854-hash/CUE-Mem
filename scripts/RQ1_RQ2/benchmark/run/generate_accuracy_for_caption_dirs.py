"""
Generate *_accuracy.json files for result_debug caption-like subdirectories.

This script scans each caption directory under result_debug and, for every
*_results.json file, computes the same accuracy summary used by run_bench.py
and writes the result to the matching *_accuracy.json file.

Usage:
    cd <repository-root>
    python generate_accuracy_for_caption_dirs.py
    python generate_accuracy_for_caption_dirs.py --root_dir benchmark/result_debug
    python generate_accuracy_for_caption_dirs.py --captions brief detailed
    python generate_accuracy_for_caption_dirs.py --root_dir benchmark/result_debug/audio_caption --captions qwen3_asr_1.7b
    python generate_accuracy_for_caption_dirs.py --root_dir benchmark/result_debug/base
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.paths import RESULT_ROOT
from benchmark.security import redact_runtime_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(RESULT_ROOT.parent)
DEFAULT_ROOT_DIR = str(RESULT_ROOT)
DEFAULT_CAPTIONS = ["brief", "medium", "detailed"]

ENTITY_EXPLICIT_TYPES = {"Relationship", "Pets"}


def _extract_choice(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = re.search(r"\b([A-D])\b", s.upper())
    return m.group(1) if m else ""


def compute_accuracy_summary(results):
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
        cat = item.get("category", "")
        pred = _extract_choice(item.get("system_answer", ""))
        gold = _extract_choice(item.get("original_answer", ""))
        correct = 1 if pred == gold else 0
        explicitness = get_explicitness(item)

        stats[(cat or "unknown").lower()]["overall"][0] += correct
        stats[(cat or "unknown").lower()]["overall"][1] += 1
        stats[(cat or "unknown").lower()][explicitness][0] += correct
        stats[(cat or "unknown").lower()][explicitness][1] += 1

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


def iter_caption_dirs(root_dir, captions):
    for caption in captions:
        yield caption, os.path.join(root_dir, caption)


def is_result_root(path):
    """判断 path 是否已经是一个直接包含 model/memory/*_results.json 的结果目录。"""
    if not os.path.isdir(path):
        return False
    for model_name in os.listdir(path):
        model_dir = os.path.join(path, model_name)
        if not os.path.isdir(model_dir):
            continue
        for child_name in os.listdir(model_dir):
            child_dir = os.path.join(model_dir, child_name)
            if not os.path.isdir(child_dir):
                continue
            for fname in os.listdir(child_dir):
                if fname.endswith("_results.json"):
                    return True
    return False


def generate_for_dir(caption_dir):
    if not os.path.isdir(caption_dir):
        print(f"Skip missing directory: {caption_dir}")
        return 0

    written = 0
    for model_name in sorted(os.listdir(caption_dir)):
        model_dir = os.path.join(caption_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for memory_name in sorted(os.listdir(model_dir)):
            memory_dir = os.path.join(model_dir, memory_name)
            if not os.path.isdir(memory_dir):
                continue
            for fname in sorted(os.listdir(memory_dir)):
                if not fname.endswith("_results.json"):
                    continue
                results_path = os.path.join(memory_dir, fname)
                accuracy_path = results_path.replace("_results.json", "_accuracy.json")
                try:
                    with open(results_path, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    if not isinstance(results, list):
                        print(f"Skip non-list results: {results_path}")
                        continue
                    summary = compute_accuracy_summary(results)
                    with open(accuracy_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)
                    written += 1
                except Exception as e:
                    print(f"WARN: failed to process {results_path}: {redact_runtime_text(e)}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate *_accuracy.json for result_debug caption directories."
    )
    parser.add_argument(
        "--root_dir",
        default=DEFAULT_ROOT_DIR,
        help="Result root directory, default: result_debug",
    )
    parser.add_argument(
        "--captions",
        nargs="+",
        default=DEFAULT_CAPTIONS,
        help="Subdirectories under --root_dir to process, e.g. brief medium detailed qwen3_asr_1.7b.",
    )
    args = parser.parse_args()

    total = 0
    root_dir = os.path.abspath(args.root_dir)
    if is_result_root(root_dir):
        count = generate_for_dir(root_dir)
        total += count
        print(f"[{os.path.basename(root_dir)}] wrote {count} accuracy files")
    else:
        for caption, caption_dir in iter_caption_dirs(root_dir, args.captions):
            count = generate_for_dir(caption_dir)
            total += count
            print(f"[{caption}] wrote {count} accuracy files")
    print(f"Done. Total accuracy files written: {total}")


if __name__ == "__main__":
    main()
