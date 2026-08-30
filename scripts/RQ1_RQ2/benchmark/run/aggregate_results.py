"""统一的 RQ1/RQ2 结果统计入口。

原实验目录中有多个只负责汇总 JSON 结果的脚本。本文件将它们统一为
五种模式：

``overall``
    汇总普通 benchmark 的 model × memory × QA category 结果。
``question-only``
    汇总不注入记忆的结果，并统计 A/B/C/D 选项偏置。
``adversarial``
    汇总 question-only 的 adversarial QA 结果。
``caption``
    横向比较 brief/medium/detailed caption 结果。
``audio-caption``
    横向比较不同 audio-caption 模型的结果。

示例：
    python -m scripts.RQ1_RQ2.benchmark.run.aggregate_results --mode overall
    python -m scripts.RQ1_RQ2.benchmark.run.aggregate_results --mode caption
    python -m scripts.RQ1_RQ2.benchmark.run.aggregate_results --mode question-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow both ``python -m ...`` and direct execution from benchmark/run.
    # Search upward instead of relying on fixed parent indexes so the file is
    # also easy to test from a temporary checkout.
    file_path = Path(__file__).resolve()
    import_roots: list[Path] = []
    for parent in file_path.parents:
        if (parent / "benchmark").is_dir() and parent not in import_roots:
            import_roots.append(parent)
        if (parent / "scripts").is_dir() and parent not in import_roots:
            import_roots.append(parent)
    for import_root in reversed(import_roots):
        sys.path.insert(0, str(import_root))

from benchmark.paths import (
    QA_ROOT,
    QUESTION_ONLY_RESULT_ROOT,
    RESULT_ROOT,
    TRIMMED_RESULT_ROOT,
)
from benchmark.security import redact_runtime_text

SPLITS = ("explicit", "implicit", "mixed", "unknown", "overall")
OPTIONS = ("A", "B", "C", "D")
QA_CATEGORIES = (
    "pref_img",
    "pref_text",
    "rec_img",
    "rec_text",
    "entity_img",
    "entity_text",
    "refusal_text",
)
QUESTION_ONLY_POINT_TYPES = (
    "pref_text",
    "pref_img",
    "rec_text",
    "rec_img",
    "entity_text",
    "entity_img",
    "refusal_text",
    "audio_context",
)
QA_CATEGORY_ALIASES = {"refusal": "refusal_text"}
ENTITY_CATEGORIES = {"entity", "entity_img", "entity_text"}
ENTITY_EXPLICIT_TYPES = {"Relationship", "Pets"}
COMBINED_QA_CATEGORIES = {
    "pref_text+pref_img": ("pref_text", "pref_img"),
    "rec_text+rec_img": ("rec_text", "rec_img"),
    "entity_text+entity_img": ("entity_text", "entity_img"),
}

CAPTION_NAMES = ("brief", "medium", "detailed")
AUDIO_CAPTION_NAMES = (
    "qwen3_asr_1.7b",
    "qwen_audio",
    "qwen2_audio_7b",
    "moss_audio_8b",
    "gemini-3.1-pro",
    "voice_bgm_split",
)

MEMORY_DISPLAY_NAMES = {
    "FUMemory": "Full Memory",
    "STMemory": "FIFO",
    "LTMemory": "NaiveRAG",
    "GAMemory": "Generative Agents",
    "MGMemory": "MemGPT",
    "RFMemory": "Reflexion",
    "AMemMemory": "A-Mem",
    "MemoryOSMemory": "MemoryOS",
    "MMMemory": "MMMemory",
    "MMFUMemory": "MMFUMemory",
    "NGMemory": "NGMemory",
    "AUGUSTUSMemory": "AUGUSTUS",
    "UniversalRAGMemory": "UniversalRAG",
    "__single__": "Single result set",
}
MEMORY_DISPLAY_ORDER = tuple(MEMORY_DISPLAY_NAMES)

DEFAULT_OVERALL_DIR = TRIMMED_RESULT_ROOT / "base"
DEFAULT_QUESTION_ONLY_DIR = QUESTION_ONLY_RESULT_ROOT / "base"
DEFAULT_ADVERSARIAL_DIR = (
    QUESTION_ONLY_RESULT_ROOT / "base" / "qwen3.6-35b-a3b" / "final"
)
DEFAULT_REFUSAL_SOURCE = QA_ROOT / "qa_adversarial_llm_mcq_000_002.json"

_REFUSAL_EXPR_LOOKUP: dict[str, str] | None = None
_REFUSAL_QUESTION_LOOKUP: dict[tuple[Any, str, str], str] | None = None


def empty_splits() -> dict[str, dict[str, int]]:
    return {split: {"correct": 0, "total": 0} for split in SPLITS}


def empty_group() -> dict[str, Any]:
    return {
        "characters": set(),
        **empty_splits(),
        "per_category": defaultdict(empty_splits),
    }


def normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return QA_CATEGORY_ALIASES.get(text, text) or None


def extract_choice(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text and text[0] in OPTIONS:
        return text[0]
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else ""


def accuracy(correct: int, total: int) -> float:
    return round(correct / total * 100, 1) if total else 0.0


def accuracy_text(correct: int, total: int) -> str:
    return f"{accuracy(correct, total):.1f}%" if total else "-"


def parse_accuracy_filename(filename: str) -> tuple[str, str | None]:
    """Return dataset stem and category suffix from an accuracy filename."""

    stem = filename.removesuffix("_accuracy.json")
    categories = sorted(
        set(QA_CATEGORIES) | set(QUESTION_ONLY_POINT_TYPES) | set(QA_CATEGORY_ALIASES),
        key=len,
        reverse=True,
    )
    for category in categories:
        if stem.endswith(f"_{category}"):
            return stem[: -(len(category) + 1)], normalize_category(category)
    return stem, None


def load_json(path: Path, *, warn: bool = True) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if warn:
            print(f"WARN: skip {path}: {redact_runtime_text(exc)}")
        return None


def add_split_counts(
    group: dict[str, Any],
    category: str | None,
    split_counts: dict[str, dict[str, int]],
) -> None:
    category = normalize_category(category) or "__full__"
    per_category = group["per_category"][category]
    for split in SPLITS:
        source = split_counts.get(split)
        if not isinstance(source, dict):
            continue
        correct = int(source.get("correct", 0) or 0)
        total = int(source.get("total", 0) or 0)
        group[split]["correct"] += correct
        group[split]["total"] += total
        per_category[split]["correct"] += correct
        per_category[split]["total"] += total


def result_path_for_accuracy(path: Path) -> Path:
    direct = path.with_name(path.name.replace("_accuracy.json", "_results.json"))
    if direct.is_file():
        return direct
    stem = path.name.removesuffix("_accuracy.json")
    candidates = sorted(
        item
        for item in path.parent.iterdir()
        if item.name.startswith(stem) and item.name.endswith("_results.json")
    )
    return candidates[0] if candidates else direct


def result_correct(item: dict[str, Any]) -> int:
    if isinstance(item.get("correct"), bool):
        return int(item["correct"])
    predicted = extract_choice(item.get("system_answer") or item.get("model_answer"))
    gold = extract_choice(item.get("original_answer") or item.get("answer"))
    return int(bool(gold) and predicted == gold)


def entity_splits(results_path: Path) -> dict[str, dict[str, int]]:
    output = empty_splits()
    payload = load_json(results_path, warn=False)
    if not isinstance(payload, list):
        return output
    for item in payload:
        if not isinstance(item, dict):
            continue
        correct = result_correct(item)
        qa_type = str(item.get("qa_type", ""))
        if qa_type in ENTITY_EXPLICIT_TYPES:
            bucket = "explicit"
        elif qa_type == "Items":
            bucket = str(item.get("entity_explicitness") or "unknown")
        else:
            bucket = "unknown"
        output["overall"]["correct"] += correct
        output["overall"]["total"] += 1
        if bucket in output:
            output[bucket]["correct"] += correct
            output[bucket]["total"] += 1
    return output


def _load_refusal_lookups() -> tuple[dict[str, str], dict[tuple[Any, str, str], str]]:
    global _REFUSAL_EXPR_LOOKUP, _REFUSAL_QUESTION_LOOKUP
    if _REFUSAL_EXPR_LOOKUP is not None and _REFUSAL_QUESTION_LOOKUP is not None:
        return _REFUSAL_EXPR_LOOKUP, _REFUSAL_QUESTION_LOOKUP

    by_id: dict[str, str] = {}
    by_question: dict[tuple[Any, str, str], str] = {}
    payload = load_json(DEFAULT_REFUSAL_SOURCE)
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            expression = str(item.get("expression_type", "")).lower()
            if expression not in {"explicit", "implicit"}:
                continue
            qa_id = str(item.get("qa_id", "")).strip()
            if qa_id:
                by_id[qa_id] = expression
            question = " ".join(str(item.get("Q", "")).split())
            if question:
                by_question[(item.get("p_id"), str(item.get("category", "")), question)] = expression
    _REFUSAL_EXPR_LOOKUP = by_id
    _REFUSAL_QUESTION_LOOKUP = by_question
    return by_id, by_question


def refusal_expression(item: dict[str, Any]) -> str | None:
    by_id, by_question = _load_refusal_lookups()
    qa_id = str(item.get("qa_id", ""))
    direct = by_id.get(qa_id)
    if direct:
        return direct
    expression = str(item.get("expression_type", "")).lower()
    if expression in {"explicit", "implicit"}:
        return expression

    p_id = item.get("p_id")
    if p_id is None:
        try:
            p_id = int(qa_id.split("-", 1)[0])
        except (TypeError, ValueError, IndexError):
            p_id = None
    category = str(item.get("source_category") or item.get("domain") or "")
    if not category:
        parts = qa_id.split("-")
        category = parts[1] if len(parts) > 1 else ""
    question = str(item.get("question") or item.get("Q") or "")
    first_line = " ".join(question.splitlines()[0].split()) if question else ""
    return by_question.get((p_id, category, first_line))


def refusal_splits(results_path: Path) -> dict[str, dict[str, int]]:
    output = empty_splits()
    payload = load_json(results_path, warn=False)
    if not isinstance(payload, list):
        return output
    unmatched = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not result_is_refusal_item(item):
            continue
        expression = refusal_expression(item)
        correct = result_correct(item)
        output["overall"]["correct"] += correct
        output["overall"]["total"] += 1
        if expression in {"explicit", "implicit"}:
            output[expression]["correct"] += correct
            output[expression]["total"] += 1
        else:
            unmatched += 1
    if unmatched:
        print(f"WARN: {unmatched} refusal result(s) could not be matched to expression_type")
    return output


def result_is_refusal_item(item: dict[str, Any]) -> bool:
    """Return whether a result record belongs to the refusal QA set."""

    category = normalize_category(
        item.get("point") or item.get("category") or item.get("qa_category")
    )
    if category == "refusal_text":
        return True
    qa_id = str(item.get("qa_id", "")).lower()
    return not category and ("refusal" in qa_id or "-adv" in qa_id)


def needs_entity_recompute(payload: dict[str, Any], category: str | None) -> bool:
    if category not in ENTITY_CATEGORIES:
        return False
    return any(
        isinstance(value, dict) and "unknown" in value and "explicit" not in value
        for key, value in payload.items()
        if key != "__overall__"
    )


def add_accuracy_file(
    group: dict[str, Any],
    path: Path,
    filename_category: str | None = None,
    *,
    force_filename_category: bool = False,
) -> None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return
    filename_category = normalize_category(filename_category)

    if filename_category == "refusal_text":
        add_split_counts(group, filename_category, refusal_splits(result_path_for_accuracy(path)))
        return
    if force_filename_category:
        filename_category = None

    if not any(
        isinstance(value, dict) and any(split in value for split in SPLITS)
        for value in payload.values()
    ) and filename_category is not None:
        add_split_counts(group, filename_category, empty_splits())

    for raw_category, category_data in payload.items():
        if raw_category == "__overall__" or not isinstance(category_data, dict):
            continue
        category = normalize_category(raw_category)
        if category == "refusal_text":
            add_split_counts(group, category, refusal_splits(result_path_for_accuracy(path)))
        elif needs_entity_recompute(payload, category):
            add_split_counts(group, category, entity_splits(result_path_for_accuracy(path)))
        else:
            add_split_counts(group, category or filename_category, category_data)


def model_memory_dirs(root: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield ``(model, memory, directory)`` for common result layouts."""

    if not root.is_dir():
        return
    direct_accuracy = sorted(root.glob("*_accuracy.json"))
    if direct_accuracy:
        yield root.name, "__single__", root
        return

    for model_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        direct_accuracy = sorted(model_dir.glob("*_accuracy.json"))
        if direct_accuracy:
            yield model_dir.name, "__single__", model_dir
            continue
        for memory_dir in sorted(item for item in model_dir.iterdir() if item.is_dir()):
            if any(memory_dir.glob("*_accuracy.json")):
                yield model_dir.name, memory_dir.name, memory_dir


def scan_result_root(
    root: Path,
    *,
    force_filename_category: bool = False,
) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    groups: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for model, memory, result_dir in model_memory_dirs(root):
        for accuracy_path in sorted(result_dir.glob("*_accuracy.json")):
            dataset, filename_category = parse_accuracy_filename(accuracy_path.name)
            key = (model, memory, None if force_filename_category else filename_category)
            group = groups.setdefault(key, empty_group())
            group["characters"].add(dataset)
            add_accuracy_file(
                group,
                accuracy_path,
                filename_category,
                force_filename_category=force_filename_category,
            )
    return groups


def json_group(group: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        split: dict(group[split]) for split in SPLITS
    }
    output["n_characters"] = len(group["characters"])
    output["per_category"] = {
        category: {split: dict(values[split]) for split in SPLITS}
        for category, values in group["per_category"].items()
    }
    output["combined"] = {}
    for name, categories in COMBINED_QA_CATEGORIES.items():
        output["combined"][name] = {}
        for split in ("explicit", "implicit", "overall"):
            correct = sum(
                output["per_category"].get(category, {}).get(split, {}).get("correct", 0)
                for category in categories
            )
            total = sum(
                output["per_category"].get(category, {}).get(split, {}).get("total", 0)
                for category in categories
            )
            output["combined"][name][split] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy(correct, total),
            }
    return output


def memory_sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
    model, memory = item
    try:
        order = MEMORY_DISPLAY_ORDER.index(memory)
    except ValueError:
        order = len(MEMORY_DISPLAY_ORDER)
    return model, order, memory


def group_sort_key(
    item: tuple[tuple[str, str, str | None], dict[str, Any]],
) -> tuple[str, int, str, bool, str]:
    (model, memory, category), _group = item
    model_name, memory_order, memory_name = memory_sort_key((model, memory))
    return model_name, memory_order, memory_name, category is not None, category or ""


def filter_groups(
    groups: dict[tuple[str, str, str | None], dict[str, Any]],
    *,
    model: str | None = None,
    category: str | None = None,
) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    category = normalize_category(category)
    return {
        key: group
        for key, group in groups.items()
        if (model is None or key[0] == model)
        and (category is None or key[2] == category)
    }


def print_overall(groups: dict[tuple[str, str, str | None], dict[str, Any]]) -> None:
    print(f"Found {len(groups)} model × memory × category group(s)")
    print(f"{'Model':<24} {'Memory':<20} {'Category':<18} {'Explicit':>10} {'Implicit':>10} {'Overall':>10} {'N':>6}")
    print("-" * 108)
    for (model, memory, category), group in sorted(groups.items(), key=group_sort_key):
        print(
            f"{model:<24} {MEMORY_DISPLAY_NAMES.get(memory, memory):<20} "
            f"{category or '__full__':<18} "
            f"{accuracy_text(**group['explicit']):>10} "
            f"{accuracy_text(**group['implicit']):>10} "
            f"{accuracy_text(**group['overall']):>10} "
            f"{group['overall']['total']:>6}"
        )


def print_per_category_overall(
    groups: dict[tuple[str, str, str | None], dict[str, Any]],
) -> None:
    categories = sorted(
        {
            category
            for group in groups.values()
            for category in group["per_category"]
            if category != "__full__"
        }
    )
    if not categories:
        return
    print("\nPer QA category:")
    print(
        f"{'Model':<24} {'Memory':<20} {'QA category':<18} "
        f"{'Explicit':>10} {'Implicit':>10} {'Overall':>10} {'N':>6}"
    )
    print("-" * 108)
    for (model, memory, _filter), group in sorted(groups.items(), key=group_sort_key):
        for category in categories:
            stats = group["per_category"].get(category)
            if not stats or stats["overall"]["total"] == 0:
                continue
            print(
                f"{model:<24} {MEMORY_DISPLAY_NAMES.get(memory, memory):<20} "
                f"{category:<18} "
                f"{accuracy_text(**stats['explicit']):>10} "
                f"{accuracy_text(**stats['implicit']):>10} "
                f"{accuracy_text(**stats['overall']):>10} "
                f"{stats['overall']['total']:>6}"
            )


def run_overall(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir or DEFAULT_OVERALL_DIR).expanduser().resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory not found: {result_dir}")
    groups = filter_groups(
        scan_result_root(result_dir),
        model=args.model,
        category=args.category,
    )
    if not groups:
        raise SystemExit(f"No accuracy files found under: {result_dir}")
    print_overall(groups)
    if args.per_category:
        print_per_category_overall(groups)

    output = {
        "groups": [
            {
                "model": model,
                "memory": memory,
                "category_filter": category,
                **json_group(group),
            }
            for (model, memory, category), group in sorted(groups.items(), key=group_sort_key)
        ]
    }
    output_name = "aggregated_accuracy"
    if args.category:
        output_name += f"_{args.category}"
    output_path = Path(args.output) if args.output else result_dir / f"{output_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


def chi2_uniform_test(counter: Counter[str]) -> tuple[float | None, float | None]:
    total = sum(counter.get(option, 0) for option in OPTIONS)
    if not total:
        return None, None
    expected = total / len(OPTIONS)
    statistic = sum((counter.get(option, 0) - expected) ** 2 / expected for option in OPTIONS)
    # Survival function for chi-square with df=len(OPTIONS)-1.  This lightweight
    # calculation avoids adding scipy just for the question-only report.
    shape = (len(OPTIONS) - 1) / 2
    z = statistic / 2
    term = 1.0 / shape
    series = term
    for index in range(1, 200):
        term *= z / (shape + index)
        series += term
        if abs(term) < 1e-12:
            break
    try:
        lower = (z**shape) * math.exp(-z) * series
        p_value = 1.0 - lower / math.gamma(shape)
        return statistic, max(0.0, min(1.0, p_value))
    except (OverflowError, ValueError, ZeroDivisionError):
        return statistic, None


def scan_question_only(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for model, _memory, result_dir in model_memory_dirs(root):
        for accuracy_path in sorted(result_dir.glob("*_accuracy.json")):
            dataset, point = parse_accuracy_filename(accuracy_path.name)
            key = (model, point or "unknown")
            group = groups.setdefault(key, empty_group())
            group["characters"].add(dataset)
            payload = load_json(accuracy_path)
            if not isinstance(payload, dict):
                continue
            for category_name, category_data in payload.items():
                if category_name == "__overall__":
                    continue
                if not isinstance(category_data, dict):
                    continue
                for split in SPLITS:
                    value = category_data.get(split)
                    if isinstance(value, dict):
                        group[split]["correct"] += int(value.get("correct", 0) or 0)
                        group[split]["total"] += int(value.get("total", 0) or 0)

            results_path = result_path_for_accuracy(accuracy_path)
            records = load_json(results_path, warn=False)
            if isinstance(records, list):
                model_dist = group.setdefault("model_answer_dist", Counter())
                truth_dist = group.setdefault("truth_answer_dist", Counter())
                for item in records:
                    if not isinstance(item, dict):
                        continue
                    predicted = extract_choice(item.get("original_answer"))
                    truth = extract_choice(item.get("system_answer"))
                    if predicted:
                        model_dist[predicted] += 1
                    if truth:
                        truth_dist[truth] += 1
    return groups


def point_sort_key(point: str) -> tuple[int, str]:
    try:
        order = QUESTION_ONLY_POINT_TYPES.index(point)
    except ValueError:
        order = len(QUESTION_ONLY_POINT_TYPES)
    return order, point


def question_only_output(
    groups: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for (model, point), group in sorted(
        groups.items(), key=lambda item: (item[0][0], point_sort_key(item[0][1]))
    ):
        model_dist = group.get("model_answer_dist", Counter())
        truth_dist = group.get("truth_answer_dist", Counter())
        chi2, p_value = chi2_uniform_test(model_dist)
        output.setdefault(model, {})[point] = {
            "explicit": accuracy(group["explicit"]["correct"], group["explicit"]["total"]),
            "implicit": accuracy(group["implicit"]["correct"], group["implicit"]["total"]),
            "overall": accuracy(group["overall"]["correct"], group["overall"]["total"]),
            "total": group["overall"]["total"],
            "n_characters": len(group["characters"]),
            "model_answer_distribution": {option: model_dist.get(option, 0) for option in OPTIONS},
            "truth_answer_distribution": {option: truth_dist.get(option, 0) for option in OPTIONS},
            "bias_chi2": round(chi2, 4) if chi2 is not None else None,
            "bias_p": round(p_value, 4) if p_value is not None else None,
            "bias_detected": p_value is not None and p_value < 0.05,
        }
    return output


def format_distribution(counter: Counter[str]) -> str:
    total = sum(counter.get(option, 0) for option in OPTIONS)
    if not total:
        return "-"
    return "  ".join(
        f"{option}:{counter.get(option, 0)}({counter.get(option, 0) / total * 100:.0f}%)"
        for option in OPTIONS
    )


def print_question_only_summary(
    groups: dict[tuple[str, str], dict[str, Any]],
) -> None:
    models = sorted({model for model, _point in groups})
    print(f"Found {len(models)} model(s): {', '.join(models)}")
    points = sorted({point for _model, point in groups}, key=point_sort_key)
    print(f"Found {len(points)} point type(s): {', '.join(points)}")

    for model in models:
        model_points = sorted(
            [point for current_model, point in groups if current_model == model],
            key=point_sort_key,
        )
        print(f"\nQuestion-only summary: {model}")
        print(f"{'Point type':<18} {'Explicit':>10} {'Implicit':>10} {'Overall':>10} {'N':>7}")
        print("-" * 62)
        total_correct = total = 0
        model_dist: Counter[str] = Counter()
        truth_dist: Counter[str] = Counter()
        for point in model_points:
            group = groups[(model, point)]
            total_correct += group["overall"]["correct"]
            total += group["overall"]["total"]
            model_dist.update(group.get("model_answer_dist", Counter()))
            truth_dist.update(group.get("truth_answer_dist", Counter()))
            print(
                f"{point:<18} {accuracy_text(**group['explicit']):>10} "
                f"{accuracy_text(**group['implicit']):>10} "
                f"{accuracy_text(**group['overall']):>10} {group['overall']['total']:>7}"
            )
        print("-" * 62)
        print(f"{'ALL':<18} {'-':>10} {'-':>10} {accuracy_text(total_correct, total):>10} {total:>7}")

        chi2, p_value = chi2_uniform_test(model_dist)
        p_text = f"{p_value:.4f}" if p_value is not None else "-"
        print(f"Model answer distribution: {format_distribution(model_dist)}")
        print(f"Bias test: chi2={chi2:.2f}, p={p_text}" if chi2 is not None else "Bias test: -")
        print(f"Correct answer distribution: {format_distribution(truth_dist)}")


def run_question_only(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir or DEFAULT_QUESTION_ONLY_DIR).expanduser().resolve()
    groups = scan_question_only(result_dir)
    point_filter = normalize_category(args.point or args.category)
    if point_filter is not None:
        groups = {
            key: group for key, group in groups.items() if key[1] == point_filter
        }
    if args.model is not None:
        groups = {key: group for key, group in groups.items() if key[0] == args.model}
    if not groups:
        raise SystemExit(f"No question-only accuracy files found under: {result_dir}")
    output = question_only_output(groups)
    print_question_only_summary(groups)
    output_path = Path(args.output) if args.output else result_dir / "aggregated_question_only.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


def run_adversarial(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir or DEFAULT_ADVERSARIAL_DIR).expanduser().resolve()
    files = sorted(result_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {result_dir / args.pattern}")
    stats: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_file = []
    for path in files:
        records = load_json(path)
        if not isinstance(records, list):
            continue
        local = [0, 0]
        for item in records:
            if not isinstance(item, dict):
                continue
            predicted = extract_choice(item.get("system_answer"))
            gold = extract_choice(item.get("original_answer") or item.get("original_answe"))
            correct = int(predicted == gold and predicted in OPTIONS)
            qa_type = str(item.get("qa_type") or "unknown")
            for key in ("overall", qa_type):
                stats[key][0] += correct
                stats[key][1] += 1
            local[0] += correct
            local[1] += 1
        per_file.append({"file": path.name, "correct": local[0], "total": local[1], "accuracy": accuracy(*local)})

    print(f"Result dir: {result_dir}")
    print(f"Files: {len(per_file)}")
    for row in per_file:
        print(f"  {row['file']}: {row['correct']}/{row['total']} = {row['accuracy']}%")
    print("Summary:")
    for key in ("overall", "explicit", "implicit", "unknown"):
        if stats[key][1]:
            print(f"  {key}: {stats[key][0]}/{stats[key][1]} = {accuracy(*stats[key])}%")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"files": per_file, "summary": dict(stats)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Results saved to {output_path}")


def virtual_group(
    data: dict[tuple[str, str, str | None], dict[str, Any]],
    model: str,
    memory: str,
    category: str | None,
) -> dict[str, Any] | None:
    direct = data.get((model, memory, category))
    if direct is not None:
        return direct
    base = data.get((model, memory, None))
    if base is None or category is None:
        return base
    category_data = base["per_category"].get(category)
    if category_data is None:
        return None
    group = empty_group()
    group["characters"] = base["characters"]
    for split in SPLITS:
        group[split] = dict(category_data[split])
    group["per_category"][category] = {split: dict(category_data[split]) for split in SPLITS}
    return group


def scan_caption_column(root: Path) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    return scan_result_root(root)


def scan_audio_column(root: Path) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    return scan_result_root(root, force_filename_category=True)


def collect_cross_data(
    result_dir: Path,
    columns: list[str],
    *,
    audio: bool,
) -> dict[str, dict[tuple[str, str, str | None], dict[str, Any]]]:
    all_data = {}
    audio_root = result_dir if result_dir.name == "audio_caption" else result_dir / "audio_caption"
    for column in columns:
        if audio:
            root = audio_root / column
        elif column == "base":
            root = result_dir / "base" if (result_dir / "base").is_dir() else result_dir
        else:
            root = result_dir / column
        if not root.is_dir():
            print(f"WARN: result directory not found: {root}")
            continue
        scanner = scan_audio_column if audio else scan_caption_column
        groups = scanner(root)
        if groups:
            all_data[column] = groups
        print(f"[{column}] scanned {root}: {len(groups)} group(s)")
    return all_data


def per_category_cross_rows(
    all_data: dict[str, dict[tuple[str, str, str | None], dict[str, Any]]],
    columns: list[str],
    category_filter: str | None,
    model_filter: str | None,
) -> dict[str, list[dict[str, Any]]]:
    categories: set[str] = set()
    for data in all_data.values():
        for model, memory, _category in data:
            if model_filter and model != model_filter:
                continue
            group = virtual_group(data, model, memory, category_filter)
            if group is not None:
                categories.update(group["per_category"])

    output = {}
    for category in sorted(categories):
        if category == "__full__":
            continue
        rows = cross_rows(all_data, columns, category, model_filter)
        if rows:
            output[category] = rows
    return output


def cross_rows(
    all_data: dict[str, dict[tuple[str, str, str | None], dict[str, Any]]],
    columns: list[str],
    category: str | None,
    model_filter: str | None,
) -> list[dict[str, Any]]:
    keys: set[tuple[str, str]] = set()
    for data in all_data.values():
        for model, memory, _ in data:
            if model_filter and model != model_filter:
                continue
            if virtual_group(data, model, memory, category) is not None:
                keys.add((model, memory))
    rows = []
    for model, memory in sorted(keys, key=memory_sort_key):
        row = {"model": model, "memory": memory, "memory_display": MEMORY_DISPLAY_NAMES.get(memory, memory)}
        for column in columns:
            group = virtual_group(all_data[column], model, memory, category)
            if group is None:
                row[column] = None
                continue
            details = {
                split: {
                    "accuracy": accuracy(group[split]["correct"], group[split]["total"]),
                    "correct": group[split]["correct"],
                    "total": group[split]["total"],
                }
                for split in ("overall", "explicit", "implicit")
            }
            row[column] = {
                "avg": details["overall"]["accuracy"],
                "explicit": details["explicit"]["accuracy"],
                "implicit": details["implicit"]["accuracy"],
                "correct": details["overall"]["correct"],
                "total": details["overall"]["total"],
                "n_characters": len(group["characters"]),
                "details": details,
            }
        rows.append(row)
    return rows


def print_cross_table(rows: list[dict[str, Any]], columns: list[str], title: str, category: str | None) -> None:
    label = f" ({category})" if category else " (__full__)"
    for split, split_title in (
        ("overall", "AVG"),
        ("explicit", "Explicit"),
        ("implicit", "Implicit"),
    ):
        has_data = any(
            row.get(column, {}).get("details", {}).get(split, {}).get("total", 0)
            for row in rows
            for column in columns
            if row.get(column)
        )
        if split != "overall" and not has_data:
            continue
        print(f"\n{title} — {split_title}{label}")
        print("=" * (46 + 14 * len(columns)))
        print(f"{'Model':<24} {'Memory':<20}" + "".join(f" {column:>12}" for column in columns))
        print("-" * (46 + 14 * len(columns)))
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column)
                detail = value.get("details", {}).get(split) if value else None
                values.append(
                    f"{accuracy_text(detail['correct'], detail['total']):>12}"
                    if detail
                    else f"{'-':>12}"
                )
            print(f"{row['model']:<24} {row['memory_display']:<20}" + "".join(values))


def combined_rows(
    all_data: dict[str, dict[tuple[str, str, str | None], dict[str, Any]]],
    columns: list[str],
    model_filter: str | None,
) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for combined_name, categories in COMBINED_QA_CATEGORIES.items():
        keys: set[tuple[str, str]] = set()
        for data in all_data.values():
            for model, memory, _ in data:
                if model_filter and model != model_filter:
                    continue
                if any(virtual_group(data, model, memory, category) for category in categories):
                    keys.add((model, memory))
        rows = []
        for model, memory in sorted(keys, key=memory_sort_key):
            row = {"model": model, "memory": memory, "memory_display": MEMORY_DISPLAY_NAMES.get(memory, memory)}
            for column in columns:
                column_result = {}
                for split, name in (("overall", "avg"), ("explicit", "explicit"), ("implicit", "implicit")):
                    correct = total = 0
                    for category in categories:
                        group = virtual_group(all_data[column], model, memory, category)
                        if group:
                            correct += group[split]["correct"]
                            total += group[split]["total"]
                    column_result[name] = accuracy(correct, total) if total else None
                    column_result[f"{name}_detail"] = {"correct": correct, "total": total}
                row[column] = column_result
            rows.append(row)
        if rows:
            output[combined_name] = rows
    return output


def run_cross_mode(args: argparse.Namespace, *, audio: bool) -> None:
    result_dir = Path(args.result_dir or RESULT_ROOT).expanduser().resolve()
    raw_columns = args.audio_models if audio else args.captions
    columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
    all_data = collect_cross_data(result_dir, columns, audio=audio)
    if not all_data:
        raise SystemExit(f"No result directories found under: {result_dir}")

    filters: set[str | None] = set()
    for data in all_data.values():
        for _model, _memory, category in data:
            filters.add(category)
            if category is None:
                for group in data.values():
                    filters.update(group["per_category"])
    if args.category is not None:
        filters = {normalize_category(args.category)}
    filters = sorted(filters, key=lambda value: (value is not None, value or ""))

    output: dict[str, Any] = {}
    columns_found = list(all_data)
    per_category_output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for category in filters:
        rows = cross_rows(all_data, columns_found, category, args.model)
        if not rows:
            continue
        print_cross_table(
            rows,
            columns_found,
            "Audio Caption Model Comparison" if audio else "Caption Granularity Comparison",
            category,
        )
        output[category or "__full__"] = rows
        if args.per_category:
            detailed = per_category_cross_rows(
                all_data,
                columns_found,
                category,
                args.model,
            )
            if detailed:
                filter_key = category or "__full__"
                per_category_output[filter_key] = detailed
                for qa_category, detailed_rows in detailed.items():
                    print_cross_table(
                        detailed_rows,
                        columns_found,
                        (
                            "Audio Caption Model Comparison"
                            if audio
                            else "Caption Granularity Comparison"
                        )
                        + f" — QA category {qa_category}",
                        qa_category,
                    )

    if per_category_output:
        output["__per_category__"] = per_category_output

    combined = combined_rows(all_data, columns_found, args.model)
    if combined:
        output["__combined__"] = combined

    default_name = "aggregated_by_audio_caption.json" if audio else "aggregated_by_caption.json"
    output_path = Path(args.output) if args.output else result_dir / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一汇总 RQ1/RQ2 结果；用 --mode 选择原五类统计任务。"
    )
    parser.add_argument(
        "--mode",
        choices=("overall", "question-only", "adversarial", "caption", "audio-caption"),
        default="overall",
        help="统计模式：overall、question-only、adversarial、caption 或 audio-caption。",
    )
    parser.add_argument("--result-dir", "--result_dir", dest="result_dir", default=None)
    parser.add_argument("--output", default=None, help="输出 JSON 路径；默认写入对应结果目录。")
    parser.add_argument("--category", default=None, help="只统计指定 QA category，例如 pref_text。")
    parser.add_argument("--model", default=None, help="只展示指定 backbone/model。")
    parser.add_argument(
        "--all",
        action="store_true",
        help="兼容旧参数；横向比较模式默认已展示所有可用 category filter。",
    )
    parser.add_argument(
        "--per-category",
        "--per_category",
        dest="per_category",
        action="store_true",
        help="横向比较模式额外输出各 QA 子类别统计。",
    )
    parser.add_argument("--point", default=None, help="question-only 模式保留的旧参数。")
    parser.add_argument(
        "--pattern",
        default="history_with_qa_p*_adversarial_text_results.json",
        help="adversarial 模式的文件 glob。",
    )
    parser.add_argument(
        "--captions",
        default=",".join(CAPTION_NAMES),
        help="caption 模式的列，逗号分隔；可加入 base。",
    )
    parser.add_argument(
        "--audio-models",
        "--audio_models",
        dest="audio_models",
        default=",".join(AUDIO_CAPTION_NAMES),
        help="audio-caption 模式的列，逗号分隔。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.category = normalize_category(args.category)
    if args.mode == "overall":
        run_overall(args)
    elif args.mode == "question-only":
        run_question_only(args)
    elif args.mode == "adversarial":
        run_adversarial(args)
    elif args.mode == "caption":
        run_cross_mode(args, audio=False)
    else:
        run_cross_mode(args, audio=True)


if __name__ == "__main__":
    main()
