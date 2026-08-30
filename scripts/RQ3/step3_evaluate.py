"""Step 3: 汇总实验结果，计算各种指标，输出表格。

用法:
    python -m scripts.RQ3.step3_evaluate --result-root <path>
    python -m scripts.RQ3.step3_evaluate --variants TT MM
    python -m scripts.RQ3.step3_evaluate --oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import (
    ADVERSARIAL_QA_FILE as DEFAULT_ADVERSARIAL_QA_FILE,
)
from .config import (
    ENTITY_SOURCE_DIR,
    PROJECT_ROOT,
    RESULT_DIR,
    RETRIEVAL_TOP_K_LIST,
    VARIANTS,
    resolve_path,
)
from .evaluator import compute_accuracy, compute_implicit_gap, compute_retrieval_metrics

ADVERSARIAL_QA_FILE: Path | None = DEFAULT_ADVERSARIAL_QA_FILE
BENCHMARK_BASE_DIALOG_DIR: Path | None = ENTITY_SOURCE_DIR

DATA_FORM_GROUPS = [
    ("long_pattern", "Long Pattern", ("pref_text", "pref_img")),
    ("entity_recall", "Entity Recall", ("entity_text", "entity_img")),
    ("personalized_recommendation", "Personalized Recommendation", ("rec_text", "rec_img")),
    ("answer_refusal", "Answer Refusal", ("refusal_text", "adversarial_text")),
]

ENTITY_EXPLICIT_TYPES = {"relationship", "pets"}
ENTITY_IMPLICIT_TYPES = {"items"}

_REFUSAL_EXPRESSION_BY_QA_ID: dict[str, str] | None = None
_ENTITY_EXPLICITNESS_BY_KEY: dict[str, str] | None = None


def _display_path(path: Path | None) -> str:
    if path is None:
        return "<not configured>"
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "<external path>"


def _load_refusal_expression_map() -> dict[str, str]:
    """Load refusal explicit/implicit labels from the adversarial QA source file."""
    global _REFUSAL_EXPRESSION_BY_QA_ID
    if _REFUSAL_EXPRESSION_BY_QA_ID is not None:
        return _REFUSAL_EXPRESSION_BY_QA_ID

    mapping: dict[str, str] = {}
    if ADVERSARIAL_QA_FILE is None or not ADVERSARIAL_QA_FILE.exists():
        print(f"[WARN] Refusal expression source not found: {ADVERSARIAL_QA_FILE}")
        _REFUSAL_EXPRESSION_BY_QA_ID = mapping
        return mapping

    with open(ADVERSARIAL_QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[WARN] Unsupported refusal expression source shape: {ADVERSARIAL_QA_FILE}")
        _REFUSAL_EXPRESSION_BY_QA_ID = mapping
        return mapping

    for row in data:
        if not isinstance(row, dict):
            continue
        qa_id = str(row.get("qa_id", "")).strip()
        expression_type = str(row.get("expression_type", "")).strip().lower()
        if qa_id and expression_type in {"explicit", "implicit"}:
            mapping[qa_id] = expression_type

    _REFUSAL_EXPRESSION_BY_QA_ID = mapping
    return mapping


def _load_entity_explicitness_map() -> dict[str, str]:
    """Load Items explicit/implicit labels from benchmark input QAs.

    RQ3 result files may only keep qa_type="Items" for entity recall QAs.
    The benchmark input built by qa/build_bench_input.py stores the corrected
    entity_explicitness field, so we reuse it here to avoid treating all Items
    as implicit.
    """
    global _ENTITY_EXPLICITNESS_BY_KEY
    if _ENTITY_EXPLICITNESS_BY_KEY is not None:
        return _ENTITY_EXPLICITNESS_BY_KEY

    mapping: dict[str, str] = {}
    if BENCHMARK_BASE_DIALOG_DIR is None or not BENCHMARK_BASE_DIALOG_DIR.exists():
        print(f"[WARN] Entity explicitness source not found: {BENCHMARK_BASE_DIALOG_DIR}")
        _ENTITY_EXPLICITNESS_BY_KEY = mapping
        return mapping

    for path in sorted(BENCHMARK_BASE_DIALOG_DIR.glob("history_with_qa_p*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, TypeError, ValueError) as exc:
            print(f"[WARN] Failed to load entity explicitness source {path}: {exc}")
            continue

        profile = payload.get("character_profile", {})
        p_id = profile.get("p_id")
        if p_id is None:
            try:
                p_id = int(path.stem.rsplit("_p", 1)[-1])
            except ValueError:
                p_id = None

        for qa in payload.get("human-annotated QAs", []):
            if not isinstance(qa, dict):
                continue
            entity_explicitness = str(qa.get("entity_explicitness", "")).strip().lower()
            if entity_explicitness not in {"explicit", "implicit"}:
                continue
            qa_id = str(qa.get("qa_id", "")).strip()
            point = str(qa.get("point", "")).strip()
            qa_uid = str(qa.get("qa_uid", "")).strip()
            if qa_uid:
                mapping[qa_uid] = entity_explicitness
            if p_id is not None and qa_id and point:
                mapping[f"p{p_id}::{point}::{qa_id}"] = entity_explicitness
            if qa_id:
                mapping[qa_id] = entity_explicitness

    _ENTITY_EXPLICITNESS_BY_KEY = mapping
    return mapping


def _entity_explicitness(row: dict) -> str | None:
    value = str(row.get("entity_explicitness", "")).strip().lower()
    if value in {"explicit", "implicit"}:
        return value
    mapping = _load_entity_explicitness_map()
    for key_name in ("qa_uid", "qa_id"):
        key = str(row.get(key_name, "")).strip()
        if key and mapping.get(key) in {"explicit", "implicit"}:
            return mapping[key]
    return None


def _attach_entity_explicitness(row: dict) -> dict:
    """Attach entity_explicitness to a result row when it can be inferred."""
    if str(row.get("entity_explicitness", "")).strip().lower() in {"explicit", "implicit"}:
        return row
    value = _entity_explicitness(row)
    if value in {"explicit", "implicit"}:
        row = dict(row)
        row["entity_explicitness"] = value
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 结果汇总")
    parser.add_argument("--variants", type=str, nargs="+", default=VARIANTS)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "统计 oracle_evidence 结果。默认递归扫描 "
            "RQ3/results/oracle_evidence，并输出 Oracle 专用表格。"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_DIR,
        help=(
            "结果根目录。可以是具体实验目录，例如 "
            "RQ3/results/aliyun_qwen3.5-omni-plus；也可以是 RQ3/results，"
            "此时会自动发现其下包含 TT/TM/MT/MM 的实验目录。"
        ),
    )
    parser.add_argument(
        "--adversarial-qa-file",
        type=Path,
        default=None,
        help="Optional JSON source containing refusal explicit/implicit labels.",
    )
    parser.add_argument(
        "--entity-source-dir",
        type=Path,
        default=None,
        help="Optional directory containing benchmark QA JSON for entity labels.",
    )
    return parser.parse_args()


def _has_variant_dirs(path: Path, variants: list[str]) -> bool:
    return any((path / variant).is_dir() for variant in variants)


def discover_result_roots(result_root: Path, variants: list[str]) -> list[Path]:
    """发现实际实验结果根目录。

    step2 的结果目录形如:
        RQ3/results/aliyun_qwen3.5-omni-plus/TT/p0_results.json

    老版 step3 默认查:
        RQ3/results/TT/p0_results.json

    因此当传入 RQ3/results 时，需要向下一层自动发现包含 variant 子目录的
    实验目录；当传入的本身就是实验目录时，则直接使用。
    """
    result_root = result_root.resolve()
    if _has_variant_dirs(result_root, variants):
        return [result_root]
    if not result_root.exists():
        return []
    roots = [
        child
        for child in sorted(result_root.iterdir())
        if child.is_dir() and _has_variant_dirs(child, variants)
    ]
    return roots


def load_variant_results(variant: str, result_root: Path = RESULT_DIR) -> list[dict]:
    """加载一个 variant 的所有 profile 结果。"""
    all_results = []
    variant_dir = result_root / variant
    if not variant_dir.exists():
        return []
    for p_file in sorted(variant_dir.glob("p*_results.json")):
        with open(p_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and isinstance(data.get("results"), list):
            results = data["results"]
        else:
            print(f"[WARN] Unsupported result file shape: {p_file}")
            continue
        all_results.extend(_attach_entity_explicitness(row) for row in results if isinstance(row, dict))
    return all_results


def discover_oracle_result_roots(result_root: Path) -> list[Path]:
    """发现包含 p*_results.json 的 Oracle 模型结果目录。"""
    result_root = result_root.resolve()
    if not result_root.exists():
        return []
    if any(result_root.glob("p*_results.json")):
        return [result_root]
    return sorted({
        path.parent
        for path in result_root.rglob("p*_results.json")
        if path.is_file()
    })


def load_oracle_results(result_root: Path) -> list[dict]:
    """加载一个 Oracle 模型目录下的全部 profile 结果。"""
    all_results = []
    for p_file in sorted(result_root.glob("p*_results.json")):
        with open(p_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and isinstance(data.get("results"), list):
            results = data["results"]
        else:
            print(f"[WARN] Unsupported Oracle result file shape: {p_file}")
            continue
        all_results.extend(_attach_entity_explicitness(row) for row in results if isinstance(row, dict))
    return all_results


def _metric_at(metric_dict: dict, k: int) -> float:
    """JSON 会把 dict key 写成字符串；这里同时兼容 int 和 str key。"""
    return metric_dict.get(k, metric_dict.get(str(k), 0.0))


def _is_correct(row: dict) -> bool:
    return str(row.get("model_answer", "")).strip() == str(row.get("answer", "")).strip()


def _form_split(row: dict, group_key: str) -> str | None:
    qa_type = str(row.get("qa_type", "")).strip()
    qa_type_lower = qa_type.lower()

    if group_key in {"long_pattern", "personalized_recommendation"}:
        if qa_type_lower in {"explicit", "implicit"}:
            return qa_type_lower
        return None

    if group_key == "entity_recall":
        if qa_type_lower in ENTITY_EXPLICIT_TYPES:
            return "explicit"
        if qa_type_lower in ENTITY_IMPLICIT_TYPES:
            split = _entity_explicitness(row)
            return split if split in {"explicit", "implicit"} else "implicit"
        if qa_type_lower in {"explicit", "implicit"}:
            return qa_type_lower
        return None

    if group_key == "answer_refusal":
        expression_type = str(row.get("expression_type", "")).strip().lower()
        if expression_type in {"explicit", "implicit"}:
            return expression_type
        qa_id = str(row.get("qa_id", "")).strip()
        split = _load_refusal_expression_map().get(qa_id)
        if split in {"explicit", "implicit"}:
            return split
        if qa_type_lower in {"explicit", "implicit"}:
            return qa_type_lower
        return None

    return None


def _empty_counts() -> dict[str, dict[str, int]]:
    return {
        "explicit": {"correct": 0, "total": 0},
        "implicit": {"correct": 0, "total": 0},
    }


def _acc_from_counts(counts: dict[str, int]) -> float | None:
    total = counts.get("total", 0)
    if total <= 0:
        return None
    return counts.get("correct", 0) / total


def _pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}"


def _pct_console(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def compute_data_form_variant_metrics(results: list[dict]) -> dict:
    """计算 data_form.png 需要的四大类 explicit/implicit/EI Gap。

    pref/rec/entity/refusal 都按各自包含的 point 直接累加 correct/total；
    因此 pref_text 与 pref_img 等组合是按 QA 数量加权的。
    """
    output = {}
    for group_key, label, points in DATA_FORM_GROUPS:
        counts = _empty_counts()
        for row in results:
            if row.get("point") not in points:
                continue
            split = _form_split(row, group_key)
            if split not in counts:
                continue
            counts[split]["total"] += 1
            if _is_correct(row):
                counts[split]["correct"] += 1

        exp_acc = _acc_from_counts(counts["explicit"])
        imp_acc = _acc_from_counts(counts["implicit"])
        gap = (exp_acc - imp_acc) if exp_acc is not None and imp_acc is not None else None
        output[group_key] = {
            "label": label,
            "points": list(points),
            "explicit": {
                **counts["explicit"],
                "accuracy": exp_acc,
            },
            "implicit": {
                **counts["implicit"],
                "accuracy": imp_acc,
            },
            "ei_gap": gap,
        }
    return output


def build_data_form_rows(backbone: str, variant_results: dict[str, list[dict]]) -> list[dict]:
    rows = []

    for variant in VARIANTS:
        if variant not in variant_results:
            continue
        metrics = compute_data_form_variant_metrics(variant_results[variant])
        row = {"Backbone": backbone, "Memory": variant}
        explicit_values: list[float] = []
        implicit_values: list[float] = []
        gap_values: list[float] = []
        for group_key, label, _points in DATA_FORM_GROUPS:
            group = metrics[group_key]
            for split_key, col_suffix in (
                ("explicit", "explicit"),
                ("implicit", "implicit"),
            ):
                col = f"{label} {col_suffix}"
                value = group[split_key]["accuracy"]
                row[col] = value
                if value is not None:
                    if split_key == "explicit":
                        explicit_values.append(value)
                    else:
                        implicit_values.append(value)
            gap_col = f"{label} EI Gap"
            row[gap_col] = group["ei_gap"]
            if group["ei_gap"] is not None:
                gap_values.append(group["ei_gap"])
            row[f"{label} explicit detail"] = {
                "correct": group["explicit"]["correct"],
                "total": group["explicit"]["total"],
            }
            row[f"{label} implicit detail"] = {
                "correct": group["implicit"]["correct"],
                "total": group["implicit"]["total"],
            }
        row["AVG. Explicit AVG."] = (
            sum(explicit_values) / len(explicit_values) if explicit_values else None
        )
        row["AVG. Implicit AVG."] = (
            sum(implicit_values) / len(implicit_values) if implicit_values else None
        )
        row["AVG. EI Gap AVG."] = (
            sum(gap_values) / len(gap_values) if gap_values else None
        )
        rows.append(row)

    return rows


def build_oracle_data_form_row(backbone: str, results: list[dict]) -> dict:
    """构造 oracle_evidence_data_form.png 对应的单行统计。"""
    metrics = compute_data_form_variant_metrics(results)
    row = {"Backbone": backbone, "Memory": "oracle_evidence"}
    explicit_values: list[float] = []
    implicit_values: list[float] = []
    gap_values: list[float] = []

    for group_key, label, _points in DATA_FORM_GROUPS:
        group = metrics[group_key]
        explicit = group["explicit"]["accuracy"]
        implicit = group["implicit"]["accuracy"]
        gap = group["ei_gap"]
        row[f"{label} explicit"] = explicit
        row[f"{label} implicit"] = implicit
        row[f"{label} EI Gap"] = gap
        row[f"{label} explicit detail"] = {
            "correct": group["explicit"]["correct"],
            "total": group["explicit"]["total"],
        }
        row[f"{label} implicit detail"] = {
            "correct": group["implicit"]["correct"],
            "total": group["implicit"]["total"],
        }
        if explicit is not None:
            explicit_values.append(explicit)
        if implicit is not None:
            implicit_values.append(implicit)
        if gap is not None:
            gap_values.append(gap)

    row["AVG. Explicit AVG."] = (
        sum(explicit_values) / len(explicit_values) if explicit_values else None
    )
    row["AVG. Implicit AVG."] = (
        sum(implicit_values) / len(implicit_values) if implicit_values else None
    )
    row["AVG. EI Gap AVG."] = (
        sum(gap_values) / len(gap_values) if gap_values else None
    )
    return row


def print_accuracy_table(variant_metrics: dict[str, dict]) -> None:
    """打印准确率对比表格。"""
    points = ["pref_text", "pref_img", "rec_text", "rec_img",
              "entity_text", "entity_img", "adversarial_text"]

    print("\n" + "="*100)
    print("Table 1: Downstream QA Accuracy")
    print("="*100)

    header = f"{'Variant':<8}"
    header += f"{'Overall':>8}"
    for p in points:
        short = p.replace("_text", "_T").replace("_img", "_I")
        if p == "adversarial_text":
            short = "adv_T"
        header += f"{short:>10}"
    header += f"{'Explicit':>10}{'Implicit':>10}{'Gap':>8}"
    print(header)
    print("-"*100)

    for variant, metrics in variant_metrics.items():
        acc = metrics["accuracy"]
        gap = metrics["implicit_gap"]
        row = f"{variant:<8}"
        row += f"{acc['overall']['accuracy']*100:>7.1f}%"
        for p in points:
            if p in acc["by_point"]:
                row += f"{acc['by_point'][p]['accuracy']*100:>9.1f}%"
            else:
                row += f"{'—':>10}"
        exp_acc = gap.get("explicit_acc", 0) * 100
        imp_acc = gap.get("implicit_acc", 0) * 100
        g = gap.get("gap", 0) * 100
        row += f"{exp_acc:>9.1f}%{imp_acc:>9.1f}%{g:>7.1f}%"
        print(row)


def print_retrieval_table(variant_metrics: dict[str, dict]) -> None:
    """打印 Retrieval 的 Recall@k 和 Precision@k 指标表格。"""
    k_list = RETRIEVAL_TOP_K_LIST
    table_width = 24 + 10 * len(k_list) * 2

    print("\n" + "=" * table_width)
    print("Table 2: Evidence Retrieval (Recall@k / Precision@k)")
    print("=" * table_width)

    header = f"{'Index':<12}{'Expression':<12}"
    for k in k_list:
        header += f"{'R@'+str(k):>10}"
    for k in k_list:
        header += f"{'P@'+str(k):>10}"
    print(header)
    print("-" * table_width)

    for variant in ["TT", "MT"]:
        if variant not in variant_metrics:
            continue
        ret = variant_metrics[variant].get("retrieval", {})
        index_label = "Text" if variant == "TT" else "MM"

        # Overall
        overall = ret.get("overall", {})
        row = f"{index_label:<12}{'overall':<12}"
        for k in k_list:
            val = _metric_at(overall.get("recall@k", {}), k)
            row += f"{val*100:>9.1f}%"
        for k in k_list:
            val = _metric_at(overall.get("precision@k", {}), k)
            row += f"{val*100:>9.1f}%"
        print(row)

        # By expression
        by_expr = ret.get("by_expression", {})
        for expr in ["explicit", "implicit"]:
            if expr in by_expr:
                row = f"{'':>12}{expr:<12}"
                for k in k_list:
                    val = _metric_at(by_expr[expr].get("recall@k", {}), k)
                    row += f"{val*100:>9.1f}%"
                for k in k_list:
                    val = _metric_at(by_expr[expr].get("precision@k", {}), k)
                    row += f"{val*100:>9.1f}%"
                print(row)
        print()


def data_form_columns() -> list[str]:
    columns = ["Backbone", "Memory"]
    for _group_key, label, _points in DATA_FORM_GROUPS:
        columns.extend([
            f"{label} explicit",
            f"{label} implicit",
            f"{label} EI Gap",
        ])
    columns.extend([
        "AVG. Explicit AVG.",
        "AVG. Implicit AVG.",
        "AVG. EI Gap AVG.",
    ])
    return columns


def print_data_form_table(
    rows: list[dict],
    title: str = "Table 0: Data Form Accuracy",
) -> None:
    if not rows:
        return
    console_columns = [
        ("Backbone", "Backbone", 26),
        ("Memory", "Mem", 18),
        ("Long Pattern explicit", "LP-Exp", 9),
        ("Long Pattern implicit", "LP-Imp", 9),
        ("Long Pattern EI Gap", "LP-Gap", 9),
        ("Entity Recall explicit", "Ent-Exp", 9),
        ("Entity Recall implicit", "Ent-Imp", 9),
        ("Entity Recall EI Gap", "Ent-Gap", 9),
        ("Personalized Recommendation explicit", "Rec-Exp", 9),
        ("Personalized Recommendation implicit", "Rec-Imp", 9),
        ("Personalized Recommendation EI Gap", "Rec-Gap", 9),
        ("Answer Refusal explicit", "Ref-Exp", 9),
        ("Answer Refusal implicit", "Ref-Imp", 9),
        ("Answer Refusal EI Gap", "Ref-Gap", 9),
        ("AVG. Explicit AVG.", "Avg-Exp", 9),
        ("AVG. Implicit AVG.", "Avg-Imp", 9),
        ("AVG. EI Gap AVG.", "Avg-Gap", 9),
    ]

    total_width = sum(width for _key, _label, width in console_columns)
    print("\n" + "=" * total_width)
    print(title)
    print("LP=pref, Ent=entity, Rec=recommendation, Ref=refusal; Gap=Exp-Imp")
    print("=" * total_width)
    header = "".join(f"{label:<{width}}" for _key, label, width in console_columns)
    print(header)
    print("-" * total_width)
    previous_backbone = None
    for row in rows:
        display_row = dict(row)
        if display_row["Backbone"] == previous_backbone:
            display_row["Backbone"] = ""
        else:
            previous_backbone = display_row["Backbone"]
        line = ""
        for key, _label, width in console_columns:
            value = display_row.get(key, "")
            if isinstance(value, float) or value is None:
                text = _pct_console(value)
            else:
                text = str(value)
            line += f"{text:<{width}}"
        print(line)


def save_data_form_table(
    rows: list[dict],
    output_dir: Path,
    filename_stem: str = "data_form_table",
) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = data_form_columns()
    csv_path = output_dir / f"{filename_stem}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                col: _pct(row.get(col)) if isinstance(row.get(col), float) or row.get(col) is None else row.get(col, "")
                for col in columns
            })

    json_path = output_dir / f"{filename_stem}.json"
    json_rows = []
    for row in rows:
        item = {"Backbone": row.get("Backbone"), "Memory": row.get("Memory")}
        for _group_key, label, _points in DATA_FORM_GROUPS:
            item[label] = {
                "explicit": row.get(f"{label} explicit"),
                "implicit": row.get(f"{label} implicit"),
                "ei_gap": row.get(f"{label} EI Gap"),
                "explicit_detail": row.get(f"{label} explicit detail", {}),
                "implicit_detail": row.get(f"{label} implicit detail", {}),
            }
        item["AVG."] = {
            "explicit_avg": row.get("AVG. Explicit AVG."),
            "implicit_avg": row.get("AVG. Implicit AVG."),
            "ei_gap_avg": row.get("AVG. EI Gap AVG."),
        }
        json_rows.append(item)
    payload = {
        "notes": {
            "long_pattern": "pref_text + pref_img weighted by QA count",
            "entity_recall": "entity_text + entity_img weighted by QA count; Relationship/Pets=explicit; Items use entity_explicitness when available, otherwise fall back to implicit",
            "personalized_recommendation": "rec_text + rec_img weighted by QA count",
            "answer_refusal": (
                "refusal_text/adversarial_text; explicit/implicit labels are read from "
                f"{_display_path(ADVERSARIAL_QA_FILE)} expression_type"
            ),
            "avg": (
                "For each memory variant, arithmetic mean across Long Pattern, "
                "Entity Recall, Personalized Recommendation, and Answer Refusal. "
                "EI Gap AVG. is the arithmetic mean of the four category EI gaps."
            ),
        },
        "rows": json_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Data form CSV: {csv_path}")
    print(f"Data form JSON: {json_path}")


def evaluate_oracle_results(args: argparse.Namespace) -> None:
    oracle_root = args.result_root.resolve()
    if oracle_root == RESULT_DIR.resolve():
        oracle_root = RESULT_DIR / "oracle_evidence"

    result_roots = discover_oracle_result_roots(oracle_root)
    if not result_roots:
        print(f"No Oracle result roots found under {oracle_root}.")
        print("Expected layout example:")
        print(
            "  RQ3/results/oracle_evidence/multimodal/aliyun/"
            "qwen3.5-omni-plus/p0_results.json"
        )
        return

    if len(result_roots) > 1:
        print("Discovered Oracle result roots:")
        for root in result_roots:
            print(f"  - {root}")

    for result_root in result_roots:
        results = load_oracle_results(result_root)
        if not results:
            print(f"[WARN] No Oracle results under {result_root}, skipping")
            continue

        try:
            relative_parts = result_root.relative_to(oracle_root).parts
        except ValueError:
            relative_parts = result_root.parts
        input_mode = next(
            (part for part in relative_parts if part in {"text", "multimodal"}),
            str(results[0].get("oracle_input_mode", "unknown")),
        )
        provider = str(results[0].get("eval_provider", "unknown"))
        model = str(results[0].get("llm_name") or result_root.name)

        print("\n" + "#" * 100)
        print(f"Evaluating Oracle result root: {result_root}")
        print(f"Oracle input mode: {input_mode}")
        print(f"Eval provider: {provider}")
        print(f"Backbone: {model}")
        print(f"Oracle results loaded: {len(results)}")
        print("#" * 100)

        rows = [build_oracle_data_form_row(model, results)]
        print_data_form_table(
            rows,
            title=f"Oracle Evidence Data Form Accuracy ({input_mode})",
        )
        save_data_form_table(
            rows,
            result_root / "summary",
            filename_stem="oracle_evidence_data_form",
        )


def save_csv(variant_metrics: dict[str, dict], output_dir: Path) -> None:
    """保存 CSV 格式的结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy CSV
    acc_path = output_dir / "accuracy_table.csv"
    with open(acc_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        points = ["pref_text", "pref_img", "rec_text", "rec_img",
                  "entity_text", "entity_img", "refusal_text"]
        header = ["Variant", "Overall"] + points + ["Explicit", "Implicit", "Gap"]
        writer.writerow(header)
        for variant, metrics in variant_metrics.items():
            acc = metrics["accuracy"]
            gap = metrics["implicit_gap"]
            row = [variant, f"{acc['overall']['accuracy']:.4f}"]
            for p in points:
                if p in acc["by_point"]:
                    row.append(f"{acc['by_point'][p]['accuracy']:.4f}")
                else:
                    row.append("")
            row.extend([
                f"{gap.get('explicit_acc', 0):.4f}",
                f"{gap.get('implicit_acc', 0):.4f}",
                f"{gap.get('gap', 0):.4f}",
            ])
            writer.writerow(row)
    print(f"\nAccuracy CSV: {acc_path}")

    # Retrieval CSV
    ret_path = output_dir / "retrieval_metrics.csv"
    with open(ret_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        k_list = RETRIEVAL_TOP_K_LIST
        header = ["Variant", "Index", "Expression"] + [f"Recall@{k}" for k in k_list] + [f"Precision@{k}" for k in k_list]
        writer.writerow(header)
        for variant, metrics in variant_metrics.items():
            ret = metrics.get("retrieval", {})
            index_label = "Text" if variant.startswith("T") else "MM"
            overall = ret.get("overall", {})
            row = [variant, index_label, "overall"]
            for k in k_list:
                row.append(f"{_metric_at(overall.get('recall@k', {}), k):.4f}")
            for k in k_list:
                row.append(f"{_metric_at(overall.get('precision@k', {}), k):.4f}")
            writer.writerow(row)
            for expr in ["explicit", "implicit"]:
                expr_data = ret.get("by_expression", {}).get(expr, {})
                if expr_data:
                    row = [variant, index_label, expr]
                    for k in k_list:
                        row.append(f"{_metric_at(expr_data.get('recall@k', {}), k):.4f}")
                    for k in k_list:
                        row.append(f"{_metric_at(expr_data.get('precision@k', {}), k):.4f}")
                    writer.writerow(row)
    print(f"Retrieval CSV: {ret_path}")


def main():
    global ADVERSARIAL_QA_FILE, BENCHMARK_BASE_DIALOG_DIR
    global _REFUSAL_EXPRESSION_BY_QA_ID, _ENTITY_EXPLICITNESS_BY_KEY

    args = parse_args()
    args.result_root = resolve_path(args.result_root, RESULT_DIR)
    ADVERSARIAL_QA_FILE = resolve_path(
        args.adversarial_qa_file,
        DEFAULT_ADVERSARIAL_QA_FILE,
    )
    BENCHMARK_BASE_DIALOG_DIR = (
        resolve_path(args.entity_source_dir)
        if args.entity_source_dir is not None
        else ENTITY_SOURCE_DIR
    )
    _REFUSAL_EXPRESSION_BY_QA_ID = None
    _ENTITY_EXPLICITNESS_BY_KEY = None

    if args.oracle:
        evaluate_oracle_results(args)
        return

    result_roots = discover_result_roots(args.result_root, args.variants)
    if not result_roots:
        print(f"No result roots found under {args.result_root}. Run step2 first.")
        print("Expected layout examples:")
        print("  RQ3/results/aliyun_qwen3.5-omni-plus/TT/p0_results.json")
        print("  RQ3/results/imagebind_qwen3.6-omni/MM/p2_results.json")
        return

    if len(result_roots) > 1:
        print("Discovered result roots:")
        for root in result_roots:
            print(f"  - {root}")

    any_metrics = False
    for result_root in result_roots:
        print("\n" + "#" * 100)
        print(f"Evaluating result root: {result_root}")
        print("#" * 100)

        variant_metrics: dict[str, dict] = {}
        variant_results: dict[str, list[dict]] = {}

        for variant in args.variants:
            results = load_variant_results(variant, result_root)
            if not results:
                print(f"[WARN] No results for variant {variant}, skipping")
                continue

            print(f"\nVariant {variant}: {len(results)} results loaded")
            variant_results[variant] = results

            accuracy = compute_accuracy(results)
            retrieval = compute_retrieval_metrics(results)
            gap = compute_implicit_gap(results)

            variant_metrics[variant] = {
                "accuracy": accuracy,
                "retrieval": retrieval,
                "implicit_gap": gap,
                "num_results": len(results),
            }

        if not variant_metrics:
            print(f"No variant results found under {result_root}, skipping.")
            continue

        any_metrics = True
        data_form_rows = build_data_form_rows(result_root.name, variant_results)
        print_data_form_table(data_form_rows)
        print_accuracy_table(variant_metrics)
        print_retrieval_table(variant_metrics)

        summary_dir = result_root / "summary"
        save_data_form_table(data_form_rows, summary_dir)
        save_csv(variant_metrics, summary_dir)

        # 保存完整 JSON
        json_path = summary_dir / "full_metrics.json"

        def _convert(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(variant_metrics, f, ensure_ascii=False, indent=2, default=_convert)
        print(f"Full metrics JSON: {json_path}")

    if not any_metrics:
        print("No results found. Run step2 first.")


if __name__ == "__main__":
    main()
