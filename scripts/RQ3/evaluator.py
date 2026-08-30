"""评估模块：准确率、Recall@k、Precision@k、implicit gap。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

RETRIEVAL_EXPLICIT_QA_TYPES = {"explicit", "relationship", "pets"}
RETRIEVAL_IMPLICIT_QA_TYPES = {"implicit", "items"}


def _retrieval_expression(row: dict) -> str | None:
    """Return the explicit/implicit split used for evidence retrieval tables.

    RQ3 entity QA uses semantic qa_type labels instead of explicit/implicit.
    Relationship/Pets are explicit entity recall. Items use the
    entity_explicitness label attached by step3_evaluate when available, with
    implicit as a conservative fallback for older result files.
    """
    qa_type = str(row.get("qa_type", "")).strip().lower()
    if qa_type in RETRIEVAL_EXPLICIT_QA_TYPES:
        return "explicit"
    if qa_type == "items":
        entity_explicitness = str(row.get("entity_explicitness", "")).strip().lower()
        if entity_explicitness in {"explicit", "implicit"}:
            return entity_explicitness
        return "implicit"
    if qa_type in RETRIEVAL_IMPLICIT_QA_TYPES:
        return "implicit"
    return None


def compute_accuracy(results: list[dict]) -> dict[str, Any]:
    """按多个维度计算准确率。

    每条 result 需包含: model_answer, answer, point, qa_type, category
    """
    def _acc(items: list[dict]) -> float:
        if not items:
            return 0.0
        correct = sum(1 for r in items if r["model_answer"] == r["answer"])
        return correct / len(items)

    def _detail(items: list[dict]) -> dict:
        correct = sum(1 for r in items if r["model_answer"] == r["answer"])
        return {"accuracy": _acc(items), "correct": correct, "total": len(items)}

    # 整体
    overall = _detail(results)

    # 按 QA 类型 (point)
    by_point: dict[str, dict] = {}
    groups = _group_by(results, "point")
    for point, items in groups.items():
        by_point[point] = _detail(items)

    # 按 explicit / implicit
    by_expression: dict[str, dict] = {}
    explicit_items = [r for r in results if r.get("qa_type") in ("explicit",)]
    implicit_items = [r for r in results if r.get("qa_type") in ("implicit",)]
    if explicit_items:
        by_expression["explicit"] = _detail(explicit_items)
    if implicit_items:
        by_expression["implicit"] = _detail(implicit_items)

    # 按 text QA / image QA
    by_modality: dict[str, dict] = {}
    text_qas = [r for r in results if r.get("point", "").endswith("_text")]
    img_qas = [r for r in results if r.get("point", "").endswith("_img")]
    if text_qas:
        by_modality["text_qa"] = _detail(text_qas)
    if img_qas:
        by_modality["image_qa"] = _detail(img_qas)

    # 按类别 (category)
    by_category: dict[str, dict] = {}
    cat_groups = _group_by(results, "category")
    for cat, items in cat_groups.items():
        if cat:
            by_category[cat] = _detail(items)

    return {
        "overall": overall,
        "by_point": by_point,
        "by_expression": by_expression,
        "by_modality": by_modality,
        "by_category": by_category,
    }


def compute_retrieval_metrics(results: list[dict]) -> dict[str, Any]:
    """汇总 Recall@k 和 Precision@k（排除 refusal 类）。

    每条 result 需包含: recall@k, precision@k (已由 retriever 计算)
    """
    non_refusal = [r for r in results if r.get("point") != "refusal_text"]
    if not non_refusal:
        return {}

    classified = [
        (r, expr)
        for r in non_refusal
        if (expr := _retrieval_expression(r)) is not None
    ]
    if not classified:
        return {}

    k_values: set[int] = set()
    for r in non_refusal:
        rk = r.get("recall@k", {})
        for k in rk:
            try:
                k_values.add(int(k))
            except (TypeError, ValueError):
                continue
    k_values = sorted(k_values)

    def _metric_value(metric_dict: dict, k: int) -> float:
        return metric_dict.get(k, metric_dict.get(str(k), 0.0))

    def _avg_metric(items: list[dict], metric_key: str) -> dict[int, float]:
        avgs = {}
        for k in k_values:
            vals = [_metric_value(r.get(metric_key, {}), k) for r in items]
            avgs[k] = sum(vals) / len(vals) if vals else 0.0
        return avgs

    classified_items = [r for r, _expr in classified]
    overall_recall = _avg_metric(classified_items, "recall@k")
    overall_precision = _avg_metric(classified_items, "precision@k")

    # 按 explicit / implicit。entity 的 relationship/pets/items 在
    # _retrieval_expression 中映射，避免 overall 和分组行口径不一致。
    explicit = [r for r, expr in classified if expr == "explicit"]
    implicit = [r for r, expr in classified if expr == "implicit"]

    by_expression = {}
    if explicit:
        by_expression["explicit"] = {
            "recall@k": _avg_metric(explicit, "recall@k"),
            "precision@k": _avg_metric(explicit, "precision@k"),
            "count": len(explicit),
        }
    if implicit:
        by_expression["implicit"] = {
            "recall@k": _avg_metric(implicit, "recall@k"),
            "precision@k": _avg_metric(implicit, "precision@k"),
            "count": len(implicit),
        }

    return {
        "overall": {
            "recall@k": overall_recall,
            "precision@k": overall_precision,
            "count": len(classified_items),
            "unclassified_count": len(non_refusal) - len(classified_items),
        },
        "by_expression": by_expression,
    }


def compute_implicit_gap(results: list[dict]) -> dict[str, float]:
    """implicit gap = explicit_acc - implicit_acc。"""
    explicit = [r for r in results if r.get("qa_type") == "explicit"]
    implicit = [r for r in results if r.get("qa_type") == "implicit"]

    def _acc(items):
        if not items:
            return 0.0
        return sum(1 for r in items if r["model_answer"] == r["answer"]) / len(items)

    exp_acc = _acc(explicit)
    imp_acc = _acc(implicit)
    return {
        "explicit_acc": exp_acc,
        "implicit_acc": imp_acc,
        "gap": exp_acc - imp_acc,
    }


def _group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item.get(key, "unknown")].append(item)
    return dict(groups)
