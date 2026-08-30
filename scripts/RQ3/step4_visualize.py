"""Step 4: 生成可视化图表 — Recall@k 折线图、准确率柱状图。

用法:
    python -m scripts.RQ3.step4_visualize --metrics-file <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import RESULT_DIR, resolve_path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _metric_at(metric_dict: dict, k: int) -> float:
    """Read JSON metric maps whose keys may be strings or integers."""

    return metric_dict.get(k, metric_dict.get(str(k), 0.0))


def load_metrics(path: Path) -> dict:
    if not path.exists():
        print(f"Metrics file not found: {path}")
        print("Run step3_evaluate.py first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_font():
    """尝试设置中文字体。"""
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei"]:
        fonts = fm.findSystemFonts()
        for fp in fonts:
            try:
                prop = fm.FontProperties(fname=fp)
                if name.lower() in prop.get_name().lower():
                    plt.rcParams["font.sans-serif"] = [name]
                    plt.rcParams["axes.unicode_minus"] = False
                    return
            except (OSError, RuntimeError, ValueError):
                continue
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_recall_at_k(metrics: dict, output_dir: Path) -> None:
    """Recall@k 折线图: Text-Index vs MM-Index。"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    styles = {
        "TT": {"color": "#1f77b4", "marker": "o", "label": "Text-Index"},
        "MT": {"color": "#ff7f0e", "marker": "s", "label": "MM-Index"},
    }

    for variant, style in styles.items():
        if variant not in metrics:
            continue
        ret = metrics[variant].get("retrieval", {}).get("overall", {})
        recall = ret.get("recall@k", {})
        if not recall:
            continue
        ks = sorted(int(k) for k in recall)
        vals = [_metric_at(recall, k) * 100 for k in ks]
        ax.plot(ks, vals, marker=style["marker"], color=style["color"],
                label=style["label"], linewidth=2, markersize=8)

    ax.set_xlabel("k", fontsize=13)
    ax.set_ylabel("Supporting Evidence Recall@k (%)", fontsize=13)
    ax.set_title("Retrieval Recall: Text-Index vs MM-Index", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 3, 5, 10])

    path = output_dir / "recall_at_k.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recall@k plot: {path}")


def plot_recall_by_expression(metrics: dict, output_dir: Path) -> None:
    """Recall@k 折线图按 explicit / implicit 分拆。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    styles = {
        "TT": {"color": "#1f77b4", "marker": "o", "label": "Text-Index"},
        "MT": {"color": "#ff7f0e", "marker": "s", "label": "MM-Index"},
    }

    for col, expr in enumerate(["explicit", "implicit"]):
        ax = axes[col]
        for variant, style in styles.items():
            if variant not in metrics:
                continue
            by_expr = metrics[variant].get("retrieval", {}).get("by_expression", {})
            recall = by_expr.get(expr, {}).get("recall@k", {})
            if not recall:
                continue
            ks = sorted(int(k) for k in recall)
            vals = [_metric_at(recall, k) * 100 for k in ks]
            ax.plot(ks, vals, marker=style["marker"], color=style["color"],
                    label=style["label"], linewidth=2, markersize=8)

        ax.set_xlabel("k", fontsize=13)
        ax.set_ylabel("Recall@k (%)" if col == 0 else "", fontsize=13)
        ax.set_title(f"{expr.capitalize()} Samples", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 3, 5, 10])

    path = output_dir / "recall_by_expression.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Recall by expression: {path}")


def plot_accuracy_bar(metrics: dict, output_dir: Path) -> None:
    """准确率柱状图: 4 variants 对比，分 explicit/implicit。"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    variants = [v for v in ["TT", "TM", "MT", "MM"] if v in metrics]
    if not variants:
        return

    x_labels = []
    explicit_vals = []
    implicit_vals = []

    for v in variants:
        gap = metrics[v].get("implicit_gap", {})
        x_labels.append(v)
        explicit_vals.append(gap.get("explicit_acc", 0) * 100)
        implicit_vals.append(gap.get("implicit_acc", 0) * 100)

    import numpy as np
    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax.bar(x - width/2, explicit_vals, width, label="Explicit", color="#5B9BD5")
    bars2 = ax.bar(x + width/2, implicit_vals, width, label="Implicit", color="#ED7D31")

    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("QA Accuracy by Pipeline Variant", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    path = output_dir / "accuracy_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Accuracy bar chart: {path}")


def generate_html_report(metrics: dict, output_dir: Path) -> None:
    """生成 HTML 综合报告。"""
    variants = [v for v in ["TT", "TM", "MT", "MM"] if v in metrics]

    rows_html = ""
    for v in variants:
        acc = metrics[v]["accuracy"]["overall"]
        gap = metrics[v].get("implicit_gap", {})
        rows_html += f"""
        <tr>
            <td><strong>{v}</strong></td>
            <td>{acc['accuracy']*100:.1f}%</td>
            <td>{acc['correct']}/{acc['total']}</td>
            <td>{gap.get('explicit_acc', 0)*100:.1f}%</td>
            <td>{gap.get('implicit_acc', 0)*100:.1f}%</td>
            <td>{gap.get('gap', 0)*100:.1f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RQ3 Experiment Results</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1200px; margin: 0 auto; padding: 24px; background: #f5f5f5; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 20px 0; }}
th, td {{ padding: 12px 16px; text-align: center; border-bottom: 1px solid #eee; }}
th {{ background: #f0f0f0; font-weight: 600; }}
tr:hover {{ background: #fafafa; }}
.chart {{ text-align: center; margin: 24px 0; }}
.chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
</style>
</head>
<body>
<h1>RQ3: Multimodal Memory Experiment Results</h1>

<h2>Accuracy Overview</h2>
<table>
<thead><tr><th>Variant</th><th>Overall Acc</th><th>Correct/Total</th><th>Explicit</th><th>Implicit</th><th>Gap</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>

<h2>Retrieval Performance</h2>
<div class="chart"><img src="recall_at_k.png" alt="Recall@k"></div>
<div class="chart"><img src="recall_by_expression.png" alt="Recall by Expression"></div>

<h2>Accuracy Comparison</h2>
<div class="chart"><img src="accuracy_bar.png" alt="Accuracy Bar Chart"></div>
</body>
</html>"""

    path = output_dir / "report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 结果可视化")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Step 3 full_metrics.json; defaults under the configured result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures and HTML report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = resolve_path(
        args.metrics_file,
        RESULT_DIR / "summary" / "full_metrics.json",
    )
    metrics = load_metrics(metrics_path)

    if not HAS_MPL:
        print("[WARN] matplotlib not installed, skipping plots")
        print("  pip install matplotlib numpy")
    else:
        setup_font()

    output_dir = resolve_path(args.output_dir, metrics_path.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MPL:
        print("Generating plots ...")
        plot_recall_at_k(metrics, output_dir)
        plot_recall_by_expression(metrics, output_dir)
        plot_accuracy_bar(metrics, output_dir)

    print("\nGenerating HTML report ...")
    generate_html_report(metrics, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
