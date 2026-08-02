"""
Visualize evaluation results for all models.
Generates bar charts, radar plots, pie charts, and comparison figures.

Usage:
    python -m evaluation.visualize_results
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ─── Evaluation Data (from evaluation_results.md) ───────────────────────────

RESULTS = [
    {
        "model_name": "MLP",
        "checkpoint": "mlp_fp32.pt",
        "nmse_db": -5.44,
        "nmse_lin": 0.2858,
        "f1_score": 0.6992,
        "precision": 0.7174,
        "recall": 0.6819,
        "params": 148_481,
        "macs_per_sec": 9_280_062,
        "total_ram_kb": 583.76,
    },
    {
        "model_name": "T-MLP",
        "checkpoint": "t_mlp_fp32.pt",
        "nmse_db": -5.90,
        "nmse_lin": 0.2569,
        "f1_score": 0.7195,
        "precision": 0.7726,
        "recall": 0.6732,
        "params": 148_225,
        "macs_per_sec": 9_264_062,
        "total_ram_kb": 582.76,
    },
    {
        "model_name": "T-MLP (QAT INT8)",
        "checkpoint": "t_mlp_int8.pt",
        "nmse_db": -6.14,
        "nmse_lin": 0.2430,
        "f1_score": 0.7641,
        "precision": 0.7180,
        "recall": 0.8166,
        "params": 148_225,
        "macs_per_sec": 9_264_062,
        "total_ram_kb": 148.50,
    },
    {
        "model_name": "GRU",
        "checkpoint": "gru_fp32.pt",
        "nmse_db": -5.97,
        "nmse_lin": 0.2529,
        "f1_score": 0.7578,
        "precision": 0.6984,
        "recall": 0.8282,
        "params": 149_857,
        "macs_per_sec": 9_366_062,
        "total_ram_kb": 588.51,
    },
]

MODEL_NAMES = [r["model_name"] for r in RESULTS]

# ─── Color Palette ───────────────────────────────────────────────────────────

COLORS = {
    "MLP":              "#4E79A7",   # Steel Blue
    "T-MLP":            "#59A14F",   # Fern Green
    "T-MLP (QAT INT8)": "#E15759",  # Coral Red
    "GRU":              "#F28E2B",   # Tangerine Orange
}
BAR_COLORS = [COLORS[n] for n in MODEL_NAMES]

# ─── Styling ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def add_bar_values(ax, bars, fmt="{:.4f}", offset=0.01):
    """Annotate bar tops with values."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#333333",
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Quality Metrics Bar Charts
# ═══════════════════════════════════════════════════════════════════════════════

def plot_quality_metrics(save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Model Evaluation — Quality Metrics",
        fontsize=18, fontweight="bold", color="#222222", y=0.98,
    )

    # 1) NMSE (dB)  — lower is better
    ax = axes[0, 0]
    vals = [abs(r["nmse_db"]) for r in RESULTS]  # plot absolute for visual clarity
    bars = ax.bar(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, width=0.55)
    add_bar_values(ax, bars, fmt="{:.2f}", offset=0.04)
    ax.set_title("NMSE (dB) ↓  — lower is better")
    ax.set_ylabel("|NMSE| (dB)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    # Mark the best
    best_idx = np.argmax(vals)  # highest absolute = most negative = best
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)

    # 2) F1-Score  — higher is better
    ax = axes[0, 1]
    vals = [r["f1_score"] for r in RESULTS]
    bars = ax.bar(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, width=0.55)
    add_bar_values(ax, bars, fmt="{:.4f}", offset=0.005)
    ax.set_title("F1-Score ↑  — higher is better")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0.6, max(vals) * 1.08)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)

    # 3) Precision  — higher is better
    ax = axes[1, 0]
    vals = [r["precision"] for r in RESULTS]
    bars = ax.bar(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, width=0.55)
    add_bar_values(ax, bars, fmt="{:.4f}", offset=0.005)
    ax.set_title("Precision ↑  — higher is better")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.6, max(vals) * 1.08)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)

    # 4) Recall  — higher is better
    ax = axes[1, 1]
    vals = [r["recall"] for r in RESULTS]
    bars = ax.bar(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, width=0.55)
    add_bar_values(ax, bars, fmt="{:.4f}", offset=0.005)
    ax.set_title("Recall ↑  — higher is better")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.55, max(vals) * 1.08)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    best_idx = np.argmax(vals)
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "quality_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Radar / Spider Chart (Multi-axis comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar_chart(save_dir):
    categories = ["NMSE\n(normalized)", "F1-Score", "Precision", "Recall", "Memory\nEfficiency"]
    N = len(categories)

    # Normalize all metrics to [0, 1] for fair radar comparison
    nmse_vals = [abs(r["nmse_db"]) for r in RESULTS]
    nmse_norm = [(v - min(nmse_vals)) / (max(nmse_vals) - min(nmse_vals) + 1e-9) for v in nmse_vals]

    ram_vals = [r["total_ram_kb"] for r in RESULTS]
    ram_max = max(ram_vals)
    mem_eff = [(ram_max - v) / (ram_max - min(ram_vals) + 1e-9) for v in ram_vals]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Multi-Metric Radar Comparison",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for i, r in enumerate(RESULTS):
        values = [
            nmse_norm[i],
            r["f1_score"],
            r["precision"],
            r["recall"],
            mem_eff[i],
        ]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=BAR_COLORS[i])
        ax.plot(angles, values, "o-", linewidth=2, markersize=6,
                color=BAR_COLORS[i], label=r["model_name"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666666")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "radar_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — RAM Footprint Pie Chart + Quantization Savings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ram_pie_charts(save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Memory Footprint Analysis",
        fontsize=16, fontweight="bold", color="#222222", y=1.02,
    )

    # ── Pie 1: RAM share across models ──
    ax = axes[0]
    ram_vals = [r["total_ram_kb"] for r in RESULTS]
    explode = [0.03] * len(RESULTS)
    # Highlight T-MLP QAT (smallest footprint)
    min_idx = np.argmin(ram_vals)
    explode[min_idx] = 0.10

    wedges, texts, autotexts = ax.pie(
        ram_vals,
        labels=MODEL_NAMES,
        autopct=lambda pct: f"{pct:.1f}%\n({pct / 100 * sum(ram_vals):.0f} KB)",
        colors=BAR_COLORS,
        explode=explode,
        startangle=140,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)
    ax.set_title("RAM Footprint Share\nAcross Models", fontsize=13, fontweight="bold", pad=15)

    # ── Pie 2: FP32 vs INT8 Quantization Breakdown ──
    ax = axes[1]
    fp32_ram = 582.76
    int8_ram = 148.50
    saved_ram = fp32_ram - int8_ram

    sizes = [int8_ram, saved_ram]
    labels = [f"INT8 Weights\n({int8_ram:.0f} KB)", f"Savings via\nQuantization\n({saved_ram:.0f} KB)"]
    colors_q = ["#E15759", "#B0D9B1"]
    explode_q = [0.05, 0.05]

    wedges2, texts2, autotexts2 = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_q,
        explode=explode_q,
        startangle=90,
        pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for t in autotexts2:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    for t in texts2:
        t.set_fontsize(10)
    ax.set_title("T-MLP Quantization Savings\n(FP32 → INT8)", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(save_dir, "ram_pie_charts.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Efficiency vs Quality Scatter Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficiency_vs_quality(save_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "Noise Reduction Quality vs. Memory Efficiency",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    for i, r in enumerate(RESULTS):
        size = r["params"] / 500  # bubble size proportional to param count
        ax.scatter(
            r["total_ram_kb"],
            abs(r["nmse_db"]),
            s=size, c=BAR_COLORS[i], edgecolors="#333333",
            linewidths=1.5, alpha=0.85, zorder=5,
        )
        ax.annotate(
            r["model_name"],
            (r["total_ram_kb"], abs(r["nmse_db"])),
            textcoords="offset points", xytext=(12, 8),
            fontsize=11, fontweight="bold", color=BAR_COLORS[i],
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2),
        )

    ax.set_xlabel("RAM Footprint (KB) — lower is better →", fontsize=12)
    ax.set_ylabel("|NMSE| (dB) — higher is better ↑", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a quadrant annotation
    ax.annotate(
        "*  Ideal: High Quality\n     Low Memory",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=10, fontstyle="italic", color="#666666",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc"),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "efficiency_vs_quality.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Grouped Bar Chart (Precision, Recall, F1 side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grouped_classification_metrics(save_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Classification Metrics — Precision / Recall / F1",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    x = np.arange(len(MODEL_NAMES))
    width = 0.22

    prec = [r["precision"] for r in RESULTS]
    rec  = [r["recall"]    for r in RESULTS]
    f1   = [r["f1_score"]  for r in RESULTS]

    b1 = ax.bar(x - width, prec, width, label="Precision", color="#6C8EBF", edgecolor="white", linewidth=1)
    b2 = ax.bar(x,         rec,  width, label="Recall",    color="#82B366", edgecolor="white", linewidth=1)
    b3 = ax.bar(x + width, f1,   width, label="F1-Score",  color="#D6604D", edgecolor="white", linewidth=1)

    add_bar_values(ax, b1, fmt="{:.3f}", offset=0.005)
    add_bar_values(ax, b2, fmt="{:.3f}", offset=0.005)
    add_bar_values(ax, b3, fmt="{:.3f}", offset=0.005)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.6, 0.92)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "classification_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Parameter Count & MACs Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_compute_comparison(save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Computational Cost Comparison",
        fontsize=16, fontweight="bold", color="#222222", y=1.02,
    )

    # Params
    ax = axes[0]
    vals = [r["params"] for r in RESULTS]
    bars = ax.barh(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 200, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlabel("Parameter Count")
    ax.set_title("Trainable Parameters", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # MACs/sec
    ax = axes[1]
    vals = [r["macs_per_sec"] for r in RESULTS]
    bars = ax.barh(MODEL_NAMES, vals, color=BAR_COLORS, edgecolor="white", linewidth=1.2, height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 20000, bar.get_y() + bar.get_height() / 2,
                f"{v/1e6:.2f}M", va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_xlabel("MACs per Second")
    ax.set_title("Multiply-Accumulate Ops / sec", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    plt.tight_layout()
    path = os.path.join(save_dir, "compute_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    save_dir = os.path.join(root, "evaluation_figures")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  Generating Evaluation Visualizations")
    print("=" * 60)

    paths = []
    paths.append(plot_quality_metrics(save_dir))
    paths.append(plot_radar_chart(save_dir))
    paths.append(plot_ram_pie_charts(save_dir))
    paths.append(plot_efficiency_vs_quality(save_dir))
    paths.append(plot_grouped_classification_metrics(save_dir))
    paths.append(plot_compute_comparison(save_dir))

    print("\n" + "=" * 60)
    print(f"  All {len(paths)} figures saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
