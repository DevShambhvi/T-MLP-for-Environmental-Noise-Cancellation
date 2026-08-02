"""
Quantization Loss Analysis: T-MLP FP32 vs T-MLP QAT INT8
Detailed visual comparison of what changed due to quantization.

Usage:
    python -m evaluation.quantization_analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ─── Data ────────────────────────────────────────────────────────────────────

FP32 = {
    "label": "T-MLP (FP32)",
    "nmse_db": -5.90,
    "nmse_lin": 0.2569,
    "f1_score": 0.7195,
    "precision": 0.7726,
    "recall": 0.6732,
    "params": 148_225,
    "macs_per_sec": 9_264_062,
    "total_ram_kb": 582.76,
    "checkpoint_kb": 596_025 / 1024,  # ~582 KB file
}

INT8 = {
    "label": "T-MLP QAT (INT8)",
    "nmse_db": -6.14,
    "nmse_lin": 0.2430,
    "f1_score": 0.7641,
    "precision": 0.7180,
    "recall": 0.8166,
    "params": 148_225,
    "macs_per_sec": 9_264_062,
    "total_ram_kb": 148.50,
    "checkpoint_kb": 167_541 / 1024,  # ~163.6 KB file
}

# ─── Styling ─────────────────────────────────────────────────────────────────

COLOR_FP32 = "#4E79A7"   # Steel Blue
COLOR_INT8 = "#E15759"   # Coral Red
COLOR_GAIN = "#2D9F2D"   # Green for improvement
COLOR_LOSS = "#D62728"   # Red for degradation
COLOR_BG   = "#FAFAFA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


def delta_color(delta, higher_is_better=True):
    """Return green if improvement, red if degradation."""
    if higher_is_better:
        return COLOR_GAIN if delta >= 0 else COLOR_LOSS
    else:
        return COLOR_GAIN if delta <= 0 else COLOR_LOSS


def delta_arrow(delta, higher_is_better=True):
    """Return arrow symbol for direction."""
    if higher_is_better:
        return "^" if delta >= 0 else "v"
    else:
        return "v" if delta <= 0 else "^"


def delta_label(delta, fmt="+.4f", higher_is_better=True):
    """Format delta with direction indicator."""
    arrow = " (BETTER)" if (delta >= 0) == higher_is_better else " (WORSE)"
    return f"{delta:{fmt}}{arrow}"


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Side-by-Side Metric Comparison with Delta Annotations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(save_dir):
    metrics = [
        ("NMSE (dB)",      "nmse_db",    "{:.2f}",  False, "dB"),
        ("Linear NMSE",    "nmse_lin",   "{:.4f}",  False, ""),
        ("F1-Score",       "f1_score",   "{:.4f}",  True,  ""),
        ("Precision",      "precision",  "{:.4f}",  True,  ""),
        ("Recall",         "recall",     "{:.4f}",  True,  ""),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle(
        "Quantization Impact: T-MLP FP32 vs INT8 -- Metric-by-Metric",
        fontsize=17, fontweight="bold", color="#222222", y=1.02,
    )

    for ax, (name, key, fmt, higher_better, unit) in zip(axes, metrics):
        fp_val = FP32[key]
        int_val = INT8[key]
        delta = int_val - fp_val

        # For NMSE (negative values), use absolute for bar heights
        if key in ("nmse_db",):
            bar_vals = [abs(fp_val), abs(int_val)]
        else:
            bar_vals = [fp_val, int_val]

        bars = ax.bar(
            ["FP32", "INT8"], bar_vals,
            color=[COLOR_FP32, COLOR_INT8],
            edgecolor="white", linewidth=1.5, width=0.5,
        )

        # Value labels on bars
        for bar, val in zip(bars, [fp_val, int_val]):
            display = fmt.format(val)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bar_vals) * 0.02,
                f"{display} {unit}".strip(),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333",
            )

        # Delta annotation
        dc = delta_color(delta, higher_better)
        dl = delta_label(delta, fmt="+.4f" if "4f" in fmt else "+.2f",
                         higher_is_better=higher_better)
        ax.text(
            0.5, 0.02, f"Delta: {dl}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=dc,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=dc, alpha=0.9),
        )

        ax.set_title(name, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, max(bar_vals) * 1.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(save_dir, "quantization_metric_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Percentage Change Waterfall Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_percentage_change(save_dir):
    metrics_info = [
        ("NMSE (dB)",   "nmse_db",   False),
        ("Linear NMSE", "nmse_lin",  False),
        ("F1-Score",    "f1_score",  True),
        ("Precision",   "precision", True),
        ("Recall",      "recall",    True),
        ("RAM (KB)",    "total_ram_kb", False),
    ]

    names = []
    pct_changes = []
    colors = []
    higher_betters = []

    for name, key, higher_better in metrics_info:
        fp_val = FP32[key]
        int_val = INT8[key]

        if key == "nmse_db":
            # For negative dB values, more negative = better
            # pct change on absolute values
            pct = ((abs(int_val) - abs(fp_val)) / abs(fp_val)) * 100
            is_improvement = pct > 0  # higher absolute NMSE means better
        elif key in ("nmse_lin", "total_ram_kb"):
            pct = ((int_val - fp_val) / abs(fp_val)) * 100
            is_improvement = pct < 0  # lower is better
        else:
            pct = ((int_val - fp_val) / fp_val) * 100
            is_improvement = pct > 0  # higher is better

        names.append(name)
        pct_changes.append(pct)
        colors.append(COLOR_GAIN if is_improvement else COLOR_LOSS)
        higher_betters.append(higher_better)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Quantization Effect: Percentage Change (FP32 --> INT8)",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    x = np.arange(len(names))
    bars = ax.bar(x, pct_changes, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55, zorder=3)

    # Annotate each bar
    for bar, pct, color in zip(bars, pct_changes, colors):
        h = bar.get_height()
        label_text = f"{pct:+.1f}%"
        y_offset = 0.8 if h >= 0 else -1.5
        va = "bottom" if h >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_offset,
            label_text,
            ha="center", va=va,
            fontsize=12, fontweight="bold", color=color,
        )

    # Zero line
    ax.axhline(y=0, color="#333333", linewidth=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylabel("% Change from FP32 Baseline", fontsize=12)
    ax.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_GAIN, label="Improvement"),
        Patch(facecolor=COLOR_LOSS, label="Degradation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "quantization_pct_change.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Memory & File Size Breakdown (Pie + Bar)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_memory_breakdown(save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Memory & Storage: FP32 vs INT8 Quantization",
        fontsize=16, fontweight="bold", color="#222222", y=1.03,
    )

    # ── Panel 1: RAM Comparison Bar ──
    ax = axes[0]
    ram_vals = [FP32["total_ram_kb"], INT8["total_ram_kb"]]
    bars = ax.bar(
        ["FP32\n(32-bit weights)", "INT8\n(8-bit weights)"],
        ram_vals, color=[COLOR_FP32, COLOR_INT8],
        edgecolor="white", linewidth=1.5, width=0.5,
    )
    for bar, val in zip(bars, ram_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:.1f} KB", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#333333",
        )
    # Reduction arrow
    reduction_pct = ((FP32["total_ram_kb"] - INT8["total_ram_kb"]) / FP32["total_ram_kb"]) * 100
    ax.annotate(
        f"{reduction_pct:.1f}%\nreduction",
        xy=(1, INT8["total_ram_kb"]),
        xytext=(0.5, FP32["total_ram_kb"] * 0.75),
        fontsize=12, fontweight="bold", color=COLOR_GAIN,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=COLOR_GAIN, lw=2),
    )
    ax.set_title("RAM Footprint", fontsize=13, fontweight="bold")
    ax.set_ylabel("Memory (KB)")
    ax.set_ylim(0, max(ram_vals) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Panel 2: Checkpoint File Size ──
    ax = axes[1]
    ckpt_vals = [FP32["checkpoint_kb"], INT8["checkpoint_kb"]]
    bars = ax.bar(
        ["FP32\nCheckpoint", "INT8\nCheckpoint"],
        ckpt_vals, color=[COLOR_FP32, COLOR_INT8],
        edgecolor="white", linewidth=1.5, width=0.5,
    )
    for bar, val in zip(bars, ckpt_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:.1f} KB", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#333333",
        )
    ckpt_reduction = ((FP32["checkpoint_kb"] - INT8["checkpoint_kb"]) / FP32["checkpoint_kb"]) * 100
    ax.annotate(
        f"{ckpt_reduction:.1f}%\nsmaller",
        xy=(1, INT8["checkpoint_kb"]),
        xytext=(0.5, FP32["checkpoint_kb"] * 0.75),
        fontsize=12, fontweight="bold", color=COLOR_GAIN,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=COLOR_GAIN, lw=2),
    )
    ax.set_title("Checkpoint File Size", fontsize=13, fontweight="bold")
    ax.set_ylabel("Size (KB)")
    ax.set_ylim(0, max(ckpt_vals) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Panel 3: Pie — What the INT8 model saves ──
    ax = axes[2]
    saved = FP32["total_ram_kb"] - INT8["total_ram_kb"]
    sizes = [INT8["total_ram_kb"], saved]
    labels = [
        f"INT8 Model\n{INT8['total_ram_kb']:.0f} KB",
        f"Saved Memory\n{saved:.0f} KB",
    ]
    colors_pie = [COLOR_INT8, "#B0D9B1"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors_pie, explode=[0.05, 0.05],
        startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=2.5),
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)
    ax.set_title("RAM Savings Breakdown", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    path = os.path.join(save_dir, "quantization_memory_breakdown.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Radar Overlay (FP32 vs INT8 only)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_quantization_radar(save_dir):
    categories = ["|NMSE| (dB)", "F1-Score", "Precision", "Recall", "Mem. Efficiency"]
    N = len(categories)

    # Normalize to comparable scales
    nmse_max = max(abs(FP32["nmse_db"]), abs(INT8["nmse_db"]))
    ram_max = max(FP32["total_ram_kb"], INT8["total_ram_kb"])

    def make_values(d):
        return [
            abs(d["nmse_db"]) / nmse_max,
            d["f1_score"],
            d["precision"],
            d["recall"],
            1 - (d["total_ram_kb"] / ram_max),  # inverted: lower RAM = higher score
        ]

    fp32_vals = make_values(FP32)
    int8_vals = make_values(INT8)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle(
        "FP32 vs INT8: Radar Overlay",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fp32_vals += fp32_vals[:1]
    int8_vals += int8_vals[:1]

    ax.fill(angles, fp32_vals, alpha=0.20, color=COLOR_FP32)
    ax.plot(angles, fp32_vals, "o-", linewidth=2.5, markersize=8,
            color=COLOR_FP32, label="FP32", zorder=5)

    ax.fill(angles, int8_vals, alpha=0.20, color=COLOR_INT8)
    ax.plot(angles, int8_vals, "s-", linewidth=2.5, markersize=8,
            color=COLOR_INT8, label="INT8 (QAT)", zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666666")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=12, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "quantization_radar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Summary Dashboard (single comprehensive figure)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(save_dir):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Quantization Loss Analysis Dashboard: T-MLP FP32 vs INT8",
        fontsize=18, fontweight="bold", color="#222222", y=0.99,
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # ── Panel 1: NMSE Comparison ──
    ax = fig.add_subplot(gs[0, 0])
    vals = [abs(FP32["nmse_db"]), abs(INT8["nmse_db"])]
    bars = ax.bar(["FP32", "INT8"], vals, color=[COLOR_FP32, COLOR_INT8],
                  edgecolor="white", linewidth=1.5, width=0.45)
    for bar, orig in zip(bars, [FP32["nmse_db"], INT8["nmse_db"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{orig:.2f} dB", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    delta_nmse = INT8["nmse_db"] - FP32["nmse_db"]
    ax.set_title(f"NMSE (dB)\nDelta: {delta_nmse:+.2f} dB", fontsize=11, fontweight="bold",
                 color=COLOR_GAIN if delta_nmse < 0 else COLOR_LOSS)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("|NMSE| (dB)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Panel 2: F1 Score ──
    ax = fig.add_subplot(gs[0, 1])
    vals = [FP32["f1_score"], INT8["f1_score"]]
    bars = ax.bar(["FP32", "INT8"], vals, color=[COLOR_FP32, COLOR_INT8],
                  edgecolor="white", linewidth=1.5, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    delta_f1 = INT8["f1_score"] - FP32["f1_score"]
    ax.set_title(f"F1-Score\nDelta: {delta_f1:+.4f}", fontsize=11, fontweight="bold",
                 color=COLOR_GAIN if delta_f1 > 0 else COLOR_LOSS)
    ax.set_ylim(0.65, max(vals) * 1.06)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Panel 3: Precision vs Recall Trade-off ──
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    w = 0.3
    prec_vals = [FP32["precision"], INT8["precision"]]
    rec_vals = [FP32["recall"], INT8["recall"]]
    b1 = ax.bar(x - w/2, prec_vals, w, label="Precision", color=["#7BAFD4", "#F09294"], edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w/2, rec_vals, w, label="Recall", color=[COLOR_FP32, COLOR_INT8], edgecolor="white", linewidth=1.2)
    for bar, v in zip(list(b1) + list(b2), prec_vals + rec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["FP32", "INT8"])
    ax.set_title("Precision / Recall Trade-off", fontsize=11, fontweight="bold")
    ax.set_ylim(0.6, 0.90)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # ── Panel 4: RAM Pie ──
    ax = fig.add_subplot(gs[1, 0])
    saved = FP32["total_ram_kb"] - INT8["total_ram_kb"]
    sizes = [INT8["total_ram_kb"], saved]
    labels = [f"INT8\n{INT8['total_ram_kb']:.0f} KB", f"Saved\n{saved:.0f} KB"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=[COLOR_INT8, "#B0D9B1"], explode=[0.04, 0.04],
        startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("RAM Savings (74.5%)", fontsize=11, fontweight="bold", color=COLOR_GAIN)

    # ── Panel 5: Percentage Change Bars ──
    ax = fig.add_subplot(gs[1, 1:])
    metrics_pct = [
        ("|NMSE| (dB)",  ((abs(INT8["nmse_db"]) - abs(FP32["nmse_db"])) / abs(FP32["nmse_db"])) * 100, True),
        ("Linear NMSE",  ((INT8["nmse_lin"] - FP32["nmse_lin"]) / FP32["nmse_lin"]) * 100, False),
        ("F1-Score",     ((INT8["f1_score"] - FP32["f1_score"]) / FP32["f1_score"]) * 100, True),
        ("Precision",    ((INT8["precision"] - FP32["precision"]) / FP32["precision"]) * 100, True),
        ("Recall",       ((INT8["recall"] - FP32["recall"]) / FP32["recall"]) * 100, True),
        ("RAM",          ((INT8["total_ram_kb"] - FP32["total_ram_kb"]) / FP32["total_ram_kb"]) * 100, False),
    ]

    names = [m[0] for m in metrics_pct]
    pcts = [m[1] for m in metrics_pct]
    # Determine improvement: for |NMSE|, F1, Precision, Recall: positive % = better
    # For Linear NMSE, RAM: negative % = better
    bar_colors = []
    for name, pct, higher_better in metrics_pct:
        if higher_better:
            bar_colors.append(COLOR_GAIN if pct > 0 else COLOR_LOSS)
        else:
            bar_colors.append(COLOR_GAIN if pct < 0 else COLOR_LOSS)

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, pcts, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
    for bar, pct, col in zip(bars, pcts, bar_colors):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.5 if h >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f"{pct:+.1f}%", ha="center", va=va,
                fontsize=10, fontweight="bold", color=col)

    ax.axhline(y=0, color="#333333", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylabel("% Change", fontsize=11)
    ax.set_title("Metric-wise % Change (FP32 --> INT8)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_GAIN, label="Improved"),
        Patch(facecolor=COLOR_LOSS, label="Degraded"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9)

    path = os.path.join(save_dir, "quantization_dashboard.png")
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
    print("  Quantization Loss Analysis: T-MLP FP32 vs INT8")
    print("=" * 60)
    print()
    print(f"  FP32 NMSE:  {FP32['nmse_db']:.2f} dB | INT8 NMSE:  {INT8['nmse_db']:.2f} dB")
    print(f"  Delta NMSE: {INT8['nmse_db'] - FP32['nmse_db']:+.2f} dB")
    print(f"  FP32 F1:    {FP32['f1_score']:.4f}   | INT8 F1:    {INT8['f1_score']:.4f}")
    print(f"  FP32 Prec:  {FP32['precision']:.4f}   | INT8 Prec:  {INT8['precision']:.4f}")
    print(f"  FP32 Rec:   {FP32['recall']:.4f}   | INT8 Rec:   {INT8['recall']:.4f}")
    print(f"  FP32 RAM:   {FP32['total_ram_kb']:.2f} KB | INT8 RAM:   {INT8['total_ram_kb']:.2f} KB")
    print(f"  RAM Saved:  {((FP32['total_ram_kb'] - INT8['total_ram_kb'])/FP32['total_ram_kb'])*100:.1f}%")
    print()

    paths = []
    paths.append(plot_metric_comparison(save_dir))
    paths.append(plot_percentage_change(save_dir))
    paths.append(plot_memory_breakdown(save_dir))
    paths.append(plot_quantization_radar(save_dir))
    paths.append(plot_summary_dashboard(save_dir))

    print()
    print("=" * 60)
    print(f"  All {len(paths)} quantization analysis figures saved to:")
    print(f"  {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
