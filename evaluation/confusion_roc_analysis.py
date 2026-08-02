"""
Confusion Matrix & ROC-AUC Analysis for all models.
Loads each trained checkpoint, runs inference on the test set,
then generates per-model confusion matrices and ROC-AUC curves.

Usage:
    python -m evaluation.confusion_roc_analysis
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score
)

from DataPipeline.dataset import create_dataloaders
from Models.mlp import BasicMLP
from Models.t_mlp import TMLP
from Models.t_mlp_qat import TMLP_QAT
from Models.gru_baseline import GRUBaseline

# ─── Config ──────────────────────────────────────────────────────────────────

THRESHOLD = 0.5  # Binary classification threshold (same as metrics.py)

MODEL_CONFIGS = [
    ("MLP",              BasicMLP,     "mlp_fp32.pt",    False),
    ("T-MLP",            TMLP,         "t_mlp_fp32.pt",  False),
    ("T-MLP (QAT INT8)", TMLP_QAT,     "t_mlp_int8.pt",  True),
    ("GRU",              GRUBaseline,   "gru_fp32.pt",    False),
]

COLORS = {
    "MLP":              "#4E79A7",
    "T-MLP":            "#59A14F",
    "T-MLP (QAT INT8)": "#E15759",
    "GRU":              "#F28E2B",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

# Maximum number of elements to use for ROC computation (memory-safe sampling)
MAX_ROC_SAMPLES = 2_000_000


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(model_name, model_class, checkpoint_file, is_int8, test_loader):
    """Load a model and collect predictions + targets from the test set."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    ckpt_path = os.path.join(root, 'results', checkpoint_file)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = model_class()
    if is_int8:
        model.fuse_and_prepare()
        model.convert_to_int8()

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    print(f"  {model_name}: {preds_arr.shape[0]} frames x {preds_arr.shape[1]} freq bins "
          f"= {preds_arr.size:,} total elements")

    return preds_arr, targets_arr


def compute_confusion_and_roc(preds, targets, threshold=THRESHOLD):
    """Compute confusion matrix, ROC curve, and AUC from prediction arrays."""
    # Flatten to element-wise binary classification
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    # Binary confusion matrix
    pred_bin = (preds_flat >= threshold).astype(int)
    target_bin = (targets_flat >= threshold).astype(int)
    cm = confusion_matrix(target_bin, pred_bin, labels=[0, 1])

    # Subsample for ROC curve if dataset is very large (memory safety)
    n = len(preds_flat)
    if n > MAX_ROC_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, MAX_ROC_SAMPLES, replace=False)
        preds_sample = preds_flat[idx]
        targets_sample = target_bin[idx]
    else:
        preds_sample = preds_flat
        targets_sample = target_bin

    fpr, tpr, thresholds = roc_curve(targets_sample, preds_sample)
    roc_auc = auc(fpr, tpr)

    return cm, fpr, tpr, roc_auc


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Confusion Matrices (2x2 grid, one per model)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(all_results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Confusion Matrices -- Binary Mask Classification (threshold = 0.5)",
        fontsize=17, fontweight="bold", color="#222222", y=1.01,
    )

    for ax, (name, cm, _, _, roc_auc) in zip(axes.flat, all_results):
        # Normalize to percentages
        cm_pct = cm.astype(float) / cm.sum() * 100
        cm_count = cm

        color = COLORS[name]

        # Custom colormap from white to model color
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", color], N=256)

        im = ax.imshow(cm_pct, interpolation='nearest', cmap=cmap, vmin=0, vmax=cm_pct.max() * 1.1)

        labels = ["Noise\n(Mask < 0.5)", "Speech\n(Mask >= 0.5)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=11, fontweight="bold")

        # Annotate cells with count + percentage
        for i in range(2):
            for j in range(2):
                count = cm_count[i, j]
                pct = cm_pct[i, j]
                # Cell label
                if i == j:
                    cell_name = "TN" if i == 0 else "TP"
                else:
                    cell_name = "FP" if i == 0 else "FN"

                text_color = "white" if pct > cm_pct.max() * 0.55 else "#333333"
                ax.text(j, i,
                        f"{cell_name}\n{count:,}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=text_color)

        # Title with F1 and AUC
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        ax.set_title(
            f"{name}\nF1={f1:.4f} | AUC={roc_auc:.4f}",
            fontsize=12, fontweight="bold", color=color, pad=12,
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — ROC-AUC Curves (all models overlaid)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(all_results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.suptitle(
        "ROC Curves -- All Models",
        fontsize=17, fontweight="bold", color="#222222", y=0.98,
    )

    for name, _, fpr, tpr, roc_auc in all_results:
        color = COLORS[name]
        ax.plot(fpr, tpr, linewidth=2.5, color=color,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13, fontweight="bold")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", fontsize=12, framealpha=0.95,
              fancybox=True, shadow=True)

    # Shade the area to ideal
    ax.fill_between([0, 0, 1], [0, 1, 1], alpha=0.03, color="green")
    ax.annotate(
        "Ideal Region",
        xy=(0.15, 0.92), fontsize=10, fontstyle="italic", color="#666666",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "roc_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Individual ROC curves (one per model, with filled AUC area)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_individual_roc(all_results, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "ROC-AUC Per Model (with shaded area under curve)",
        fontsize=17, fontweight="bold", color="#222222", y=1.01,
    )

    for ax, (name, _, fpr, tpr, roc_auc) in zip(axes.flat, all_results):
        color = COLORS[name]

        # Shaded AUC area
        ax.fill_between(fpr, tpr, alpha=0.25, color=color, step="post")
        ax.plot(fpr, tpr, linewidth=2.5, color=color)

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)

        ax.set_xlabel("FPR", fontsize=10)
        ax.set_ylabel("TPR", fontsize=10)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(f"{name}\nAUC = {roc_auc:.4f}", fontsize=13,
                     fontweight="bold", color=color, pad=10)

        # AUC value in center
        ax.text(0.55, 0.35,
                f"AUC\n{roc_auc:.4f}",
                transform=ax.transAxes, fontsize=18, fontweight="bold",
                color=color, alpha=0.6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.7))

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_individual.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — AUC Bar Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_auc_comparison(all_results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "ROC-AUC Score Comparison",
        fontsize=16, fontweight="bold", color="#222222", y=0.98,
    )

    names = [r[0] for r in all_results]
    aucs = [r[4] for r in all_results]
    colors = [COLORS[n] for n in names]

    bars = ax.bar(names, aucs, color=colors, edgecolor="white", linewidth=1.5, width=0.5)

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#333333")

    # Highlight best
    best_idx = np.argmax(aucs)
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)

    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_ylim(min(aucs) * 0.95, max(aucs) * 1.04)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Reference line for perfect AUC
    ax.axhline(y=1.0, color="#999999", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(len(names) - 0.5, 1.001, "Perfect AUC = 1.0",
            fontsize=9, color="#999999", ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, "auc_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Combined Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined_dashboard(all_results, save_dir):
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "Confusion Matrix & ROC-AUC Dashboard",
        fontsize=20, fontweight="bold", color="#222222", y=0.99,
    )

    # Top row: 4 confusion matrices; Bottom left: ROC overlay; Bottom right: AUC bars
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.35,
                           left=0.05, right=0.97, top=0.92, bottom=0.06)

    # ── Top Row: Confusion Matrices ──
    for idx, (name, cm, fpr, tpr, roc_auc) in enumerate(all_results):
        ax = fig.add_subplot(gs[0, idx])
        cm_pct = cm.astype(float) / cm.sum() * 100
        color = COLORS[name]

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("c", ["#FFFFFF", color], N=256)
        ax.imshow(cm_pct, interpolation='nearest', cmap=cmap, vmin=0, vmax=cm_pct.max() * 1.1)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise", "Speech"], fontsize=9)
        ax.set_yticklabels(["Noise", "Speech"], fontsize=9)
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")

        for i in range(2):
            for j in range(2):
                pct = cm_pct[i, j]
                cell_name = ["TN", "FP", "FN", "TP"][i * 2 + j]
                text_color = "white" if pct > cm_pct.max() * 0.55 else "#333333"
                ax.text(j, i, f"{cell_name}\n{cm[i,j]:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color=text_color)

        tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
        f1 = 2*tp / (2*tp + fp + fn + 1e-10)
        ax.set_title(f"{name}\nF1={f1:.3f} | AUC={roc_auc:.4f}",
                     fontsize=11, fontweight="bold", color=color, pad=8)

    # ── Bottom Left: ROC Overlay ──
    ax = fig.add_subplot(gs[1, :2])
    for name, _, fpr, tpr, roc_auc in all_results:
        ax.plot(fpr, tpr, linewidth=2.5, color=COLORS[name],
                label=f"{name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_title("ROC Curves Overlay", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Bottom Right: AUC Bars ──
    ax = fig.add_subplot(gs[1, 2:])
    names = [r[0] for r in all_results]
    aucs = [r[4] for r in all_results]
    bar_colors = [COLORS[n] for n in names]
    bars = ax.barh(names, aucs, color=bar_colors, edgecolor="white", linewidth=1.5, height=0.5)
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
    best_idx = np.argmax(aucs)
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2.5)
    ax.set_xlabel("AUC Score", fontsize=11, fontweight="bold")
    ax.set_xlim(min(aucs) * 0.95, max(aucs) * 1.04)
    ax.axvline(x=1.0, color="#999999", linewidth=1, linestyle=":", alpha=0.6)
    ax.set_title("AUC Score Ranking", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    path = os.path.join(save_dir, "confusion_roc_dashboard.png")
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
    print("  Confusion Matrix & ROC-AUC Analysis")
    print("=" * 60)

    # Load test data once (shared across all models)
    print("\n[1/3] Loading test dataset...")
    _, _, test_loader = create_dataloaders(
        os.path.join(root, 'data', 'speech'),
        os.path.join(root, 'data', 'noise'),
        batch_size=64
    )

    # Run inference for each model
    print("\n[2/3] Running inference on all models...")
    all_results = []  # List of (name, cm, fpr, tpr, auc)

    for name, model_cls, ckpt, is_int8 in MODEL_CONFIGS:
        print(f"\n  --- {name} ({ckpt}) ---")
        preds, targets = run_inference(name, model_cls, ckpt, is_int8, test_loader)
        cm, fpr, tpr, roc_auc = compute_confusion_and_roc(preds, targets)

        tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        print(f"    TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}")
        print(f"    Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")
        print(f"    ROC-AUC={roc_auc:.4f}")

        all_results.append((name, cm, fpr, tpr, roc_auc))

    # Generate visualizations
    print("\n[3/3] Generating visualizations...")
    paths = []
    paths.append(plot_confusion_matrices(all_results, save_dir))
    paths.append(plot_roc_curves(all_results, save_dir))
    paths.append(plot_individual_roc(all_results, save_dir))
    paths.append(plot_auc_comparison(all_results, save_dir))
    paths.append(plot_combined_dashboard(all_results, save_dir))

    print("\n" + "=" * 60)
    print(f"  All {len(paths)} figures saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
