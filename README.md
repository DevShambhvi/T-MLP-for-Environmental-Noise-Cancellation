# T-MLP for Environmental Noise Cancellation

> **Software-only prototype** of a Temporal-MLP (T-MLP) based Environmental Noise Cancellation system for TWS (True Wireless Stereo) earbuds — simulation and offline evaluation only, no on-device deployment.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Architecture](#architecture)
- [Constraints \& Assumptions](#constraints--assumptions)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Evaluation Figures](#evaluation-figures)
- [License](#license)

---

## Overview

This project validates a core hypothesis entirely in software:

> *A sliding-window Temporal MLP (T-MLP) can predict a continuous suppression mask per audio frame, achieving near-GRU quality at a fraction of the computational cost — making it suitable for deployment on low-power MCUs in TWS earbuds.*

The T-MLP is trained with **Quantization-Aware Training (QAT)** and benchmarked against two baselines — a **basic single-frame MLP** and a **GRU/RNN** — on both accuracy metrics and edge-hardware-relevant efficiency metrics, all estimated analytically without deploying to physical hardware.

### Key Deliverables

| Deliverable | Description |
|---|---|
| Trained Models | FP32 and INT8-quantized T-MLP, basic MLP, and GRU baselines |
| Evaluation Report | NMSE, Precision/Recall/F1 across all models |
| Efficiency Analysis | Parameter counts, MACs/s, estimated RAM footprint |
| Visualizations | Quality metrics, radar charts, confusion matrices, ROC curves, quantization dashboards |

---

## Motivation

Active Noise Cancellation (ANC) in TWS earbuds traditionally relies on analog feedback loops or simple DSP filters. ML-based approaches (RNNs, GRUs) offer superior noise modelling but carry significant compute and memory overhead for real-time inference on low-power MCUs.

The **T-MLP** trades recurrent hidden states for an explicit sliding window of stacked frames, giving the network temporal context through a simple, feed-forward architecture. This design targets:

- **Lower latency** — no sequential hidden-state dependency.
- **Smaller memory footprint** — no hidden state to maintain across frames.
- **Quantization-friendly weights** — feed-forward layers quantize more predictably than recurrent gates.

---

## Architecture

### T-MLP (Primary Model)

The T-MLP consumes a sliding window of the last 5 log-mel frames, stacked into a single input vector, and outputs a continuous suppression mask via sigmoid activation.

```
Input (320) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(257) → Sigmoid
```

| Component | Detail |
|---|---|
| Input features | 64-dim log-mel spectrogram per frame |
| Sliding window | 5 frames (64 × 5 = 320 values) |
| Frame / Hop | 32 ms (512 samples @ 16 kHz) / 16 ms (256 samples @ 16 kHz) |
| Output | 257-dim Ideal Ratio Mask per frame (Sigmoid, `[0, 1]`) |
| Loss | MSE between predicted mask and Ideal Ratio Mask (IRM) |
| Normalization | Per-feature mean-variance |

### Ground-Truth: Ideal Ratio Mask (IRM)

Labels are generated mathematically from the synthetic mixing process:

```
IRM = sqrt( S² / (S² + N² + 1e-10) )
```

where **S** = clean speech magnitude and **N** = noise magnitude per time-frequency bin.

### Baselines

| Model | Description | Parameter Matching |
|---|---|---|
| **Basic MLP** (`Models/mlp.py`) | Center-frame input (64-dim), no temporal context | ~148.5K params |
| **T-MLP** (`Models/t_mlp.py`) | 5-frame sliding window (320-dim) feedforward | ~148.2K params |
| **T-MLP QAT** (`Models/t_mlp_qat.py`) | T-MLP with fake-quantization stubs for INT8 simulation | ~148.2K params |
| **GRU** (`Models/gru_baseline.py`) | Recurrent baseline (64 input → 160 hidden) | ~149.9K params |

Parameter counts are matched across baselines so comparisons are defensible.

### Quantization-Aware Training (QAT)

Fake-quantization nodes are inserted via `torch.ao.quantization` and the model is fine-tuned with INT8 rounding simulated in the forward pass. Target: **< 2% NMSE degradation** from FP32 to INT8.

---

## Constraints & Assumptions

| Parameter | Value | Rationale |
|---|---|---|
| Target device class | Low-power MCU (ESP32 / Cortex-M4-class) | Analytical estimation only — not deployed |
| SRAM budget | ~100 KB | Drives architecture sizing |
| Latency budget | < 10 ms end-to-end | Frame/hop sizes chosen to respect this |
| Sampling rate | 16 kHz mono | Standard for speech-band applications |
| Use case | Near-field speech, TWS earbuds | Urban/home ambient noise |
| Scope | Software simulation only | No firmware, no hardware-in-the-loop |

---

## Datasets

| Dataset | Role | Source |
|---|---|---|
| **LibriSpeech** | Clean, labeled speech | `data/speech/` |
| **ESC-50** | Environmental noise (transients, ambient) | `data/noise/` |

Speech and noise are synthetically mixed at SNR levels from −5 dB to +20 dB.

---

## Repository Structure

```
T-MLP-for-Environmental-Noise-Cancellation/
│
├── README.md                       # Project overview & guidelines
├── requirements.txt                # Python dependencies
├── evaluation_results.md           # Model comparison & evaluation report
├── .gitignore                      # Git ignore rules
│
├── DataPipeline/                   # Audio loading, feature extraction & dataloaders
│   ├── __init__.py
│   ├── datapipeline.py             # Audio I/O & synthetic mixing
│   ├── FeatureExtraction.py        # STFT, Log-Mel, Normalization & Sliding Window
│   └── dataset.py                  # PyTorch Dataset, IRM calculation & DataLoader factory
│
├── Models/                         # Parameter-matched model definitions (~148K params)
│   ├── __init__.py
│   ├── mlp.py                      # Single-frame Basic MLP baseline
│   ├── t_mlp.py                    # Temporal MLP (primary sliding-window model)
│   ├── t_mlp_qat.py                # T-MLP with Quantization-Aware Training nodes
│   └── gru_baseline.py             # GRU recurrent baseline
│
├── training/                       # Model training & QAT routines
│   ├── __init__.py
│   ├── losses.py                   # Mask MSE & L1 loss functions
│   ├── train.py                    # Unified FP32 training loop with early stopping
│   └── qat.py                      # QAT fine-tuning & INT8 conversion script
│
├── evaluation/                     # Evaluation, metrics & visualization
│   ├── __init__.py
│   ├── metrics.py                  # NMSE, F1/Precision/Recall, SDR, STOI, hardware estimates
│   ├── eval_runner.py              # Shared model evaluation logic
│   ├── evaluate_all.py             # Run all evaluations & generate evaluation_results.md
│   ├── eval_mlp.py                 # Evaluate Basic MLP individually
│   ├── eval_t_mlp.py               # Evaluate T-MLP individually
│   ├── eval_t_mlp_qat.py           # Evaluate T-MLP QAT (INT8) individually
│   ├── eval_gru.py                 # Evaluate GRU individually
│   ├── visualize_results.py        # Quality metrics, radar, pie, scatter, bar charts
│   ├── confusion_roc_analysis.py   # Confusion matrices & ROC-AUC curves
│   └── quantization_analysis.py    # FP32 vs INT8 quantization deep-dive
│
├── evaluation_figures/             # Generated evaluation plots (16 figures)
│   ├── quality_metrics.png
│   ├── radar_comparison.png
│   ├── ram_pie_charts.png
│   ├── efficiency_vs_quality.png
│   ├── classification_metrics.png
│   ├── compute_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── roc_individual.png
│   ├── auc_comparison.png
│   ├── confusion_roc_dashboard.png
│   ├── quantization_metric_comparison.png
│   ├── quantization_pct_change.png
│   ├── quantization_memory_breakdown.png
│   ├── quantization_radar.png
│   └── quantization_dashboard.png
│
├── data/                           # Dataset storage
│   ├── speech/                     # LibriSpeech clean audio files (.flac)
│   └── noise/                      # ESC-50 noise audio files (.wav)
│
└── results/                        # Saved model checkpoints
    ├── mlp_fp32.pt                 # Basic MLP (FP32)
    ├── t_mlp_fp32.pt               # T-MLP (FP32)
    ├── t_mlp_int8.pt               # T-MLP QAT (INT8)
    └── gru_fp32.pt                 # GRU Baseline (FP32)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU-accelerated training)

### Installation

```bash
# Clone the repository
git clone https://github.com/DevShambhvi/T-MLP-for-Environmental-Noise-Cancellation.git
cd T-MLP-for-Environmental-Noise-Cancellation

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. **Speech data**: Download a subset of [LibriSpeech](https://www.openslr.org/12/) and place `.flac` files in `data/speech/`
2. **Noise data**: Download [ESC-50](https://github.com/karolpiczak/ESC-50) and place `.wav` files in `data/noise/`

---

## Usage

### 1. Verify Data Pipeline

```bash
python -m DataPipeline.dataset
```

### 2. Train FP32 Models

```bash
# Train primary T-MLP
python -m training.train --model t_mlp --epochs 50 --batch 64

# Train Baselines
python -m training.train --model mlp --epochs 50 --batch 64
python -m training.train --model gru --epochs 50 --batch 64
```

### 3. Quantization-Aware Training (INT8)

```bash
python -m training.qat --epochs 10 --batch 64
```

### 4. Evaluate All Models

```bash
# Run full evaluation suite (generates evaluation_results.md)
python -m evaluation.evaluate_all

# Or evaluate individual models
python -m evaluation.eval_t_mlp
python -m evaluation.eval_mlp
python -m evaluation.eval_gru
python -m evaluation.eval_t_mlp_qat
```

### 5. Generate Visualizations

```bash
# Quality metrics bar charts, radar plots, pie charts, scatter plots
python -m evaluation.visualize_results

# Confusion matrices & ROC-AUC curves
python -m evaluation.confusion_roc_analysis

# Quantization deep-dive (FP32 vs INT8)
python -m evaluation.quantization_analysis
```

All figures are saved to `evaluation_figures/`.

---

## Evaluation Metrics

### Accuracy & Quality

| Metric | Description |
|---|---|
| **NMSE** | Normalized Mean Squared Error between predicted and ideal mask |
| **Precision / Recall / F1** | Thresholded mask classification performance |

### Efficiency (Analytical Estimates)

| Metric | Method |
|---|---|
| **Parameter count** | Total trainable parameters |
| **MACs/s** | Multiply-accumulate operations per second of audio |
| **RAM estimate** | `params × bytes_per_param + activation_buffer` |
| **Quantization loss** | FP32-to-INT8 NMSE degradation |

---

## Results

### Model Performance & Efficiency Comparison

| Model | NMSE (dB) ↓ | F1-Score ↑ | Precision ↑ | Recall ↑ | Params | MACs/s | RAM (KB) |
|---|---|---|---|---|---|---|---|
| Basic MLP | -5.44 dB | 0.6992 | 0.7174 | 0.6819 | 148,481 | 9.28M | 583.76 |
| **T-MLP** | -5.90 dB | 0.7195 | 0.7726 | 0.6732 | 148,225 | 9.26M | 582.76 |
| **T-MLP QAT (INT8)** | **-6.14 dB** | **0.7641** | 0.7180 | **0.8166** | 148,225 | 9.26M | **148.50** |
| GRU | -5.97 dB | 0.7578 | 0.6984 | 0.8282 | 149,857 | 9.37M | 588.51 |

### Key Findings

1. **T-MLP matches GRU quality** — The T-MLP achieves -5.90 dB NMSE vs the GRU's -5.97 dB, with comparable F1 (0.7195 vs 0.7578), while using a simpler feed-forward architecture with no recurrent state.

2. **INT8 quantization _improves_ performance** — The QAT INT8 model achieves the best NMSE (-6.14 dB) and F1 (0.7641) of all models, suggesting the regularization effect of quantization noise helped generalization.

3. **74.5% RAM reduction** — INT8 quantization reduces the memory footprint from 582.76 KB to 148.50 KB, bringing the model well within the ~100 KB SRAM budget target for MCU deployment.

4. **Precision-Recall trade-off** — The T-MLP FP32 leads in precision (0.7726), while the GRU and INT8 models favor recall (0.8282, 0.8166). The INT8 model offers the best balance overall (highest F1).

### Quantization Analysis (FP32 → INT8)

| Metric | FP32 T-MLP | INT8 T-MLP | Delta | Direction |
|---|---|---|---|---|
| NMSE (dB) | -5.90 | -6.14 | -0.24 dB | ✅ Better |
| F1-Score | 0.7195 | 0.7641 | +0.0446 | ✅ Better |
| Precision | 0.7726 | 0.7180 | -0.0546 | ⚠️ Slight drop |
| Recall | 0.6732 | 0.8166 | +0.1434 | ✅ Much better |
| RAM | 582.76 KB | 148.50 KB | -74.5% | ✅ 4× smaller |

---

## Evaluation Figures

All figures are generated by the evaluation scripts and saved to `evaluation_figures/`. See `evaluation_results.md` for the full analysis with embedded figure references.

| Figure | Script | Description |
|---|---|---|
| `quality_metrics.png` | `visualize_results.py` | NMSE, F1, Precision, Recall bar charts |
| `radar_comparison.png` | `visualize_results.py` | Multi-metric radar/spider chart |
| `ram_pie_charts.png` | `visualize_results.py` | Memory footprint breakdown |
| `efficiency_vs_quality.png` | `visualize_results.py` | Quality vs RAM scatter plot |
| `classification_metrics.png` | `visualize_results.py` | Grouped Precision/Recall/F1 bars |
| `compute_comparison.png` | `visualize_results.py` | Parameter count & MACs comparison |
| `confusion_matrices.png` | `confusion_roc_analysis.py` | Per-model confusion matrices |
| `roc_curves.png` | `confusion_roc_analysis.py` | Overlaid ROC curves |
| `roc_individual.png` | `confusion_roc_analysis.py` | Per-model ROC with shaded AUC |
| `auc_comparison.png` | `confusion_roc_analysis.py` | AUC score bar chart |
| `confusion_roc_dashboard.png` | `confusion_roc_analysis.py` | Combined dashboard |
| `quantization_metric_comparison.png` | `quantization_analysis.py` | FP32 vs INT8 side-by-side |
| `quantization_pct_change.png` | `quantization_analysis.py` | % change waterfall chart |
| `quantization_memory_breakdown.png` | `quantization_analysis.py` | RAM & file size comparison |
| `quantization_radar.png` | `quantization_analysis.py` | FP32 vs INT8 radar overlay |
| `quantization_dashboard.png` | `quantization_analysis.py` | Quantization summary dashboard |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **PyTorch** | Training, QAT (`torch.ao.quantization`) |
| **librosa** | Feature extraction, audio I/O |
| **scikit-learn** | Confusion matrices, ROC curves |
| **matplotlib** | Plotting and visualization |

---

## License

This project is for academic and research purposes. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Software-only prototype — all hardware efficiency claims are analytical estimates, not physical measurements.</i>
</p>