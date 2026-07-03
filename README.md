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
| Evaluation Report | NMSE, Precision/Recall/F1, SDR, STOI across all models |
| Efficiency Analysis | Parameter counts, MACs/s, estimated RAM footprint, ONNX latency profiling |
| Quality-vs-Efficiency Plot | Single chart comparing NMSE (or SDR) against MACs for all models |
| Audio Samples | 5–10 before/after clips for informal subjective listening checks |

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

The T-MLP consumes a sliding window of the last 3–5 log-mel frames, stacked into a single input vector, and outputs a continuous suppression mask via sigmoid activation.

```
Input (640) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Dense(1) → Sigmoid
```

| Component | Detail |
|---|---|
| Input features | 128-dim log-mel spectrogram per frame |
| Sliding window | 3–5 frames (128 × 5 = 640 values) |
| Frame / Hop | 25 ms / 10 ms |
| Output | Mask value in `[0, 1]` (sigmoid) |
| Loss | MSE between predicted mask and Ideal Ratio Mask (IRM) |
| Normalization | Per-feature mean-variance (frozen at inference) |

### Ground-Truth: Ideal Ratio Mask (IRM)

Labels are generated mathematically from the synthetic mixing process:

```
IRM = sqrt( S² / (S² + N²) )
```

where **S** = clean speech magnitude and **N** = noise magnitude per time-frequency bin.

### Baselines

| Model | Description | Parameter Matching |
|---|---|---|
| **Basic MLP** | Single-frame input (128-dim), no temporal context | Matched to T-MLP |
| **GRU** | Recurrent baseline, small but competitive | Comparable size |

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
| **LibriSpeech** or **VCTK** | Clean, labeled speech | Public |
| **MUSAN** or **ESC-50** | Environmental noise (transients, ambient) | Public |

Speech and noise are synthetically mixed at SNR levels from −5 dB to +15 dB.

| Split | Duration |
|---|---|
| Training | 10–20 hours |
| Validation | ~2 hours |
| Test | ~2 hours |
| Real-recorded (optional) | 15–30 min (test-only, phone/earbuds capture) |

---

## Repository Structure

```
T-MLP-for-Environmental-Noise-Cancellation/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── configs/                    # Experiment configuration files
│   └── default.yaml
│
├── data/                       # Data pipeline
│   ├── download.py             # Dataset download scripts
│   ├── mix.py                  # Synthetic speech + noise mixing
│   └── dataset.py              # PyTorch Dataset & DataLoader
│
├── features/                   # Feature extraction & buffering
│   ├── extractor.py            # Log-mel spectrogram extraction
│   ├── normalizer.py           # Per-feature mean-variance normalization
│   └── sliding_window.py       # Circular-buffer sliding window module
│
├── models/                     # Model definitions
│   ├── basic_mlp.py            # Single-frame MLP baseline
│   ├── t_mlp.py                # Temporal MLP (sliding window)
│   └── gru_baseline.py         # GRU/RNN baseline
│
├── training/                   # Training & QAT
│   ├── train.py                # FP32 training loop
│   ├── qat.py                  # Quantization-Aware Training
│   └── losses.py               # Loss functions (MSE, L1)
│
├── evaluation/                 # Evaluation & analysis
│   ├── metrics.py              # NMSE, SDR, STOI, PESQ, F1
│   ├── evaluate.py             # Run evaluation on test set
│   ├── hardware_estimate.py    # Analytical latency, RAM, MACs
│   └── plot.py                 # Quality-vs-efficiency plotting
│
├── export/                     # Model export
│   └── onnx_export.py          # PyTorch → ONNX conversion
│
├── notebooks/                  # Exploratory analysis
│   └── exploration.ipynb
│
├── audio_samples/              # Before/after listening samples
│
└── results/                    # Saved metrics, plots, reports
    ├── metrics/
    └── plots/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, not required)

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/T-MLP-for-Environmental-Noise-Cancellation.git
cd T-MLP-for-Environmental-Noise-Cancellation

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare Data

```bash
# Download datasets
python data/download.py --dataset librispeech musan

# Generate synthetic mixtures
python data/mix.py --snr-range -5 15 --hours 10
```

### 2. Train Models

```bash
# Train FP32 T-MLP
python training/train.py --model t_mlp --config configs/default.yaml

# Train baselines
python training/train.py --model basic_mlp --config configs/default.yaml
python training/train.py --model gru --config configs/default.yaml
```

### 3. Quantization-Aware Training

```bash
python training/qat.py --checkpoint results/t_mlp_fp32.pt
```

### 4. Evaluate

```bash
# Run evaluation across all models
python evaluation/evaluate.py --models basic_mlp t_mlp gru --quantized

# Generate hardware estimates & plots
python evaluation/hardware_estimate.py
python evaluation/plot.py
```

### 5. Export to ONNX

```bash
python export/onnx_export.py --checkpoint results/t_mlp_int8.pt
```

---

## Evaluation Metrics

### Accuracy & Quality

| Metric | Description |
|---|---|
| **NMSE** | Normalized Mean Squared Error between predicted and ideal mask |
| **Precision / Recall / F1** | Thresholded mask classification performance |
| **SDR** | Signal-to-Distortion Ratio (dB) |
| **STOI** | Short-Time Objective Intelligibility [0, 1] |
| **PESQ** (optional) | Perceptual Evaluation of Speech Quality |

### Efficiency (Analytical Estimates)

| Metric | Method |
|---|---|
| **Parameter count** | Total trainable parameters |
| **MACs/s** | Multiply-accumulate operations per second of audio |
| **RAM estimate** | `params × bytes_per_param + activation_buffer` |
| **ONNX latency** | Per-frame inference time via ONNX Runtime |
| **Quantization loss** | FP32-to-INT8 NMSE degradation (target: < 2%) |

---

## Results

All models are compared in a unified table:

| Model | NMSE ↓ | SDR ↑ | STOI ↑ | Params | MACs/s | RAM (KB) | INT8 NMSE Δ |
|---|---|---|---|---|---|---|---|
| Basic MLP | — | — | — | — | — | — | — |
| **T-MLP** | — | — | — | — | — | — | — |
| GRU | — | — | — | — | — | — | — |

*Table to be populated after training and evaluation.*

The **quality-vs-efficiency plot** (NMSE or SDR vs MACs) is the key visualization supporting the central claim.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **PyTorch** | Training, QAT (`torch.ao.quantization`) |
| **librosa / torchaudio** | Feature extraction, audio I/O |
| **ONNX Runtime** | Inference latency profiling |
| **matplotlib** | Plotting and visualization |

---

## License

This project is for academic and research purposes. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Software-only prototype — all hardware efficiency claims are analytical estimates, not physical measurements.</i>
</p>