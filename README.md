# T-MLP for Environmental Noise Cancellation

> **Lightweight Temporal-MLP for Speech Enhancement on Resource-Constrained Edge Devices**

An experimental deep-learning project investigating a lightweight **Temporal Multi-Layer Perceptron (T-MLP)** for environmental noise cancellation and speech enhancement on resource-constrained edge devices such as **True Wireless Stereo (TWS) earbuds**.

The project compares a proposed lightweight T-MLP against conventional neural-network baselines while investigating the trade-off between **speech enhancement quality, computational complexity, inference latency, and deployment efficiency**.

### Models

- **MLP** — Fully connected baseline
- **T-MLP** — Proposed lightweight temporal architecture
- **GRU** — Recurrent baseline
- **T-MLP + QAT** — Quantization-aware training experiment

> **Project Status:** Experimental / Ongoing  
> MLP and T-MLP currently provide preliminary usable results. GRU and QAT require further investigation before their results are treated as final.

## Table of Contents

- [Motivation](#motivation)
- [Objectives](#objectives)
- [System Overview](#system-overview)
- [Dataset](#dataset)
- [Dataset Split](#dataset-split)
- [Audio Preprocessing](#audio-preprocessing)
- [Noisy Speech Generation](#noisy-speech-generation)
- [Input Representation](#input-representation)
- [Model Architectures](#model-architectures)
  - [MLP Baseline](#mlp-baseline)
  - [Temporal-MLP](#temporal-mlp)
  - [GRU Baseline](#gru-baseline)
  - [T-MLP with QAT](#t-mlp-with-qat)
- [Experimental Protocol](#experimental-protocol)
- [Evaluation Metrics](#evaluation-metrics)
- [Current Results](#current-results)
- [Preliminary Findings](#preliminary-findings)
- [Known Issues](#known-issues)
- [Computational Efficiency](#computational-efficiency)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Roadmap](#roadmap)
- [Research Direction](#research-direction)
- [Limitations](#limitations)
- [Final Evaluation Plan](#final-evaluation-plan)
- [License](#license)

# Motivation

Speech enhancement for edge audio devices presents a different problem from conventional deep-learning inference on servers or desktop GPUs.

A model intended for devices such as TWS earbuds must operate under constraints involving:

| Requirement | Importance |
|---|---|
| Low parameter count | Reduces memory requirements |
| Low computational cost | Reduces processor workload |
| Low inference latency | Important for real-time audio |
| Small model size | Important for embedded storage |
| Robust speech enhancement | Maintains speech intelligibility |
| Quantization compatibility | Enables efficient integer inference |
| Low power consumption | Important for battery-operated devices |

Large neural architectures may provide strong enhancement performance but can be difficult to deploy on highly constrained hardware.

This project therefore investigates a lightweight alternative based on a **Temporal-MLP architecture**.

The central idea is to determine whether temporal information can be processed effectively using lightweight MLP-based blocks while avoiding the recurrent computation associated with architectures such as GRUs.

# Objectives

The project is organized around four major objectives.

## 1. Speech Enhancement

Evaluate how effectively each model reconstructs clean speech from noisy input.

The current models operate directly on waveform samples.

## 2. Model Efficiency

Compare the models using:

- Parameter count
- Model size
- Computational cost
- Memory requirements
- Inference latency

## 3. Speech Quality

The final evaluation will include both waveform-level and speech-specific metrics.

Planned metrics include:

- MSE
- RMSE
- MAE
- SDR
- SI-SDR
- STOI

## 4. Edge Deployment

The eventual goal is to investigate deployment using:

- Quantization-Aware Training
- INT8 inference
- TensorFlow Lite
- Edge-device benchmarking

The final evaluation will therefore consider the trade-off between:

**Speech Quality ↔ Model Efficiency ↔ Deployment Feasibility**

# System Overview

The complete pipeline is structured as follows:

                    ┌─────────────────────┐
                    │     LibriSpeech     │
                    │       Speech        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Speech           │
                    │   Preprocessing    │
                    └──────────┬──────────┘
                               │
                               │
                    ┌──────────▼──────────┐
                    │      ESC-50         │
                    │       Noise         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Noise Mixing     │
                    │  Controlled SNRs    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Noisy Speech     │
                    │   256-sample input  │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
          ┌─────────┐     ┌──────────┐     ┌─────────┐
          │   MLP   │     │  T-MLP   │     │   GRU   │
          └────┬────┘     └────┬─────┘     └────┬────┘
               │               │               │
               └───────────────┼───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Enhanced Speech     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Evaluation Pipeline │
                    │ Quality + Efficiency │
                    └─────────────────────┘


# Dataset

The project currently uses two publicly available audio datasets:

- **LibriSpeech** for clean speech
- **ESC-50** for environmental noise

## Speech Dataset — LibriSpeech

| Property | Value |
|---|---:|
| Number of files | 5,323 |
| Sample rate | 16 kHz |
| Channels | Mono |
| Total duration | ~647.48 minutes |

The speech dataset is used as the clean-speech source.

## Noise Dataset — ESC-50

| Property | Value |
|---|---:|
| Number of files | 2,000 |
| Original sample rate | 44.1 kHz |
| Channels | Mono |
| Total duration | ~166.67 minutes |

The noise files are resampled to the project's target sample rate before mixing.

# Dataset Split

The dataset is divided into training, validation, and test partitions.

## Speech Split

| Split | Files |
|---|---:|
| Training | 4,223 |
| Validation | 537 |
| Test | 563 |
| **Total** | **5,323** |

## Noise Split

| Split | Files |
|---|---:|
| Training | 1,400 |
| Validation | 300 |
| Test | 300 |
| **Total** | **2,000** |

The test data is kept separate from the training and validation data to provide a held-out evaluation set.

# Audio Preprocessing

The audio preprocessing pipeline converts the datasets into a common representation.

Input Audio
     │
     ▼
Load Audio
     │
     ▼
Convert to Mono
     │
     ▼
Check for Empty Audio
     │
     ▼
Check for NaN / Inf
     │
     ▼
Resample
     │
     ▼
16 kHz Mono Audio

## Target Audio Format

| Property | Target |
|---|---|
| Sample Rate | 16,000 Hz |
| Channels | Mono |
| Representation | Waveform |

The preprocessing pipeline uses:

- `soundfile`
- `scipy.signal.resample_poly`
- `numpy`

Audio files are validated before being used for dataset generation.

Invalid files containing empty waveforms or non-finite values are rejected.

# Noisy Speech Generation

Clean speech and environmental noise are combined to create controlled noisy-speech samples.

The mixing process uses predefined Signal-to-Noise Ratio (SNR) levels.

## SNR Configuration
-5 dB
 0 dB
+5 dB
+10 dB
+15 dB

This provides coverage from heavily noise-dominated conditions to relatively clean conditions.

## Current Generated Dataset

| Split | Mixtures |
|---|---:|
| Training | 800 |
| Validation | 100 |
| Test | 100 |
| **Total** | **1,000** |

The current dataset therefore contains:
800 training mixtures
100 validation mixtures
100 test mixtures
-------------------
1000 total mixtures


## Stored Information

Each generated mixture stores:

- Noisy speech waveform
- Clean speech waveform
- Speech filename
- Noise filename
- Target SNR

This allows the experiment to retain the clean reference required for supervised learning and evaluation.

# Input Representation

The models operate on short waveform windows.

| Parameter | Value |
|---|---:|
| Sample rate | 16 kHz |
| Window size | 256 samples |
| Hop size | 128 samples |
| Window duration | 16 ms |

The duration of a single input window is:
256 / 16000 = 0.016 seconds

Therefore:
256 samples = 16 ms

The 128-sample hop corresponds to:
128 / 16000 = 8 ms

# Model Architectures

The project currently compares three primary architectures and one quantization experiment:
MLP
 │
 ├── Conventional fully-connected baseline
 │
T-MLP
 │
 ├── Proposed lightweight architecture
 │
GRU
 │
 ├── Recurrent baseline
 │
T-MLP + QAT
 │
 └── Quantization-aware training experiment

# MLP Baseline

The MLP serves as a conventional fully-connected neural-network baseline.

## Architecture

Input
256 samples
     │
     ▼
Dense(512, ReLU)
     │
     ▼
Dense(256, ReLU)
     │
     ▼
Dense(256, Linear)
     │
     ▼
Output
256 samples

## Parameters

Approximately:
328,704 trainable parameters

The MLP is intentionally used as a relatively larger baseline against which the lightweight T-MLP can be compared.

# Temporal-MLP

The proposed model is designed to reduce parameter count while retaining useful temporal information through lightweight feature transformations and residual connections.

## Architecture

Input
256 samples
     │
     ▼
Reshape
(256, 1)
     │
     ▼
Dense(64, ReLU)
     │
     ▼
┌─────────────────────────────┐
│     Temporal MLP Block 1    │
│                             │
│ Dense(64, ReLU)             │
│        │                    │
│ Dropout(0.1)                │
│        │                    │
│ Dense(64, Linear)           │
│        │                    │
│ Residual Add                │
│        │                    │
│ LayerNorm                   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Temporal MLP Block 2    │
│                             │
│ Dense(64, ReLU)             │
│        │                    │
│ Dropout(0.1)                │
│        │                    │
│ Dense(64, Linear)           │
│        │                    │
│ Residual Add                │
│        │                    │
│ LayerNorm                   │
└──────────────┬──────────────┘
               │
               ▼
       Dense(32, ReLU)
               │
               ▼
       Dense(1, Linear)
               │
               ▼
           Reshape
               │
               ▼
        Output: 256 samples

## Parameters

18,881 trainable parameters

Compared with the MLP baseline, the T-MLP is substantially smaller.

# GRU Baseline

A GRU-based recurrent network is included to provide a recurrent comparison.

## Architecture

```text
Input
(256, 1)
     │
     ▼
GRU(64)
return_sequences=True
     │
     ▼
GRU(64)
return_sequences=True
     │
     ▼
Dense(1)
     │
     ▼
Output
(256, 1)

## Parameters

Approximately:

37,697 parameters

The GRU is intended to provide a comparison against a model that explicitly processes sequential information through recurrent state.

# T-MLP with QAT

Quantization-Aware Training was investigated to prepare the lightweight architecture for eventual integer-based deployment.

The QAT implementation uses standard Keras layers to improve compatibility with TensorFlow Model Optimization.

## Current Architecture

Input
     │
     ▼
Reshape
     │
     ▼
Dense(64, ReLU)
     │
     ▼
Dense(64, ReLU)
     │
     ▼
Dense(64, Linear)
     │
     ▼
Residual Add
     │
     ▼
Dense(64, ReLU)
     │
     ▼
Dense(64, Linear)
     │
     ▼
Residual Add
     │
     ▼
Dense(32, ReLU)
     │
     ▼
Dense(1, Linear)
     │
     ▼
Reshape
     │
     ▼
Output

## Parameter Count

| Parameter | Count |
|---|---:|
| Trainable parameters | 18,881 |
| Total QAT parameters | 18,927 |
| Non-trainable quantization parameters | 46 |

> **Note:** The current QAT architecture is not perfectly identical to the FP32 T-MLP. In particular, the current QAT implementation does not contain the Dropout and LayerNorm layers used by the FP32 T-MLP. A more controlled QAT experiment is therefore planned.

# Experimental Protocol

The current experiment uses the following configuration:

| Parameter | Configuration |
|---|---|
| Sample rate | 16 kHz |
| Channels | Mono |
| Window size | 256 samples |
| Hop size | 128 samples |
| Window duration | 16 ms |
| SNR levels | -5, 0, 5, 10, 15 dB |
| Training mixtures | 800 |
| Validation mixtures | 100 |
| Test mixtures | 100 |
| Random seed | 42 |

## Dataset Windows

The generated mixtures are converted into overlapping windows for model training and evaluation.

Current window counts:

| Split | Windows |
|---|---:|
| Training | 732,388 |
| Validation | 80,583 |
| Test | 85,649 |

The same held-out test set is intended to be used for the final comparison of all models.

# Evaluation Metrics

The evaluation is divided into three major categories.

## 1. Waveform Reconstruction

### Mean Squared Error

Measures the average squared difference between predicted and clean waveform samples.
MSE = mean((y_true - y_pred)^2)

### Root Mean Squared Error
RMSE = sqrt(MSE)

### Mean Absolute Error
MAE = mean(|y_true - y_pred|)

These metrics are useful for measuring waveform reconstruction error but do not fully represent perceived speech quality.

# 2. Speech Quality

The final evaluation will include speech-specific metrics.

### SDR

Signal-to-Distortion Ratio will be used to evaluate separation/enhancement quality.

### SI-SDR

Scale-Invariant Signal-to-Distortion Ratio will be used to provide a scale-independent measure of enhancement quality.

### STOI

Short-Time Objective Intelligibility will be used to evaluate speech intelligibility.

# 3. Efficiency

The models will also be compared using:

- Parameter count
- Model size
- MACs / computational operations
- Inference latency
- Memory requirements
- Quantized model size
- Deployment latency

# Current Results

## MLP vs T-MLP

The current FP32 results are:

| Metric | MLP | T-MLP |
|---|---:|---:|
| Parameters | ~328.7K | **18.9K** |
| Test MSE | 0.003212 | **0.001323** |
| Test RMSE | 0.056671 | **0.036371** |
| Test MAE | 0.031482 | **0.021302** |
| Mean measured latency | 75.4130 ms | **67.9323 ms** |
| Median latency | 70.9007 ms | **66.6672 ms** |
| Training time | 691.44 s | 1020.93 s |

# GRU Results

The current GRU training run completed, but evaluation produced NaN values.

| Metric | GRU |
|---|---:|
| Test MSE | NaN |
| Test RMSE | NaN |
| Test MAE | NaN |
| Training time | 4541.21 s |

The model generated predictions with shape:


(85649, 256, 1)

However, because the evaluation metrics are NaN, the current GRU result is **not considered a valid final baseline**.

The cause of the instability is currently under investigation.

# T-MLP-QAT Results

The current QAT experiment produced:

| Metric | T-MLP-QAT |
|---|---:|
| Test MSE | 0.00368405 |
| Test RMSE | 0.06069640 |
| Test MAE | 0.04045161 |
| Training time | 3329.79 s |

The current QAT run showed training instability and therefore requires a controlled retraining experiment.

# Preliminary Findings

The current results suggest that the T-MLP can achieve lower waveform reconstruction error than the conventional MLP while using substantially fewer parameters.

## Parameter Reduction

Current parameter counts:


MLP   : 328,704
T-MLP : 18,881


Approximate parameter reduction:

1 - (18,881 / 328,704)
≈ 94.3%

Therefore, the current T-MLP uses approximately **94% fewer parameters** than the MLP baseline.

## Reconstruction Performance

Current test results show:

MLP RMSE
0.056671

T-MLP RMSE
0.036371

The current T-MLP RMSE is approximately **35.8% lower** than the MLP result.

Similarly:

MLP MAE
0.031482

T-MLP MAE
0.021302

The current T-MLP MAE is approximately **32.3% lower**.

These results are currently limited to waveform-level reconstruction metrics.

# Important Interpretation

Lower MSE, RMSE, or MAE does **not automatically mean better perceived speech enhancement**.

Speech enhancement must also be evaluated using metrics such as:

SDR
SI-SDR
STOI

and ideally through qualitative listening tests.

Therefore, the current results should be interpreted as:

> **Preliminary waveform reconstruction results rather than a final claim of superior speech-enhancement quality.**

# Latency Considerations

The current latency measurements were obtained using TensorFlow/Keras on the host system.

Current measurements:

| Model | Mean Latency |
|---|---:|
| MLP | 75.4130 ms |
| T-MLP | 67.9323 ms |

The input window is:

256 samples @ 16 kHz
=
16 ms

The measured host-side T-MLP latency is approximately:

67.9 ms


Therefore, the current measurements **do not demonstrate real-time embedded operation**.

The current benchmark includes the overhead of the host-side TensorFlow/Keras inference pipeline and is not equivalent to optimized inference on an embedded processor.

Future benchmarking will use:

- TensorFlow Lite
- INT8 inference
- Representative edge hardware
- Controlled measurement methodology

before making any deployment-related claims.

# Known Issues

## GRU Numerical Instability

The GRU currently produces NaN values during test evaluation.

Possible areas for investigation include:

- Input ranges
- Output ranges
- NaN/Inf propagation
- Learning rate
- Gradient stability
- Training loss
- Batch-level numerical behavior

The GRU will be retrained only after the numerical issue has been diagnosed.

## QAT Training Instability

The current QAT experiment showed a large temporary loss spike.

Example:

Epoch 1
Loss ≈ 0.0022

Epoch 2
Loss ≈ 80.6170

Epoch 3
Loss ≈ 0.0020

Although training subsequently recovered, this behavior needs investigation before the QAT results can be treated as reliable.

---

## QAT Architecture Mismatch

The FP32 T-MLP currently uses:

Dropout
LayerNorm
Residual connections

while the QAT implementation removes Dropout and LayerNorm to improve quantization compatibility.

Therefore, the current QAT comparison is not a perfectly controlled:

FP32 T-MLP
      vs
QAT T-MLP

comparison.

A future experiment will use as similar an architecture as possible between FP32 and QAT models.

# Computational Efficiency

The final efficiency analysis will compare:

| Metric | MLP | T-MLP | GRU | T-MLP-QAT |
|---|---:|---:|---:|---:|
| Parameters | ~328.7K | ~18.9K | ~37.7K | ~18.9K |
| Model size | TBD | TBD | TBD | TBD |
| MACs | TBD | TBD | TBD | TBD |
| Memory | TBD | TBD | TBD | TBD |
| Latency | 75.41 ms | 67.93 ms | TBD | TBD |
| INT8 size | — | — | — | TBD |
| Edge latency | TBD | TBD | TBD | TBD |

This table will be updated after the remaining experiments are completed.

# Reproducibility

The dataset generation process uses a fixed random seed:

np.random.default_rng(42)

Current dataset configuration:

Training mixtures   : 800
Validation mixtures : 100
Test mixtures       : 100

Total               : 1000

SNR levels:
-5 dB
0 dB
5 dB
10 dB
15 dB

Audio configuration:

Sample Rate:
16,000 Hz

Channels:
Mono

Window:
256 samples

Hop:
128 samples

# Model Checkpoints

The intended persistent model directory is:

saved_models/
├── MLP_best.keras
├── TMLP_best.keras
├── GRU_best.keras
└── TMLP_QAT_best.keras

Best-validation-loss checkpoints are intended to be used for final evaluation rather than arbitrary final-epoch models.

---

# Repository Structure

T-MLP-for-Environmental-Noise-Cancellation/
│
├── data/
│   ├── speech/
│   └── noise/
│
├── processed_data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── data_pipeline/
│   ├── data_preprocessing.py
│   ├── inspect_data.py
│   ├── mixing.py
│   └── dataset.py
│
├── models/
│   ├── mlp.py
│   ├── t_mlp.py
│   ├── GRU.py
│   └── tmlp_qat.py
│
├── saved_models/
│   ├── MLP_best.keras
│   ├── TMLP_best.keras
│   ├── GRU_best.keras
│   └── TMLP_QAT_best.keras
│
├── results/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md

Large datasets and trained model binaries may be excluded from Git version control depending on repository size and distribution requirements.

# Roadmap

## Phase 1 — Dataset & Preprocessing

- [x] Inspect speech dataset
- [x] Inspect noise dataset
- [x] Convert audio to mono
- [x] Resample audio to 16 kHz
- [x] Validate audio files
- [x] Create train/validation/test split
- [x] Generate controlled noisy-speech mixtures
- [x] Generate 1,000 total mixtures

## Phase 2 — Model Development

- [x] Implement MLP baseline
- [x] Train MLP
- [x] Implement Temporal-MLP
- [x] Train Temporal-MLP
- [x] Implement GRU baseline
- [x] Initial QAT implementation
- [ ] Diagnose GRU NaN issue
- [ ] Retrain GRU
- [ ] Stabilize QAT training

## Phase 3 — Controlled Evaluation

- [ ] Build unified evaluation pipeline
- [ ] Evaluate MSE
- [ ] Evaluate RMSE
- [ ] Evaluate MAE
- [ ] Evaluate SDR
- [ ] Evaluate SI-SDR
- [ ] Evaluate STOI
- [ ] Evaluate noisy baseline
- [ ] Evaluate improvement over noisy input
- [ ] Evaluate performance by SNR

## Phase 4 — Efficiency Analysis

- [ ] Calculate parameter count
- [ ] Measure model size
- [ ] Calculate MACs
- [ ] Estimate memory requirements
- [ ] Standardize latency benchmarking
- [ ] Compare computational efficiency

## Phase 5 — Quantization

- [x] Initial QAT implementation
- [ ] Diagnose QAT instability
- [ ] Align FP32 and QAT architectures
- [ ] Retrain QAT model
- [ ] Export INT8 model
- [ ] Validate quantized model
- [ ] Measure quality degradation after quantization

## Phase 6 — Edge Deployment

- [ ] TensorFlow Lite conversion
- [ ] INT8 inference
- [ ] Edge-device benchmarking
- [ ] Measure inference latency
- [ ] Measure memory usage
- [ ] Measure model size
- [ ] Evaluate power/efficiency where hardware permits

# Research Direction

The central research question is:

> **Can a lightweight Temporal-MLP provide useful speech-enhancement performance while substantially reducing model complexity for resource-constrained edge audio systems?**

The final experiment will therefore not focus on a single metric.

Instead, the models will be evaluated across multiple dimensions:

                 ┌──────────────────┐
                 │  Speech Quality  │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Model Trade-off  │
                └────────────────────┘
                    ▲       ▲      ▲
                    │       │      │
                    │       │      │
             Efficiency   Latency  Size
                    │       │      │
                    └───────┼──────┘
                            │
                            ▼
                    Edge Deployment


The final analysis will investigate the trade-off between:

- Speech enhancement quality
- Parameter count
- Model size
- Computational cost
- Inference latency
- Quantization robustness
- Deployment feasibility

# Limitations

The current experiment has several limitations.

## Dataset Size

The current generated mixture dataset contains:

1000 mix samples

Although this produces a much larger number of overlapping training windows, the number of unique mixture configurations remains limited.

## Evaluation Metrics

The current reported results primarily use:

- MSE
- RMSE
- MAE

Speech-specific metrics such as SDR, SI-SDR, and STOI have not yet been incorporated into the final comparison.

## Host-Side Latency

Current latency measurements are TensorFlow/Keras host-side measurements.

They should not be interpreted as embedded-device latency.

## GRU Baseline

The GRU currently produces NaN evaluation results and therefore requires debugging before it can be used as a reliable quantitative comparison.

## QAT Experiment

The current QAT experiment is preliminary and uses an architecture that differs slightly from the FP32 T-MLP.

A controlled retraining experiment is required.

# Final Evaluation Plan

The final evaluation will use the same held-out test set for all valid models.

For every test sample:

Clean Speech
     │
     ├──────────────┐
     │              │
     ▼              ▼
Noisy Speech     Reference
     │
     ├──────────────┐
     │              │
     ▼              ▼
   MLP           T-MLP
     │              │
     └──────┬───────┘
            │
            ▼
          GRU
            │
            ▼
       T-MLP-QAT
            │
            ▼
       Evaluation

Each model will be evaluated using:

Waveform Quality
├── MSE
├── RMSE
└── MAE

Speech Quality
├── SDR
├── SI-SDR
└── STOI

Efficiency
├── Parameters
├── Model Size
├── MACs
├── Memory
└── Latency

Results will additionally be reported separately for:

-5 dB
 0 dB
+5 dB
+10 dB
+15 dB

This will make it possible to determine how model behavior changes as environmental noise severity changes.

# Expected Final Comparison

The final results table will follow this structure:

| Metric | MLP | T-MLP | GRU | T-MLP-QAT |
|---|---:|---:|---:|---:|
| Parameters | ~328.7K | ~18.9K | ~37.7K | ~18.9K |
| Model Size | TBD | TBD | TBD | TBD |
| MSE | 0.003212 | 0.001323 | TBD | 0.003684 |
| RMSE | 0.056671 | 0.036371 | TBD | 0.060696 |
| MAE | 0.031482 | 0.021302 | TBD | 0.040452 |
| SDR | TBD | TBD | TBD | TBD |
| SI-SDR | TBD | TBD | TBD | TBD |
| STOI | TBD | TBD | TBD | TBD |
| MACs | TBD | TBD | TBD | TBD |
| Latency | 75.41 ms | 67.93 ms | TBD | TBD |
| Training Time | 691 s | 1021 s | 4541 s | 3330 s |

> Values marked `TBD` will be populated after the corresponding experiments are completed.

# Project Philosophy

This project is being developed as an **engineering and experimental study**, rather than simply a model-training exercise.

The focus is therefore on:

Dataset
   ↓
Controlled Experiment
   ↓
Model Comparison
   ↓
Quality Evaluation
   ↓
Efficiency Evaluation
   ↓
Quantization
   ↓
Deployment


Every optimization should be evaluated against both:

1. **What performance is gained?**
2. **What computational/deployment cost is introduced?**

This is particularly important for audio systems intended for resource-constrained edge devices.

# License

This project is currently an experimental research and engineering project.

The licensing conditions of the underlying datasets should be reviewed independently before redistributing any dataset files.

# Acknowledgements

This project uses:

- **LibriSpeech** for speech data
- **ESC-50** for environmental noise data
- **TensorFlow / Keras** for model development
- **TensorFlow Model Optimization** for quantization-aware training

# Current Milestone

```text
[x] Dataset preparation
[x] Audio preprocessing
[x] Controlled mixture generation
[x] MLP baseline
[x] T-MLP
[x] Initial GRU
[x] Initial QAT
[ ] GRU stabilization
[ ] QAT stabilization
[ ] SDR / SI-SDR / STOI evaluation
[ ] Computational analysis
[ ] INT8 deployment
[ ] Edge-device benchmarking
```

> **Next milestone:** Build a unified evaluation pipeline and obtain reliable speech-quality and efficiency measurements across the complete model set.