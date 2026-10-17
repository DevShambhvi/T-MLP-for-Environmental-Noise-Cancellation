# T-MLP for Environmental Noise Cancellation

> **A lightweight Temporal-MLP investigation for waveform-based speech enhancement on resource-constrained edge devices**

This project investigates whether a compact Temporal Multi-Layer Perceptron (T-MLP) can provide useful speech-enhancement performance with fewer parameters than a conventional multilayer perceptron (MLP), while exploring the trade-offs among reconstruction quality, speech intelligibility, model size, and inference latency.

The experiments use clean speech from **LibriSpeech** and environmental noise from **ESC-50**. The current unified evaluation compares three saved FP32 models: MLP, T-MLP, and GRU. A separate T-MLP quantization-aware training (QAT) experiment exists, but it is **not included in the unified comparison reported below**.

> **Project status:** Experimental research project. The reported results are from the current evaluation run on a 100-sample held-out test set. They do not establish real-time performance or production readiness on earbuds or other embedded hardware.

## Table of contents

- [Motivation](#motivation)
- [Objectives](#objectives)
- [Pipeline overview](#pipeline-overview)
- [Datasets](#datasets)
- [Dataset generation and preprocessing](#dataset-generation-and-preprocessing)
- [Input representation](#input-representation)
- [Models](#models)
  - [MLP baseline](#mlp-baseline)
  - [Temporal-MLP](#temporal-mlp)
  - [GRU baseline](#gru-baseline)
  - [T-MLP QAT experiment](#t-mlp-qat-experiment)
- [Evaluation protocol](#evaluation-protocol)
- [Evaluation metrics](#evaluation-metrics)
- [Current results](#current-results)
- [Results by input SNR](#results-by-input-snr)
- [Interpretation](#interpretation)
- [Efficiency and latency notes](#efficiency-and-latency-notes)
- [Known issues and limitations](#known-issues-and-limitations)
- [Reproducibility](#reproducibility)
- [Expected repository structure](#expected-repository-structure)
- [Roadmap](#roadmap)
- [Research direction](#research-direction)
- [Datasets and acknowledgements](#datasets-and-acknowledgements)
- [License](#license)

## Motivation

Speech enhancement on small audio devices differs from inference on a desktop or server. An eventual earbud-class implementation would need to balance:

- Speech quality and intelligibility
- Parameter count and model storage
- Computational cost and runtime memory
- End-to-end latency
- Quantization compatibility
- Power consumption

This project explores a lightweight waveform-based model and compares it with conventional MLP and recurrent GRU baselines. The target application is a motivation for the work, not a claim that the current models have already been deployed or validated on TWS hardware.

## Objectives

1. **Waveform reconstruction:** Estimate clean speech from noisy waveform input.
2. **Model-size comparison:** Compare parameter count and saved checkpoint size.
3. **Speech-enhancement evaluation:** Assess reconstruction error, SI-SDR improvement, and STOI.
4. **SNR-specific analysis:** Check how results vary across different input noise levels.
5. **Future deployment investigation:** Explore QAT, integer quantization, TensorFlow Lite, and representative edge-device benchmarks.

## Pipeline overview

```text
LibriSpeech clean speech ──> Audio preprocessing ──┐
                                                   ├──> SNR-controlled mixing
ESC-50 environmental noise ─> Audio preprocessing ┘
                                                            │
                                                            ▼
                                                 Noisy speech mixtures
                                                            │
                                                            ▼
                                                256-sample waveform windows
                                                            │
                                   ┌────────────────────────┼──────────────────────┐
                                   ▼                        ▼                      ▼
                                  MLP                      T-MLP                  GRU
                                   └────────────────────────┼──────────────────────┘
                                                            ▼
                                               Enhanced waveform predictions
                                                            │
                                                            ▼
                                  MSE / RMSE / MAE / SI-SDR / SI-SDR improvement / STOI
                                                            │
                                                            ▼
                                              Model-size and latency analysis
```

## Datasets

The project uses two public audio datasets.

### LibriSpeech — clean speech

| Property | Value |
|---|---:|
| Files inspected | 5,323 |
| Target sample rate | 16,000 Hz |
| Channel format | Mono |
| Total duration reported during inspection | Approximately 647.48 minutes |

LibriSpeech provides clean speech recordings used as the reference signal.

### ESC-50 — environmental noise

| Property | Value |
|---|---:|
| Files inspected | 2,000 |
| Original sample rate | 44,100 Hz |
| Channel format | Mono |
| Total duration reported during inspection | Approximately 166.67 minutes |

ESC-50 provides environmental sounds used to create noisy mixtures. Noise audio is resampled to the target sample rate before mixing.

These figures describe the files inspected during project preparation; they should not be interpreted as proof that every file from either original dataset is used in each generated split.

## Dataset generation and preprocessing

### Audio preprocessing

The preprocessing pipeline uses:

- `soundfile` to load audio
- `numpy` for waveform handling and validation
- `scipy.signal.resample_poly` for resampling

The target representation is **16 kHz, mono, floating-point waveform audio**. Stereo input is converted to mono by averaging channels. Empty waveforms and non-finite values are checked before use.

### Noisy-speech generation

Clean speech and environmental noise are mixed at five target signal-to-noise ratios:

- −5 dB
- 0 dB
- +5 dB
- +10 dB
- +15 dB

The generated dataset currently contains 1,000 saved mixtures:

| Split | Mixtures |
|---|---:|
| Training | 800 |
| Validation | 100 |
| Test | 100 |
| **Total** | **1,000** |

The intended distribution is 200 mixtures per target SNR across the full dataset. The generated samples retain the noisy waveform, clean reference waveform, source filenames, and target SNR.

The split and generation configuration uses a random seed of `42`. The number of overlapping windows produced from these mixtures is much larger than the number of distinct mixture configurations; overlapping windows should not be treated as independent recordings.

### Important waveform-range limitation

The current mixing implementation adds scaled noise to clean speech without clipping or normalizing the resulting sum. A previous inspection found some mixed waveform values outside the conventional `[-1, 1]` range. The current reported evaluation should therefore be interpreted in light of the actual stored waveform amplitudes and the current mixing implementation.

This README does **not** claim that this issue has been corrected, and the published results should not be described as results from a corrected or regenerated dataset.

## Input representation

The models process short waveform windows.

| Parameter | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Window size | 256 samples |
| Window duration | 16 ms |
| Hop size | 128 samples |
| Hop duration | 8 ms |

At 16 kHz, a 256-sample window spans \(256 / 16000 = 0.016\) seconds, or 16 ms. A 128-sample hop corresponds to 8 ms.

The window and hop describe the audio framing configuration. They do not, by themselves, establish end-to-end real-time performance.

## Models

The unified evaluation compares the saved **MLP**, **T-MLP**, and **GRU** checkpoints. Parameter counts and checkpoint sizes in the results section are taken from the current evaluation outputs; these measured values take precedence over older architecture estimates elsewhere in project notes.

### MLP baseline

The MLP is the conventional fully connected baseline. Its current evaluated checkpoint contains **328,704 parameters**.

It provides a larger dense-network reference for comparison with the compact T-MLP.

### Temporal-MLP

The T-MLP is the proposed lightweight model. Its design uses compact dense transformations and temporal-window processing to estimate a clean waveform window.

The currently evaluated checkpoint contains **19,137 parameters** and has a reported checkpoint size of **0.286 MB**. This is approximately **94.2% fewer parameters** than the evaluated MLP checkpoint.

The current checkpoint's measured parameter count is used in the results, rather than the earlier approximate architecture count of 18,881.

### GRU baseline

The GRU provides a recurrent baseline for comparison with the MLP-based approaches. Its evaluated checkpoint contains **37,889 parameters** and has a reported checkpoint size of **0.467 MB**.

In the current unified evaluation, the GRU achieves the strongest mean SI-SDR improvement and lowest mean MSE and MAE, but its measured host-side latency is substantially higher than that of the MLP and T-MLP.

### T-MLP QAT experiment

A separate quantization-aware training experiment was developed to investigate a possible path toward quantized inference. The earlier QAT implementation differs from the FP32 T-MLP architecture, including differences involving Dropout and Layer Normalization.

An earlier QAT run reported MSE of 0.00368405, RMSE of 0.06069640, and MAE of 0.04045161, alongside training instability. These are **historical preliminary results from a separate run**, not results from the current unified evaluation. They should not be compared directly with the unified table as if they were produced under an identical protocol.

No claim is made here that an exported INT8 model has been validated or benchmarked on target hardware.

## Evaluation protocol

The current unified evaluation uses:

| Setting | Configuration |
|---|---|
| Test set | 100 held-out test samples |
| Sample rate | 16,000 Hz |
| Window size | 256 samples (16 ms) |
| Hop size | 128 samples (8 ms) |
| Target SNRs | −5, 0, 5, 10, 15 dB |
| Models in unified comparison | MLP, T-MLP, GRU |
| Speech-intelligibility metric | STOI available |
| Latency modes | Batch-1 eager and `tf.function` graph mode |

The same test set is used for the three models in the current unified comparison. Results are reported as means over the evaluation set, with per-SNR results reported separately below.

The latency benchmark used 20 warm-up runs and 300 timed repetitions per mode. Graph tracing was excluded from the timed graph-mode repetitions. These measurements describe the current host TensorFlow/Keras execution path and are not measurements from an earbud, microcontroller, DSP, or other target edge device.

## Evaluation metrics

### Waveform reconstruction

- **MSE (Mean Squared Error):** Mean squared difference between the predicted and clean waveform samples.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed on the waveform amplitude scale.
- **MAE (Mean Absolute Error):** Mean absolute difference between predicted and clean waveform samples.

Lower MSE, RMSE, and MAE indicate lower sample-wise reconstruction error, but they do not fully characterize perceived speech quality.

### Speech quality and intelligibility

- **SI-SDR:** Scale-Invariant Signal-to-Distortion Ratio. Higher values generally indicate better scale-invariant reconstruction.
- **SI-SDR improvement:** The change in SI-SDR from the noisy input to the enhanced output. Positive values indicate improvement relative to the noisy baseline; negative values indicate degradation under this metric.
- **STOI:** Short-Time Objective Intelligibility. Higher values generally indicate better estimated speech intelligibility.

The unified results report SI-SDR improvement and STOI. The README does not report a separate SDR result because one is not included in the current results table.

## Current results

The following results are from the current unified evaluation outputs.

| Metric | MLP | T-MLP | GRU |
|---|---:|---:|---:|
| Parameters | 328,704 | **19,137** | 37,889 |
| Checkpoint size (MB) | 3.786 | **0.286** | 0.467 |
| Mean MSE | 0.001460 | 0.001473 | **0.000905** |
| Mean RMSE | 0.03400 | 0.03371 | **0.02536** |
| Mean MAE | 0.019815 | 0.021969 | **0.015349** |
| Mean SI-SDR (dB) | 2.137 | 5.118 | **6.527** |
| Mean SI-SDR improvement (dB) | −2.856 | +0.125 | **+1.534** |
| Mean STOI | 0.6764 | **0.8433** | 0.8295 |
| Graph-mode median latency (ms) | 1.096 | **0.979** | 24.226 |
| Graph-mode p95 latency (ms) | 2.405 | **1.923** | 35.593 |

**Checkpoint size** is the size reported for the saved model file in this evaluation. It is not a measurement of runtime RAM use or the size of a fully optimized deployment binary.

### Results by input SNR

Each SNR group contains 20 test samples.

| Target SNR | MLP SI-SDR improvement (dB) | T-MLP SI-SDR improvement (dB) | GRU SI-SDR improvement (dB) | MLP STOI | T-MLP STOI | GRU STOI |
|---:|---:|---:|---:|---:|---:|---:|
| −5 dB | +0.682 | +1.290 | **+4.303** | 0.554 | 0.670 | **0.683** |
| 0 dB | +1.437 | +0.589 | **+4.287** | 0.669 | **0.842** | 0.820 |
| +5 dB | −1.211 | −0.197 | **+1.837** | 0.711 | **0.882** | 0.858 |
| +10 dB | −6.230 | −0.409 | **−0.339** | 0.722 | **0.890** | 0.874 |
| +15 dB | −8.958 | −0.649 | **−2.417** | 0.727 | **0.932** | 0.913 |

The SI-SDR improvement values show that enhancement behavior depends on input SNR. In this run, all three models have negative mean SI-SDR improvement at +10 dB and +15 dB. The GRU is less negative than the other models at +10 dB, but its mean SI-SDR improvement is also negative at +15 dB.

The STOI and SI-SDR metrics do not rank the models identically. This is one reason to avoid selecting a model from a single metric.

## Interpretation

The current results suggest a trade-off rather than one universally best model.

- **T-MLP is the smallest model** in this comparison, with 19,137 parameters and a 0.286 MB checkpoint. It has the highest mean STOI in this run.
- **GRU has the strongest mean distortion/error results** in this evaluation: the lowest mean MSE, RMSE, and MAE, and the highest mean SI-SDR and SI-SDR improvement. Its measured graph-mode latency is much higher than that of MLP and T-MLP.
- **MLP is the largest checkpoint** and has negative mean SI-SDR improvement in this run, despite a lower mean MSE than T-MLP by a small margin.
- **Performance varies with SNR.** All three models show negative mean SI-SDR improvement in the two highest-SNR groups, so the current results do not support a claim of consistent improvement in every condition.

T-MLP's compactness and STOI result make it worth further investigation for constrained inference. However, this experiment alone does not prove that it will outperform the GRU in perceived quality, energy use, or deployment performance.

## Efficiency and latency notes

### Model size

Relative to the evaluated MLP checkpoint, T-MLP has approximately:

- **94.2% fewer parameters**
- **92.4% smaller checkpoint size**, based on the reported file sizes

These are model-file and parameter-count comparisons. They do not automatically translate to the same reductions in runtime memory, power, or total device cost.

### Host-side latency

The graph-mode latency results are:

| Model | Median (ms) | 95th percentile (ms) |
|---|---:|---:|
| MLP | 1.096 | 2.405 |
| T-MLP | **0.979** | **1.923** |
| GRU | 24.226 | 35.593 |

The input window spans 16 ms and the hop spans 8 ms. The GRU's measured graph-mode median latency exceeds both values, while the MLP and T-MLP measurements are below them in this particular benchmark.

This comparison is **not a real-time guarantee**. The measurements are host-side TensorFlow/Keras timings; actual end-to-end streaming performance depends on preprocessing, buffering, overlap handling, framework/runtime overhead, target hardware, and scheduling. No embedded-device latency, power consumption, or runtime-memory benchmark has been completed.

## Known issues and limitations

### 1. Limited number of distinct test mixtures

The held-out test set contains 100 mixtures, with 20 mixtures per target SNR. The large number of overlapping windows does not increase the number of independent source recordings or mixture configurations to the same degree. Results should be treated as preliminary.

### 2. Waveform amplitude range

The current mixing code does not clip or normalize the sum after adding scaled noise. Previous inspection found waveform values outside `[-1, 1]`. This can affect interpretation of waveform-based metrics and may matter for deployment formats with constrained amplitude ranges. The results reported here are from the existing data and checkpoints; this README does not claim the data were regenerated or corrected.

### 3. Degradation at higher input SNR

Mean SI-SDR improvement is negative for all three models at +10 dB and +15 dB. The current models may alter already relatively clean speech in ways that worsen SI-SDR. This behavior should be retained in the analysis rather than hidden by reporting only aggregate metrics.

### 4. Metric trade-offs

The model with the best MSE/MAE or SI-SDR is not necessarily the model with the highest STOI or the best perceived audio. Listening tests and additional carefully controlled evaluations would provide useful complementary evidence.

### 5. QAT is a separate preliminary experiment

The existing QAT architecture differs from the FP32 T-MLP, and its earlier run showed instability. QAT metrics are not included in the current unified model comparison. Quantized deployment remains future work.

### 6. No target-device validation

No validated INT8 deployment, target-device latency, power, or runtime-memory result is reported. The project is motivated by earbud-class edge devices, but production readiness has not been demonstrated.

### 7. Reproducibility and environment

The recorded evaluation environment includes TensorFlow 2.21.0, Keras 3.15.1, and Windows 10. Results can vary with software versions, hardware, execution mode, and saved checkpoint versions.

## Reproducibility

The project uses the following core configuration:

```text
Sample rate:       16,000 Hz
Channels:          Mono
Window size:       256 samples (16 ms)
Hop size:          128 samples (8 ms)
Target SNRs:       -5, 0, +5, +10, +15 dB
Train mixtures:    800
Validation:        100
Test mixtures:     100
Random seed:       42
```

Core preprocessing dependencies include:

- Python
- NumPy
- SoundFile
- SciPy
- TensorFlow / Keras
- A STOI implementation for intelligibility evaluation

Install the packages specified by the repository's `requirements.txt` if available. The exact command and file paths may need to be adjusted to match the checked-out repository layout.

The evaluation script used for the unified run is `evaluation/evaluation_all.py`. From the repository root, the intended invocation is:

```bash
python evaluation/evaluation_all.py
```

The script expects the existing test data and model checkpoints to be available at the paths configured in the script. Do not assume that running evaluation will recreate missing data or checkpoints.

### Expected evaluation artifacts

The current evaluation produces these result files:

```text
evaluation/results/
├── evaluation_results.json
├── model_summary.csv
├── per_snr_results.csv
└── utterance_results.csv
```

These artifacts contain aggregate metrics, model summaries, SNR-specific results, and per-utterance results. Preserve the result files alongside the checkpoints if you want the published tables to remain traceable to a specific evaluation run.

### Checkpoints used in the unified comparison

```text
saved_models/
├── MLP_best.keras
├── TMLP_best.keras
└── GRU_best.keras
```

A separate `TMLP_QAT_best.keras` checkpoint has been mentioned in the project, but it is not part of the current unified evaluation table. Before publishing, verify that the exact checkpoint files in the repository are the same files used to produce the reported evaluation artifacts.

## Expected repository structure

The layout below is a reference structure; actual folder names can differ in the working repository.

```text
T-MLP-for-Environmental-Noise-Cancellation/
├── data/
│   ├── speech/
│   └── noise/
├── processed_data/
│   ├── train/
│   ├── validation/
│   └── test/
├── data_pipeline/
│   ├── data_preprocessing.py
│   ├── inspect_data.py
│   ├── mixing.py
│   └── dataset.py
├── models/
│   ├── mlp.py
│   ├── t_mlp.py
│   ├── GRU.py
│   └── tmlp_qat.py
├── saved_models/
│   ├── MLP_best.keras
│   ├── TMLP_best.keras
│   ├── GRU_best.keras
│   └── TMLP_QAT_best.keras
├── evaluation/
│   ├── evaluation_all.py
│   └── results/
├── notebooks/
├── requirements.txt
└── README.md
```

Large audio datasets and model binaries can make a Git repository difficult to clone. Consider keeping raw datasets out of normal Git history and using Git LFS or an appropriate artifact store for model checkpoints if needed. Ensure any shared data files are distributed in accordance with their dataset terms.

## Roadmap

### Completed or implemented

- [x] Inspect the speech and noise datasets
- [x] Implement mono conversion and resampling to 16 kHz
- [x] Generate controlled noisy-speech mixtures
- [x] Implement and train the MLP baseline
- [x] Implement and train the T-MLP
- [x] Implement and train the GRU baseline
- [x] Develop an initial QAT experiment
- [x] Run a unified evaluation for MLP, T-MLP, and GRU
- [x] Report MSE, RMSE, MAE, SI-SDR, SI-SDR improvement, and STOI
- [x] Report results by input SNR
- [x] Record model size and host-side latency

### Remaining work

- [ ] Investigate the waveform amplitude-range issue and document a controlled follow-up evaluation
- [ ] Analyze the negative SI-SDR improvements at higher input SNRs
- [ ] Add qualitative listening tests or other complementary speech-quality evaluation
- [ ] Align the FP32 and QAT architectures for a controlled quantization comparison
- [ ] Validate quantized model quality after export
- [ ] Investigate TensorFlow Lite or another suitable deployment runtime
- [ ] Benchmark on representative target hardware
- [ ] Measure runtime memory and power where hardware permits
- [ ] Verify that published checkpoints and result files correspond to the same run

The existing results and checkpoints should be preserved. Follow-up experiments should be versioned separately so the currently reported results remain reproducible.

## Research direction

The central question is:

> **Can a lightweight Temporal-MLP provide useful speech-enhancement performance with substantially lower model complexity, while maintaining acceptable quality and latency for resource-constrained audio systems?**

The current results provide an initial comparison, not a definitive answer. T-MLP offers a strong parameter-count and checkpoint-size advantage and the highest mean STOI in this run; GRU performs better on several reconstruction and SI-SDR measures but has much higher host-side graph latency. Further controlled evaluation and target-device measurements are needed to determine which trade-off is most appropriate for a real application.

## Datasets and acknowledgements

This project uses or builds upon:

- **LibriSpeech** for clean speech
- **ESC-50** for environmental noise
- **TensorFlow / Keras** for model development
- **TensorFlow Model Optimization** for the initial QAT experiment

Please consult the original dataset documentation and terms before redistributing data. This repository's use of dataset names does not imply endorsement by the dataset creators.

## License

No project license is specified in this README. Add a repository license file if you intend to grant others permission to use, modify, or distribute the code. Dataset licensing and redistribution terms must be considered separately from the code license.
