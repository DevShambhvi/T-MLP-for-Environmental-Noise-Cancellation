# Unified evaluation for existing MLP, T-MLP, and GRU checkpoints.

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "processed_data" / "test"
MODEL_PATH = PROJECT_ROOT / "saved_models"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results"

SAMPLE_RATE = 16_000
WINDOW_SIZE = 256
HOP_SIZE = 128
EXPECTED_TEST_SAMPLES = 100
LATENCY_WARMUP = 20
LATENCY_REPEATS = 300
EVAL_BATCH_SIZE = 256
RANDOM_SEED = 2026

MODEL_FILES = {
    "MLP": MODEL_PATH / "MLP_best.keras",
    "T-MLP": MODEL_PATH / "TMLP_best.keras",
    "GRU": MODEL_PATH / "GRU_best.keras",
}

try:
    from pystoi import stoi as _stoi
    STOI_AVAILABLE = True
except ImportError:
    _stoi = None
    STOI_AVAILABLE = False


@keras.saving.register_keras_serializable(package="TMLP")
class TemporalMLPBlock(keras.layers.Layer):
    """Residual temporal MLP block used by the saved T-MLP model."""

    def __init__(self, hidden_dim=64, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.dense_1 = keras.layers.Dense(self.hidden_dim, activation="relu")
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.dense_2 = keras.layers.Dense(self.hidden_dim, activation=None)
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        x = self.add([inputs, x])
        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


CUSTOM_OBJECTS = {
    "TemporalMLPBlock": TemporalMLPBlock,
    "TMLP>TemporalMLPBlock": TemporalMLPBlock,
}


def ensure_directories() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def get_npz_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Test directory not found: {folder}")
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz test files found in {folder}")
    return files


def load_test_sample(file_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(file_path, allow_pickle=True) as data:
        required = {"noisy_speech", "clean_speech", "target_snr_db"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{file_path.name} is missing keys: {sorted(missing)}")
        noisy = np.asarray(data["noisy_speech"], dtype=np.float32).reshape(-1)
        clean = np.asarray(data["clean_speech"], dtype=np.float32).reshape(-1)
        snr = float(np.asarray(data["target_snr_db"]).item())

    if noisy.size != clean.size:
        raise ValueError(
            f"Length mismatch in {file_path.name}: noisy={noisy.size}, clean={clean.size}"
        )
    if noisy.size == 0:
        raise ValueError(f"Empty audio arrays in {file_path.name}")
    if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
        raise ValueError(f"NaN/Inf values found in {file_path.name}")
    return noisy, clean, snr


def create_windows(audio: np.ndarray) -> np.ndarray:
    """Create fixed-size windows using the same hop for every model."""
    if len(audio) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE), dtype=np.float32)
    starts = range(0, len(audio) - WINDOW_SIZE + 1, HOP_SIZE)
    return np.stack([audio[s:s + WINDOW_SIZE] for s in starts]).astype(np.float32)


def adapt_input_to_model(model: keras.Model, windows: np.ndarray) -> np.ndarray:
    """Adapt (N, 256) windows to a supported single-input model shape."""
    shape = model.input_shape
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(f"Only single-input models are supported; got {shape}")
        shape = shape[0]

    expected = tuple(shape[1:])
    candidates = [windows, windows[..., np.newaxis], windows[:, np.newaxis, :]]
    for candidate in candidates:
        if candidate.ndim != len(expected) + 1:
            continue
        actual = tuple(candidate.shape[1:])
        if all(exp is None or exp == got for exp, got in zip(expected, actual)):
            return candidate.astype(np.float32, copy=False)

    raise ValueError(
        f"Cannot adapt windows {windows.shape} to model input shape {model.input_shape}. "
        "Inspect the checkpoint architecture before evaluating it."
    )


def normalize_prediction_shape(prediction: Any) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 3 and prediction.shape[-1] == 1:
        prediction = prediction[..., 0]
    elif prediction.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction[:, 0, :]
    if prediction.ndim != 2 or prediction.shape[1] != WINDOW_SIZE:
        raise ValueError(
            f"Expected output shaped (N, {WINDOW_SIZE}) or (N, {WINDOW_SIZE}, 1); "
            f"got {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Model output contains NaN or Inf values")
    return prediction


def load_model(model_name: str, model_path: Path) -> keras.Model:
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    custom_objects = CUSTOM_OBJECTS if model_name == "T-MLP" else None
    model = keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False
    )
    print(f"\nLoaded {model_name}: {model_path.name}")
    print(f"  input shape : {model.input_shape}")
    print(f"  output shape: {model.output_shape}")
    print(f"  parameters  : {model.count_params():,}")
    return model


def reconstruct_audio(predicted_windows: np.ndarray, original_length: int) -> np.ndarray:
    """Rectangular-window overlap-add; averages predictions for covered samples."""
    if len(predicted_windows) == 0:
        return np.empty(0, dtype=np.float32)

    covered_length = min(
        original_length, (len(predicted_windows) - 1) * HOP_SIZE + WINDOW_SIZE
    )
    output = np.zeros(covered_length, dtype=np.float64)
    weights = np.zeros(covered_length, dtype=np.float64)

    for i, window in enumerate(predicted_windows):
        start = i * HOP_SIZE
        end = min(start + WINDOW_SIZE, covered_length)
        count = end - start
        output[start:end] += window[:count]
        weights[start:end] += 1.0

    valid = weights > 0
    output[valid] /= weights[valid]
    return output.astype(np.float32)


def make_inference_call(model: keras.Model, graph_mode: bool):
    """Create a model inference callable; graph tracing occurs on first call."""
    if graph_mode:
        @tf.function(reduce_retracing=True)
        def graph_call(inputs):
            return model(inputs, training=False)
        return graph_call
    return lambda inputs: model(inputs, training=False)


def enhance_utterance(
    model: keras.Model, noisy: np.ndarray, inference_call=None
) -> tuple[np.ndarray, int]:
    windows = create_windows(noisy)
    if len(windows) == 0:
        return noisy.copy(), 0

    model_input = adapt_input_to_model(model, windows)
    call = inference_call or make_inference_call(model, graph_mode=True)
    predictions = []

    for start in range(0, len(model_input), EVAL_BATCH_SIZE):
        batch = tf.convert_to_tensor(model_input[start:start + EVAL_BATCH_SIZE])
        output = call(batch)
        if isinstance(output, (list, tuple)):
            raise ValueError("Multi-output models are not supported by this evaluator.")
        predictions.append(normalize_prediction_shape(output.numpy()))

    predictions_array = np.concatenate(predictions, axis=0)
    enhanced_covered = reconstruct_audio(predictions_array, len(noisy))
    enhanced = noisy.copy()
    enhanced[:len(enhanced_covered)] = enhanced_covered
    return enhanced.astype(np.float32), len(windows)


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.square(reference - estimate)))


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(mse(reference, estimate)))


def mae(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - estimate)))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-distortion ratio in dB; higher is better."""
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    length = min(reference.size, estimate.size)
    if length == 0:
        return float("nan")
    reference = reference[:length] - np.mean(reference[:length])
    estimate = estimate[:length] - np.mean(estimate[:length])
    ref_energy = np.sum(reference ** 2)
    if ref_energy < eps:
        return float("nan")
    scale = np.sum(estimate * reference) / (ref_energy + eps)
    target = scale * reference
    residual = estimate - target
    return float(10.0 * np.log10(
        (np.sum(target ** 2) + eps) / (np.sum(residual ** 2) + eps)
    ))


def stoi_score(reference: np.ndarray, estimate: np.ndarray) -> float:
    if not STOI_AVAILABLE:
        return float("nan")
    length = min(len(reference), len(estimate))
    if length < SAMPLE_RATE:
        return float("nan")
    try:
        return float(_stoi(
            np.asarray(reference[:length], dtype=np.float32),
            np.asarray(estimate[:length], dtype=np.float32),
            SAMPLE_RATE,
            extended=False,
        ))
    except Exception:
        return float("nan")


def evaluate_pair(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray
) -> dict[str, Any]:
    length = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:length], noisy[:length], enhanced[:length]

    noisy_si_sdr = si_sdr(clean, noisy)
    enhanced_si_sdr = si_sdr(clean, enhanced)
    noisy_mse = mse(clean, noisy)
    enhanced_mse = mse(clean, enhanced)
    noisy_mae = mae(clean, noisy)
    enhanced_mae = mae(clean, enhanced)

    return {
        "mse": enhanced_mse,
        "rmse": rmse(clean, enhanced),
        "mae": enhanced_mae,
        "si_sdr_db": enhanced_si_sdr,
        "si_sdr_improvement_db": (
            enhanced_si_sdr - noisy_si_sdr
            if np.isfinite(enhanced_si_sdr) and np.isfinite(noisy_si_sdr)
            else float("nan")
        ),
        "stoi": stoi_score(clean, enhanced),
        "noisy_mse": noisy_mse,
        "noisy_rmse": rmse(clean, noisy),
        "noisy_mae": noisy_mae,
        "noisy_si_sdr_db": noisy_si_sdr,
        "noisy_stoi": stoi_score(clean, noisy),
        "mse_reduction_pct": (
            100.0 * (noisy_mse - enhanced_mse) / noisy_mse
            if noisy_mse > 0 else float("nan")
        ),
        "mae_reduction_pct": (
            100.0 * (noisy_mae - enhanced_mae) / noisy_mae
            if noisy_mae > 0 else float("nan")
        ),
        "samples_evaluated": int(length),
    }


METRIC_KEYS = [
    "mse", "rmse", "mae", "si_sdr_db", "si_sdr_improvement_db", "stoi",
    "noisy_mse", "noisy_rmse", "noisy_mae", "noisy_si_sdr_db", "noisy_stoi",
    "mse_reduction_pct", "mae_reduction_pct",
]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = np.asarray(
            [record[key] for record in records if np.isfinite(record[key])],
            dtype=np.float64,
        )
        result[key] = float(np.mean(values)) if values.size else float("nan")
        result[f"{key}_std"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        )
        result[f"{key}_valid_utterances"] = int(values.size)

    result["utterances"] = len(records)
    result["samples_evaluated"] = int(sum(r["samples_evaluated"] for r in records))
    result["skipped_short_utterances"] = int(
        sum(r.get("window_count", 0) == 0 for r in records)
    )
    return result


def evaluate_model(model_name: str, model: keras.Model, test_files: list[Path]):
    records: list[dict[str, Any]] = []
    inference_call = make_inference_call(model, graph_mode=True)
    print(f"\nEvaluating {model_name} on {len(test_files)} test utterances...")

    for index, file_path in enumerate(test_files, start=1):
        noisy, clean, snr = load_test_sample(file_path)
        enhanced, window_count = enhance_utterance(model, noisy, inference_call)
        metrics = evaluate_pair(clean, noisy, enhanced)
        metrics.update({
            "model": model_name,
            "file": file_path.name,
            "snr_db": snr,
            "window_count": window_count,
        })
        records.append(metrics)
        if index % 10 == 0 or index == len(test_files):
            print(f"  Processed {index}/{len(test_files)}")

    overall = aggregate_records(records)
    per_snr = {}
    for snr in sorted(set(round(float(r["snr_db"]), 4) for r in records)):
        selected = [r for r in records if np.isclose(r["snr_db"], snr)]
        per_snr[str(snr)] = aggregate_records(selected)
    return records, overall, per_snr


def _benchmark_mode(call, inputs, warmup: int, repeats: int) -> np.ndarray:
    """Time batch-1 calls and materialize outputs before stopping each timer."""
    n = len(inputs)
    for i in range(warmup):
        output = call(tf.convert_to_tensor(inputs[i % n:i % n + 1]))
        _ = output.numpy()

    timings = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        sample = tf.convert_to_tensor(inputs[i % n:i % n + 1])
        start = time.perf_counter()
        output = call(sample)
        _ = output.numpy()
        timings[i] = (time.perf_counter() - start) * 1000.0
    return timings


def _latency_stats(latencies: np.ndarray, prefix: str) -> dict[str, Any]:
    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    window_ms = 1000.0 * WINDOW_SIZE / SAMPLE_RATE
    hop_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE
    return {
        f"{prefix}_mean_ms": mean_ms,
        f"{prefix}_median_ms": median_ms,
        f"{prefix}_p95_ms": p95_ms,
        f"{prefix}_p99_ms": p99_ms,
        f"{prefix}_min_ms": float(np.min(latencies)),
        f"{prefix}_max_ms": float(np.max(latencies)),
        f"{prefix}_throughput_windows_per_sec": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        f"{prefix}_rtf_vs_window_duration": mean_ms / window_ms if window_ms > 0 else float("nan"),
        f"{prefix}_latency_vs_hop_deadline": mean_ms / hop_ms if hop_ms > 0 else float("nan"),
        f"{prefix}_p95_vs_hop_deadline": p95_ms / hop_ms if hop_ms > 0 else float("nan"),
    }


def benchmark_latency(
    model_name: str, model: keras.Model, test_files: list[Path]
) -> dict[str, Any]:
    """
    Benchmark eager and graph inference on identical real test windows.
    Graph tracing is explicitly done before timing. Includes host-side Python
    dispatch and output materialization; excludes audio I/O/windowing.
    """
    candidate_windows = []
    for file_path in test_files:
        noisy, _, _ = load_test_sample(file_path)
        windows = create_windows(noisy)
        if len(windows):
            candidate_windows.append(windows[0])
        if len(candidate_windows) >= min(50, len(test_files)):
            break
    if not candidate_windows:
        raise RuntimeError("No valid windows available for latency benchmark.")

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(
        len(candidate_windows),
        size=LATENCY_REPEATS,
        replace=(len(candidate_windows) < LATENCY_REPEATS),
    )
    windows = np.asarray([candidate_windows[i] for i in selected], dtype=np.float32)
    inputs = adapt_input_to_model(model, windows)

    eager_call = lambda x: model(x, training=False)

    @tf.function(reduce_retracing=True)
    def graph_call(x):
        return model(x, training=False)

    # Exclude graph tracing and initial execution from timed repetitions.
    _ = graph_call(tf.convert_to_tensor(inputs[:1])).numpy()

    eager_ms = _benchmark_mode(eager_call, inputs, LATENCY_WARMUP, LATENCY_REPEATS)
    graph_ms = _benchmark_mode(graph_call, inputs, LATENCY_WARMUP, LATENCY_REPEATS)

    result = {}
    result.update(_latency_stats(eager_ms, "eager_latency"))
    result.update(_latency_stats(graph_ms, "graph_latency"))
    result["graph_vs_eager_median_speedup"] = (
        float(np.median(eager_ms) / np.median(graph_ms))
        if np.median(graph_ms) > 0 else float("nan")
    )
    result["latency_repeats"] = LATENCY_REPEATS
    result["latency_warmup"] = LATENCY_WARMUP
    result["latency_method"] = (
        "Host-side batch-1 model call plus output materialization; eager and "
        "tf.function graph modes; graph tracing excluded; not embedded-device latency"
    )
    result["window_duration_ms"] = 1000.0 * WINDOW_SIZE / SAMPLE_RATE
    result["hop_duration_ms"] = 1000.0 * HOP_SIZE / SAMPLE_RATE
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, allow_nan=False)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"  Not written (no rows): {path.name}")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: ("" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                for k, v in row.items()
            })


def main() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("FINAL EVALUATION — EXISTING MLP, T-MLP, AND GRU CHECKPOINTS")
    print("=" * 88)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Test directory: {TEST_PATH}")
    print(f"STOI: {'available' if STOI_AVAILABLE else 'not installed; STOI values will be blank'}")
    print(f"Window/hop: {WINDOW_SIZE}/{HOP_SIZE} samples at {SAMPLE_RATE} Hz")
    print(f"Latency repeats/warm-up per mode: {LATENCY_REPEATS}/{LATENCY_WARMUP}")

    test_files = get_npz_files(TEST_PATH)
    if len(test_files) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test NPZ files, found {len(test_files)}. "
            "If test-set size intentionally changed, update EXPECTED_TEST_SAMPLES."
        )

    test_metadata = []
    for file_path in test_files:
        _, _, snr = load_test_sample(file_path)
        test_metadata.append((file_path.name, snr))

    print("\nTest-set SNR distribution:")
    for snr in sorted(set(round(s, 4) for _, s in test_metadata)):
        count = sum(np.isclose(item_snr, snr) for _, item_snr in test_metadata)
        print(f"  {snr:>5g} dB: {count} utterances")

    all_results: dict[str, Any] = {
        "metadata": {
            "tensorflow_version": tf.__version__,
            "keras_version": keras.__version__,
            "platform": platform.platform(),
            "test_directory": str(TEST_PATH),
            "test_utterances": len(test_files),
            "sample_rate_hz": SAMPLE_RATE,
            "window_size_samples": WINDOW_SIZE,
            "hop_size_samples": HOP_SIZE,
            "window_duration_ms": 1000.0 * WINDOW_SIZE / SAMPLE_RATE,
            "hop_duration_ms": 1000.0 * HOP_SIZE / SAMPLE_RATE,
            "stoi_available": STOI_AVAILABLE,
            "latency_note": (
                "Host-side batch-1 eager and tf.function graph timing; not embedded-device latency."
            ),
        },
        "models": {},
    }

    summary_rows = []
    per_snr_rows = []
    utterance_rows = []

    for model_name, model_path in MODEL_FILES.items():
        model = None
        try:
            model = load_model(model_name, model_path)
            records, overall, per_snr = evaluate_model(model_name, model, test_files)
            latency = benchmark_latency(model_name, model, test_files)
            summary = {
                "model": model_name,
                "status": "success",
                "parameters_total": int(model.count_params()),
                "checkpoint_size_bytes": int(model_path.stat().st_size),
                "checkpoint_size_mb": model_path.stat().st_size / (1024.0 ** 2),
                "checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
                **overall,
                **latency,
            }
            summary_rows.append(summary)
            all_results["models"][model_name] = {
                "status": "success", "summary": summary, "per_snr": per_snr
            }
            for snr, metrics in per_snr.items():
                per_snr_rows.append({"model": model_name, "snr_db": float(snr), **metrics})
            utterance_rows.extend(records)

            print(
                f"\n{model_name}: MSE={summary['mse']:.7g}, "
                f"MAE={summary['mae']:.7g}, "
                f"SI-SDR improvement={summary['si_sdr_improvement_db']:.3f} dB, "
                f"params={summary['parameters_total']:,}, "
                f"checkpoint={summary['checkpoint_size_mb']:.3f} MB"
            )
            print(
                f"  Eager mean/median/p95: "
                f"{summary['eager_latency_mean_ms']:.3f}/"
                f"{summary['eager_latency_median_ms']:.3f}/"
                f"{summary['eager_latency_p95_ms']:.3f} ms"
            )
            print(
                f"  Graph mean/median/p95: "
                f"{summary['graph_latency_mean_ms']:.3f}/"
                f"{summary['graph_latency_median_ms']:.3f}/"
                f"{summary['graph_latency_p95_ms']:.3f} ms"
            )
        except Exception as exc:
            print(f"\nERROR evaluating {model_name}: {type(exc).__name__}: {exc}")
            all_results["models"][model_name] = {
                "status": "failed", "error_type": type(exc).__name__,
                "error": str(exc), "checkpoint": str(model_path),
            }
            summary_rows.append({
                "model": model_name, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc),
                "checkpoint": str(model_path),
            })
        finally:
            if model is not None:
                del model
            keras.backend.clear_session()
            tf.keras.backend.clear_session()

    save_json(all_results, RESULTS_PATH / "evaluation_results.json")
    save_csv(summary_rows, RESULTS_PATH / "model_summary.csv")
    save_csv(per_snr_rows, RESULTS_PATH / "per_snr_results.csv")
    save_csv(utterance_rows, RESULTS_PATH / "utterance_results.csv")

    print("\n" + "=" * 156)
    print("FINAL COMPARISON — quality metrics are macro-averages across utterances")
    print("=" * 156)
    header = (
        f"{'Model':<10}{'Status':<9}{'Params':>12}{'Size MB':>10}"
        f"{'MSE':>13}{'MAE':>12}{'SI-SDR Δ':>12}{'STOI':>9}"
        f"{'Eager med':>12}{'Graph med':>12}{'Graph p95':>12}{'Graph RTF':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        if row.get("status") != "success":
            print(f"{row['model']:<10}{'FAILED':<9} -- {row.get('error', '')}")
            continue
        print(
            f"{row['model']:<10}{'OK':<9}"
            f"{row['parameters_total']:>12,}"
            f"{row['checkpoint_size_mb']:>10.3f}"
            f"{row['mse']:>13.6g}"
            f"{row['mae']:>12.6g}"
            f"{row['si_sdr_improvement_db']:>12.3f}"
            f"{row['stoi']:>9.4f}"
            f"{row['eager_latency_median_ms']:>12.3f}"
            f"{row['graph_latency_median_ms']:>12.3f}"
            f"{row['graph_latency_p95_ms']:>12.3f}"
            f"{row['graph_latency_rtf_vs_window_duration']:>11.4f}"
        )

    print(f"\nResults directory: {RESULTS_PATH}")
    print("  model_summary.csv")
    print("  per_snr_results.csv")
    print("  utterance_results.csv")
    print("  evaluation_results.json")
    print("\nInterpretation notes:")
    print("- Lower MSE/MAE is better; higher SI-SDR improvement/STOI is better.")
    print("- Graph timing excludes graph tracing, but remains host-side timing.")
    print("- Graph RTF < 1 means mean graph inference latency is below 16 ms window duration.")
    print("- Graph latency_vs_hop_deadline < 1 means mean graph latency is below 8 ms hop interval.")
    print("- .keras file size is not the same as peak runtime RAM.")
    print("- Compare latency only on the same hardware, runtime, and software environment.")


if __name__ == "__main__":
    main()
