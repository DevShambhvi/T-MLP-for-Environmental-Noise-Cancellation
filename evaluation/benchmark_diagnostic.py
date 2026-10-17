
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_all import (
    load_model,
    adapt_input_to_model,
    WINDOW_SIZE,
)

MODEL_DIR = PROJECT_ROOT / "saved_models"

MODEL_PATHS = {
    "MLP": MODEL_DIR / "MLP_best.keras",
    "T-MLP": MODEL_DIR / "TMLP_best.keras",
    "GRU": MODEL_DIR / "GRU_best.keras",
}

rng = np.random.default_rng(42)
window = rng.normal(
    size=(1, WINDOW_SIZE)
).astype(np.float32)

WARMUP = 10
REPEATS = 30


def benchmark(call, x, warmup=WARMUP, repeats=REPEATS):
    # Warm-up is excluded from the timing results.
    for _ in range(warmup):
        output = call(x)
        _ = output.numpy()

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        output = call(x)
        _ = output.numpy()
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "mean": float(np.mean(timings)),
        "median": float(np.median(timings)),
        "min": float(np.min(timings)),
        "max": float(np.max(timings)),
    }


for name, path in MODEL_PATHS.items():
    print(f"\n--- {name} ---")

    model = load_model(name, path)
    x = adapt_input_to_model(model, window)

    # Mode 1: ordinary eager model calls.
    eager_call = lambda inputs: model(inputs, training=False)

    # Mode 2: TensorFlow graph execution.
    graph_call = tf.function(
        lambda inputs: model(inputs, training=False),
        reduce_retracing=True,
    )

    # Graph invocation traces/compiles the function.
    _ = graph_call(x).numpy()

    eager = benchmark(eager_call, x)
    graph = benchmark(graph_call, x)

    print("Input shape:", x.shape)
    print("Parameters:", model.count_params())

    for mode, result in [
        ("Eager", eager),
        ("Graph", graph),
    ]:
        print(
            f"{mode:>5} | "
            f"mean={result['mean']:.3f} ms | "
            f"median={result['median']:.3f} ms | "
            f"min={result['min']:.3f} ms | "
            f"max={result['max']:.3f} ms"
        )

    print(
        "Median speedup (eager/graph):",
        round(eager["median"] / graph["median"], 2),
        "x",
    )
