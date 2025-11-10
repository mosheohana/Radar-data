# src/optimization.py
import numpy as np
import time
from typing import Callable, Tuple
from src.utils import chunk_nonoverlap, timeit_avg

def RMS_computation_optimized(signal: np.ndarray, sampling_rate: int = 500, t_len_sec: int = 5) -> np.ndarray:
    """
    Optimized RMS computation using NumPy vectorization.
    No Python loops inside — everything done in bulk.
    """
    segment_length = sampling_rate * t_len_sec
    n_segments = len(signal) // segment_length

    # Reshape into 2D: each row is one 5-second segment
    trimmed = signal[: n_segments * segment_length]
    segments = trimmed.reshape(n_segments, segment_length)

    # Compute RMS for each segment using NumPy operations
    rms = np.sqrt(np.mean(segments ** 2, axis=1))
    return rms

def benchmark_rms(
    baseline_fn: Callable[[np.ndarray], np.ndarray],
    optimized_fn: Callable[[np.ndarray], np.ndarray],
    signal: np.ndarray,
    repeats: int = 3
) -> Tuple[float, float, float]:
    """
    Compare runtime between baseline and optimized RMS functions.
    Returns (baseline_ms, optimized_ms, speedup_factor)
    """
    baseline_times = []
    optimized_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        baseline_fn(signal)
        baseline_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        optimized_fn(signal)
        optimized_times.append((time.perf_counter() - t0) * 1000)

    base_avg = np.mean(baseline_times)
    opt_avg = np.mean(optimized_times)
    speedup = base_avg / opt_avg if opt_avg > 0 else float("inf")

    print(f"Baseline avg: {base_avg:.2f} ms | Optimized avg: {opt_avg:.2f} ms | Speed-up: {speedup:.2f}x")
    return base_avg, opt_avg, speedup



'''just for my own checks'''

if __name__ == "__main__":
    import numpy as np
    from src.RMS_computation import RMS_computation as RMS_baseline  # שנה לשם הקובץ שלך

    # Load your signal
    with np.load(r"C:\Users\moshe\PycharmProjects\pythonProject\imlprojects\homeproject\data\radar_data_optimized.npz",
                 allow_pickle=False) as f:
        signal = f["signal"]

    from src.optimization import RMS_computation_optimized, benchmark_rms
    benchmark_rms(RMS_baseline, RMS_computation_optimized, signal, repeats=3)
