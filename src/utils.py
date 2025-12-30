# src/utils.py

import time
import numpy as np
from contextlib import contextmanager

def bytes_to_kb(n_bytes: int) -> float:
    return n_bytes / 1024.0

def bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)

def human_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{bytes_to_kb(n_bytes):.1f} KB"
    else:
        return f"{bytes_to_mb(n_bytes):.2f} MB"

def chunk_nonoverlap(arr: np.ndarray, chunk_size: int) -> np.ndarray:
    """Trim to a multiple of chunk_size and reshape into non-overlapping chunks."""
    n = (arr.shape[0] // chunk_size) * chunk_size
    return arr[:n].reshape(-1, chunk_size)

def timeit_avg(func, *args, repeats=5, **kwargs) -> float:
    """Return average runtime in milliseconds over `repeats` runs."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts))