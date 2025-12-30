# src/parallelization.py
import os
import glob
import time
from typing import List, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from src.RMS_computation import RMS_computation
from typing import Optional


def generate_copies(src_path: str, dst_folder: str, n: int = 1000) -> List[str]:
    """
    Generate N copies of a .npz file inside `dst_folder`.
    """
    os.makedirs(dst_folder, exist_ok=True)
    with np.load(src_path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}

    paths = []
    for i in range(n):
        p = os.path.join(dst_folder, f"radar_data_{i:04d}.npz")
        # use uncompressed save to stress IO (you can switch to savez_compressed if you prefer)
        np.savez(p, **arrays)
        paths.append(p)

    print(f"✅ Created {len(paths)} copies in: {dst_folder}")
    return paths


def run_serially(files: List[str]) -> int:
    """
    Run RMS_computation() on all files one by one (serial execution).
    Returns number of files processed.
    """
    count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as f:
            key = "signal" if "signal" in f.files else list(f.files)[0]
            signal = f[key]
        _ = RMS_computation(signal)
        count += 1
    return count


# ---------- Parallel version ----------

def _process_one(path: str) -> int:
    """Worker function: load one file, compute RMS once, return 1 on success."""
    with np.load(path, allow_pickle=False) as f:
        key = "signal" if "signal" in f.files else list(f.files)[0]
        signal = f[key]
    _ = RMS_computation(signal)
    return 1

def run_multiprocess(files: List[str], workers: Optional[int] = None, chunksize: Optional[int] = None) -> int:
    """
    Run RMS_computation() on all files in parallel using multiprocessing.Pool.
    Compatible with Python 3.9.
    """
    if workers is None:
        workers = max(1, cpu_count() - 1)

    with Pool(processes=workers) as pool:
        if chunksize is None:
            done_flags = pool.map(_process_one, files)
        else:
            done_flags = pool.map(_process_one, files, chunksize)

    return int(sum(done_flags))


# ---------- Quick test block ----------
if __name__ == "__main__":
    import glob
    import argparse

    # --- Defaults based on project layout ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../homeproject
    default_src = os.path.join(PROJECT_ROOT, "data", "radar_data_optimized.npz")
    default_copies_dir = os.path.join(PROJECT_ROOT, "data", "copies")

    parser = argparse.ArgumentParser(
        description="Generate NPZ copies (if needed) and benchmark serial vs. multiprocessing."
    )
    parser.add_argument("--src", default=default_src, help="Source NPZ to copy from.")
    parser.add_argument("--copies-dir", default=default_copies_dir, help="Destination folder for copies.")
    parser.add_argument("--n", type=int, default=1000, help="Target number of copies to have.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count()-1),
                        help="Number of worker processes for multiprocessing.")
    parser.add_argument("--chunksize", type=int, default=0,
                        help="Pool.map chunksize (0 = auto).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first K files (0 = process all available).")
    args = parser.parse_args()

    # 1) Ensure we have enough copies
    os.makedirs(args.copies_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(args.copies_dir, "*.npz")))
    need = max(0, args.n - len(existing))
    if need > 0:
        print(f"Creating {need} copies to reach {args.n} total in: {args.copies_dir}")
        generate_copies(args.src, args.copies_dir, n=need)
        existing = sorted(glob.glob(os.path.join(args.copies_dir, "*.npz")))

    print(f"Found {len(existing)} files in copies/")

    # 2) Choose the working set
    if args.limit and args.limit > 0:
        files = existing[:args.limit]
        print(f"🧪 Using subset: {len(files)} files (limit={args.limit})")
    else:
        files = existing[:args.n]
        print(f"🧪 Using {len(files)} files")

    # 3) Serial benchmark
    t0 = time.time()
    n_serial = run_serially(files)
    serial_sec = time.time() - t0
    print(f"✅ Serial processed: {n_serial} files in {serial_sec:.2f}s")

    # 4) Parallel benchmark
    use_chunksize = args.chunksize if args.chunksize > 0 else max(1, len(files) // (args.workers * 4))
    t0 = time.time()
    n_parallel = run_multiprocess(files, workers=args.workers, chunksize=use_chunksize)
    parallel_sec = time.time() - t0
    print(f"✅ Parallel processed: {n_parallel} files in {parallel_sec:.2f}s")

    # 5) Summary
    speedup = (serial_sec / parallel_sec) if parallel_sec > 0 else float("inf")
    print(f"⚡ Speed-up: {speedup:.2f}x  (workers={args.workers}, chunksize={use_chunksize})")
