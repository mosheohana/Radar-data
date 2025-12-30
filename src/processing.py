"""
Task 3.1 — NPZ storage optimization
- Inspect fields
- Shrink data types (lossless for discrete fields; optional lossy for signal)
- Save with or without ZIP compression

Usage examples (run from project root where `src/` is located):
  python -m src.processing --src data/radar_data.npz --dst data/out.npz --op both
  python -m src.processing --src data/radar_data.npz --dst data/dtype_only.npz --op dtype
  python -m src.processing --src data/radar_data.npz --dst data/zip_only.npz   --op zip
  # Optional: reduce signal from float64 -> float32 (slight precision loss)
  python -m src.processing --src data/radar_data.npz --dst data/out_lossy.npz --op both --lossy-signal
"""

from typing import Dict, Any, Tuple
import os
import json
import numpy as np

from src.io import load_npz_generic
from src.utils import human_size, bytes_to_kb


# -------- helpers --------
def _minimal_int_dtype(vmin: int, vmax: int, signed: bool) -> np.dtype:
    """
    Pick the smallest integer dtype that can hold [vmin, vmax].
    """
    if signed:
        candidates = [np.int8, np.int16, np.int32, np.int64]
    else:
        candidates = [np.uint8, np.uint16, np.uint32, np.uint64]
    for dt in candidates:
        info = np.iinfo(dt)
        if vmin >= info.min and vmax <= info.max:
            return np.dtype(dt)
    return np.dtype(np.int64 if signed else np.uint64)


def inspect_npz(path: str) -> Dict[str, Any]:
    """
    Return field metadata and file size.
    """
    arrays, meta = load_npz_generic(path)
    size = os.path.getsize(path) if os.path.exists(path) else None
    return {"fields": meta, "__size_bytes__": size}


def optimize_arrays_lossless(
    arrays: Dict[str, np.ndarray],
    lossy_signal: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Shrink arrays in a safe way:
    - If a float array is actually integer-like (e.g., 0/1 or 0..4), cast to int and then downcast to the minimal width.
      If values are only 0/1, you can store as bool.
    - True continuous float arrays stay float64 by default.
      If lossy_signal=True, continuous arrays will be stored as float32 (optional, slight precision loss).
    """
    out: Dict[str, np.ndarray] = {}
    plan: Dict[str, Any] = {}

    for key, arr in arrays.items():
        action = "keep"
        new_arr = arr
        new_dtype = str(arr.dtype)

        if np.issubdtype(arr.dtype, np.integer):
            # Downcast integer arrays to minimal width
            vmin, vmax = int(arr.min()), int(arr.max())
            signed = np.issubdtype(arr.dtype, np.signedinteger)
            target = _minimal_int_dtype(vmin, vmax, signed)
            if target != arr.dtype:
                new_arr = arr.astype(target, copy=False)
                new_dtype = str(target)
                action = "downcast_int"

        elif np.issubdtype(arr.dtype, np.floating):
            # If floats are integer-like (e.g., 0/1 or 0..4), safely convert to int and then downcast.
            if np.all(np.isfinite(arr)) and np.all(np.isclose(arr, np.round(arr))):
                arr_int = arr.astype(np.int64, copy=False)
                vmin, vmax = int(arr_int.min()), int(arr_int.max())
                if vmin >= 0 and vmax <= 1:
                    # store as bool when it’s 0/1
                    new_arr = (arr_int != 0)
                    new_dtype = "bool"
                    action = "float_to_bool"
                else:
                    signed = vmin < 0
                    target = _minimal_int_dtype(vmin, vmax, signed)
                    new_arr = arr_int.astype(target, copy=False)
                    new_dtype = str(target)
                    action = "float_to_int_downcast"
            else:
                # Continuous signal: keep float64 unless lossy_signal=True
                if lossy_signal:
                    new_arr = arr.astype(np.float32, copy=False)
                    new_dtype = "float32"
                    action = "float64_to_float32"

        # Other dtypes (strings/objects) are kept as-is.
        out[key] = new_arr
        plan[key] = {
            "original_dtype": str(arr.dtype),
            "new_dtype": new_dtype,
            "action": action,
        }

    return out, plan


# -------- operation modes --------
def op_dtype_only(src_path: str, dst_path: str, lossy_signal: bool) -> Dict[str, Any]:
    """
    Dtype shrink only. We save with `np.savez` (no ZIP) so you can measure dtype benefit alone.
    """
    arrays, meta = load_npz_generic(src_path)
    orig_size = os.path.getsize(src_path)

    out, plan = optimize_arrays_lossless(arrays, lossy_signal=lossy_signal)
    np.savez(dst_path, **out)  # no internal compression
    new_size = os.path.getsize(dst_path)

    reduction_pct = (1.0 - new_size / orig_size) * 100.0
    return {
        "mode": "dtype_only",
        "original_size_bytes": orig_size,
        "optimized_size_bytes": new_size,
        "original_size_human": human_size(orig_size),
        "optimized_size_human": human_size(new_size),
        "reduction_percent": round(reduction_pct, 2),
        "reduction_kb": round(bytes_to_kb(orig_size - new_size), 1),
        "fields": meta,
        "plan": plan,
        "dst_path": dst_path,
    }


def op_zip_only(src_path: str, dst_path: str) -> Dict[str, Any]:
    """
    ZIP only. Repackage with `np.savez_compressed` without changing dtypes.
    """
    arrays, meta = load_npz_generic(src_path)
    orig_size = os.path.getsize(src_path)

    np.savez_compressed(dst_path, **arrays)
    new_size = os.path.getsize(dst_path)

    reduction_pct = (1.0 - new_size / orig_size) * 100.0
    return {
        "mode": "zip_only",
        "original_size_bytes": orig_size,
        "optimized_size_bytes": new_size,
        "original_size_human": human_size(orig_size),
        "optimized_size_human": human_size(new_size),
        "reduction_percent": round(reduction_pct, 2),
        "reduction_kb": round(bytes_to_kb(orig_size - new_size), 1),
        "fields": meta,
        "plan": {k: {"original_dtype": v["dtype"], "new_dtype": v["dtype"], "action": "keep"} for k, v in meta.items()},
        "dst_path": dst_path,
    }


def op_both(src_path: str, dst_path: str, lossy_signal: bool) -> Dict[str, Any]:
    """
    Dtype shrink first, then ZIP (recommended).
    """
    arrays, meta = load_npz_generic(src_path)
    orig_size = os.path.getsize(src_path)

    out, plan = optimize_arrays_lossless(arrays, lossy_signal=lossy_signal)
    np.savez_compressed(dst_path, **out)
    new_size = os.path.getsize(dst_path)

    reduction_pct = (1.0 - new_size / orig_size) * 100.0
    return {
        "mode": "both",
        "original_size_bytes": orig_size,
        "optimized_size_bytes": new_size,
        "original_size_human": human_size(orig_size),
        "optimized_size_human": human_size(new_size),
        "reduction_percent": round(reduction_pct, 2),
        "reduction_kb": round(bytes_to_kb(orig_size - new_size), 1),
        "fields": meta,
        "plan": plan,
        "dst_path": dst_path,
    }


# -------- CLI --------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NPZ storage optimization")
    parser.add_argument("--src", required=True, help="path to input .npz (e.g., data/radar_data.npz)")
    parser.add_argument("--dst", required=True, help="path to output .npz (e.g., data/out.npz)")
    parser.add_argument("--op", choices=["dtype", "zip", "both"], default="both",
                        help="operation mode: dtype-only, zip-only, or both (dtype then zip)")
    parser.add_argument("--lossy-signal", action="store_true",
                        help="optionally convert continuous float64 arrays to float32 (small precision loss)")
    parser.add_argument("--report", default=None, help="optional JSON report path (e.g., outputs/opt_report.json)")
    args = parser.parse_args()

    info = inspect_npz(args.src)
    print("Original file size:", human_size(info["__size_bytes__"]))
    for k, v in info["fields"].items():
        print(f"- {k}: dtype={v['dtype']} shape={v['shape']} size={v['nbytes']} bytes")

    if args.op == "dtype":
        rep = op_dtype_only(args.src, args.dst, lossy_signal=args.lossy_signal)
    elif args.op == "zip":
        rep = op_zip_only(args.src, args.dst)
    else:
        rep = op_both(args.src, args.dst, lossy_signal=args.lossy_signal)

    print("\n=== Optimization Report ===")
    print(json.dumps(rep, indent=2))

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print("Report written to:", args.report)
