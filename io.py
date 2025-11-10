# src/io.py
from typing import Dict, Any, Tuple
import numpy as np

def load_npz_generic(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load any .npz file without assuming specific field names.

    Returns:
      - arrays: dict mapping field name -> np.ndarray
      - meta:   dict mapping field name -> {dtype, shape, nbytes}
    """
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}

    meta = {
        name: {
            "dtype": str(arr.dtype),
            "shape": tuple(arr.shape),
            "nbytes": int(arr.nbytes),
        }
        for name, arr in arrays.items()
    }

    return arrays, meta
