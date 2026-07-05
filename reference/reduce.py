"""CPU reference implementation for parallel reduction."""

import numpy as np


def reduce(data: np.ndarray, op: str = "sum") -> float:
    ops = {
        "sum": np.sum,
        "min": np.min,
        "max": np.max,
    }
    if op not in ops:
        raise ValueError(f"Unsupported op '{op}'. Choose from: {list(ops)}")
    return float(ops[op](data))
