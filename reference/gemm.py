"""CPU reference implementation for matrix multiply."""

import numpy as np


def gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
