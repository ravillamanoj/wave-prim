"""CPU reference implementation for parallel prefix scan."""

import numpy as np


def scan(data: np.ndarray, inclusive: bool = True) -> np.ndarray:
    result = np.cumsum(data.astype(np.float64)).astype(data.dtype)
    if not inclusive:
        result = np.roll(result, 1)
        result[0] = data.dtype.type(0)
    return result
