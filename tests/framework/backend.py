from abc import ABC, abstractmethod
import numpy as np


class Backend(ABC):
    """Abstract interface that every primitive backend must implement."""

    @abstractmethod
    def reduce(self, data: np.ndarray, op: str = "sum") -> float: ...

    @abstractmethod
    def scan(self, data: np.ndarray, inclusive: bool = True) -> np.ndarray: ...

    @abstractmethod
    def sort(self, data: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: ...


class CPUReference(Backend):
    """NumPy-based reference implementation — ground truth for all correctness tests."""

    def reduce(self, data: np.ndarray, op: str = "sum") -> float:
        ops = {
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
        }
        if op not in ops:
            raise ValueError(f"Unsupported reduce op '{op}'. Choose from: {list(ops)}")
        return float(ops[op](data))

    def scan(self, data: np.ndarray, inclusive: bool = True) -> np.ndarray:
        result = np.cumsum(data.astype(np.float64)).astype(data.dtype)
        if not inclusive:
            result = np.roll(result, 1)
            result[0] = 0
        return result

    def sort(self, data: np.ndarray) -> np.ndarray:
        return np.sort(data)

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)


class OpenCLBackend(Backend):
    """OpenCL backend — runs kernels via pyopencl (POCL on CPU, Mali GPU on hardware)."""

    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        import pyopencl as cl
        self._cl = cl
        platforms = cl.get_platforms()
        device = platforms[platform_idx].get_devices()[device_idx]
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

    def reduce(self, data: np.ndarray, op: str = "sum") -> float:
        raise NotImplementedError("Reduce kernel implemented in Phase 3")

    def scan(self, data: np.ndarray, inclusive: bool = True) -> np.ndarray:
        raise NotImplementedError("Scan kernel implemented in Phase 4")

    def sort(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sort kernel implemented in Phase 5")

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GEMM kernel implemented in Phase 6")
