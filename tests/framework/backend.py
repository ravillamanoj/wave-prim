from abc import ABC, abstractmethod
from pathlib import Path
import math

import numpy as np


KERNELS_DIR = Path(__file__).resolve().parent.parent.parent / "kernels"


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

    _WG_SIZE = 256  # power-of-2 workgroup size; suits Mali Valhall and POCL CPU

    def __init__(self, platform_idx: int = 0, device_idx: int = 0):
        import pyopencl as cl
        self._cl = cl
        platforms = cl.get_platforms()
        device = platforms[platform_idx].get_devices()[device_idx]
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        self._programs: dict = {}
        self._kernels: dict = {}

    def _load_kernel(self, kernel_file: str, kernel_name: str):
        """Compile, cache, and return a named OpenCL kernel."""
        key = (kernel_file, kernel_name)
        if key not in self._kernels:
            if kernel_file not in self._programs:
                src = (KERNELS_DIR / kernel_file).read_text()
                self._programs[kernel_file] = self._cl.Program(self.ctx, src).build()
            self._kernels[key] = self._cl.Kernel(self._programs[kernel_file], kernel_name)
        return self._kernels[key]

    def reduce(self, data: np.ndarray, op: str = "sum") -> float:
        if op not in ("sum", "min", "max"):
            raise ValueError(f"Unsupported op '{op}'")
        if len(data) == 0:
            raise ValueError("Cannot reduce an empty array")

        data = data.astype(np.float32)
        n = len(data)
        wg = self._WG_SIZE
        global_size = math.ceil(n / wg) * wg
        num_groups = global_size // wg

        kernel = self._load_kernel("reduce.cl", f"reduce_{op}")
        mf = self._cl.mem_flags

        d_input   = self._cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=data)
        d_partial = self._cl.Buffer(self.ctx, mf.WRITE_ONLY, num_groups * data.itemsize)
        local_mem = self._cl.LocalMemory(wg * data.itemsize)

        kernel(self.queue, (global_size,), (wg,), d_input, d_partial, local_mem, np.int32(n))

        partial = np.empty(num_groups, dtype=np.float32)
        self._cl.enqueue_copy(self.queue, partial, d_partial)
        self.queue.finish()

        # Second-pass aggregation on CPU
        return float({"sum": np.sum, "min": np.min, "max": np.max}[op](partial))

    def scan(self, data: np.ndarray, inclusive: bool = True) -> np.ndarray:
        raise NotImplementedError("Scan kernel implemented in Phase 4")

    def sort(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sort kernel implemented in Phase 5")

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        raise NotImplementedError("GEMM kernel implemented in Phase 6")
