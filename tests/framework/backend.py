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
        data = data.astype(np.float32)
        n = len(data)
        wg = self._WG_SIZE
        block_size = wg * 2
        num_blocks = math.ceil(n / block_size)
        global_size = num_blocks * wg

        mf = self._cl.mem_flags
        d_input      = self._cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=data)
        d_output     = self._cl.Buffer(self.ctx, mf.READ_WRITE, data.nbytes)
        d_block_sums = self._cl.Buffer(self.ctx, mf.READ_WRITE, num_blocks * data.itemsize)
        local_mem    = self._cl.LocalMemory(block_size * data.itemsize)

        k_scan = self._load_kernel("scan.cl", "scan_exclusive")
        k_scan(self.queue, (global_size,), (wg,),
               d_output, d_input, d_block_sums, local_mem, np.int32(n))

        if num_blocks > 1:
            block_sums = np.empty(num_blocks, dtype=np.float32)
            self._cl.enqueue_copy(self.queue, block_sums, d_block_sums)
            self.queue.finish()
            offsets = np.concatenate([[0.0], np.cumsum(block_sums, dtype=np.float64)[:-1]]).astype(np.float32)
            d_offsets = self._cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=offsets)

            add_global = math.ceil(n / wg) * wg
            k_add = self._load_kernel("scan.cl", "scan_add_offsets")
            k_add(self.queue, (add_global,), (wg,),
                  d_output, d_offsets, np.int32(n), np.int32(block_size))

        result = np.empty(n, dtype=np.float32)
        self._cl.enqueue_copy(self.queue, result, d_output)
        self.queue.finish()

        if inclusive:
            result += data
        return result

    def sort(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        n = len(data)

        log2_n = max(1, math.ceil(math.log2(max(n, 2))))
        n_padded = 1 << log2_n

        padded = np.full(n_padded, np.finfo(np.float32).max, dtype=np.float32)
        padded[:n] = data

        mf = self._cl.mem_flags
        d_data = self._cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=padded)

        kernel = self._load_kernel("sort.cl", "bitonic_sort_step")
        for stage in range(log2_n):
            for step in range(stage, -1, -1):
                kernel(self.queue, (n_padded,), None,
                       d_data, np.int32(stage), np.int32(step))

        self._cl.enqueue_copy(self.queue, padded, d_data)
        self.queue.finish()
        return padded[:n].copy()

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Shape mismatch: A is {M}x{K}, B is {K2}x{N}")

        TILE = 16
        global_m = math.ceil(M / TILE) * TILE
        global_n = math.ceil(N / TILE) * TILE

        mf = self._cl.mem_flags
        d_A = self._cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=A)
        d_B = self._cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=B)
        d_C = self._cl.Buffer(self.ctx, mf.WRITE_ONLY, M * N * 4)

        kernel = self._load_kernel("gemm.cl", "gemm")
        kernel(self.queue, (global_m, global_n), (TILE, TILE),
               d_A, d_B, d_C,
               np.int32(M), np.int32(K), np.int32(N),
               self._cl.LocalMemory(TILE * TILE * 4),
               self._cl.LocalMemory(TILE * TILE * 4))

        C = np.empty((M, N), dtype=np.float32)
        self._cl.enqueue_copy(self.queue, C, d_C)
        self.queue.finish()
        return C
