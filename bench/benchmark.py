#!/usr/bin/env python3
"""Throughput benchmark for each wave-prim primitive.

Run from the project root:
    python bench/benchmark.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.framework.backend import OpenCLBackend


def _bench(fn, *args, warmup: int = 3, runs: int = 10) -> float:
    """Return the minimum wall-clock seconds over `runs` timed invocations."""
    for _ in range(warmup):
        fn(*args)
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _row(label: str, metric: str, ms: float) -> str:
    return f"  {label:<28}  {metric:>14}  ({ms:7.3f} ms)"


def bench_reduce(backend: OpenCLBackend) -> None:
    print("\nReduce  (GB/s = bytes read / time)")
    for n in [4_096, 65_536, 1_048_576, 4_194_304]:
        data = np.random.default_rng(0).random(n).astype(np.float32)
        ms = _bench(backend.reduce, data, "sum") * 1e3
        gbs = n * 4 / 1e9 / (ms * 1e-3)
        print(_row(f"sum  n={n:>9,}", f"{gbs:.2f} GB/s", ms))


def bench_scan(backend: OpenCLBackend) -> None:
    print("\nScan    (GB/s = 2 × bytes / time — read + write)")
    for n in [4_096, 65_536, 1_048_576, 4_194_304]:
        data = np.random.default_rng(0).random(n).astype(np.float32)
        ms = _bench(backend.scan, data) * 1e3
        gbs = n * 4 * 2 / 1e9 / (ms * 1e-3)
        print(_row(f"excl n={n:>9,}", f"{gbs:.2f} GB/s", ms))


def bench_sort(backend: OpenCLBackend) -> None:
    print("\nSort    (Melem/s)")
    for n in [4_096, 65_536, 524_288]:
        data = np.random.default_rng(0).random(n).astype(np.float32)
        ms = _bench(backend.sort, data) * 1e3
        meps = n / (ms * 1e-3) / 1e6
        print(_row(f"bitonic n={n:>7,}", f"{meps:.2f} Melem/s", ms))


def bench_gemm(backend: OpenCLBackend) -> None:
    print("\nGEMM    (GFLOP/s = 2·M·K·N / time)")
    for m in [64, 128, 256, 512]:
        rng = np.random.default_rng(0)
        A = rng.random((m, m)).astype(np.float32)
        B = rng.random((m, m)).astype(np.float32)
        ms = _bench(backend.gemm, A, B) * 1e3
        gflops = 2 * m ** 3 / 1e9 / (ms * 1e-3)
        print(_row(f"C={m}x{m} = A·B  ({m}x{m})", f"{gflops:.3f} GFLOP/s", ms))


def main() -> None:
    try:
        backend = OpenCLBackend()
    except Exception as e:
        print(f"No OpenCL device: {e}", file=sys.stderr)
        sys.exit(1)

    import pyopencl as cl
    dev = backend.ctx.devices[0]
    print("=" * 58)
    print(f"Device : {dev.name.strip()}")
    print(f"CUs    : {dev.max_compute_units}")
    print(f"Freq   : {dev.max_clock_frequency} MHz")
    print("=" * 58)
    print("(best of 10 runs, warmup=3, all sizes in elements of float32)")

    bench_reduce(backend)
    bench_scan(backend)
    bench_sort(backend)
    bench_gemm(backend)

    print()


if __name__ == "__main__":
    main()
