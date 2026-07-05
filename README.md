# wave-prim

[![CI](https://github.com/ravillamanoj/wave-prim/actions/workflows/ci.yml/badge.svg)](https://github.com/ravillamanoj/wave-prim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCL](https://img.shields.io/badge/OpenCL-1.2%2B-red.svg)](https://www.khronos.org/opencl/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ARM Mali](https://img.shields.io/badge/Target-ARM%20Mali%20Valhall-0091BD.svg)](https://developer.arm.com/Processors/Mali-G715)
[![POCL](https://img.shields.io/badge/CI%20runtime-POCL-green.svg)](http://portablecl.org/)

OpenCL parallel primitives for ARM Mali GPU — reduce, prefix scan, bitonic sort,
and tiled GEMM — with a property-based correctness testing framework that runs
on CPU via POCL without GPU hardware.

---

## What it is

wave-prim is a library of four foundational GPU compute primitives implemented
in OpenCL C, each paired with a C++ host driver and a Python test harness. The
kernels are tuned for the ARM Mali Valhall/Immortalis architecture
(work-group size 256, 16×16 GEMM tiles) but run on any OpenCL 1.2+ device.

| Primitive | Algorithm | Work | Depth |
|-----------|-----------|------|-------|
| **Reduce** | Two-pass tree reduction | O(n) | O(log n) |
| **Scan**   | Blelloch work-efficient prefix scan | O(2n) | O(log n) |
| **Sort**   | Batcher bitonic sort | O(n log²n) | O(log²n) |
| **GEMM**   | Tiled shared-memory matrix multiply | O(2MKN) | O(K / TILE) |

---

## The problem it solves

GPU code fails silently. A reduction kernel that reads out of bounds produces a
wrong number with no crash; a scan kernel with a missing barrier gives the right
answer on one device and garbage on another. Most GPU projects ship with no tests
at all — only benchmarks — so correctness is discovered in production.

wave-prim addresses this with three ideas:

**1. Property-based tests, not example-based tests.**
Instead of checking `sum([1,2,3,4]) == 10`, the test suite verifies the full
algebraic contract — reduce is decomposable (`reduce(A||B) == reduce(A) op reduce(B)`),
scan's last element equals the total sum, sort output is always a permutation of
the input, GEMM satisfies the transpose identity `(AB)ᵀ = BᵀAᵀ`. These properties
catch whole classes of bugs that a single example misses.

**2. Dual-backend testing.**
Every test runs against both a CPU NumPy reference (ground truth, float64
precision) and the actual OpenCL kernel. Disagreement is caught immediately with
a diff showing expected vs actual values and the numerical tolerance violated.

**3. Hardware-free CI.**
[POCL](http://portablecl.org/) (Portable Computing Language) is an open-source
OpenCL implementation that runs kernels on CPU. All 207 tests and all four C++
validation drivers execute in GitHub Actions with no GPU in the loop, making
correctness a hard gate on every push.

---

## Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| CMake | 3.16+ | C++ build |
| C++ compiler | C++17 | Host wrappers |
| POCL | any | CPU OpenCL runtime (development / CI) |
| Python | 3.9+ | Test harness and benchmark |
| pyopencl | 2024+ | Python → OpenCL bridge |
| numpy | 1.24+ | Reference implementations |
| pytest | 7+ | Test runner |

On real hardware, replace POCL with the vendor OpenCL ICD (Mali OpenCL driver,
Apple GPU, ROCm, etc.) — the kernels and host code are portable.

---

## Setup

### 1. Install POCL (CPU OpenCL — no GPU required)

```bash
# Ubuntu / Debian
sudo apt-get install pocl-opencl-icd ocl-icd-opencl-dev

# macOS (Homebrew)
brew install pocl
```

### 2. Clone and build the C++ drivers

```bash
git clone https://github.com/ravillamanoj/wave-prim.git
cd wave-prim

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Verify OpenCL is reachable:

```bash
./build/device_query
```

Expected output (POCL on CPU):

```
Found 1 OpenCL platform(s):

Platform : Portable Computing Language
Version  : OpenCL 3.0 POCL ...

  Device         : pthread-<cpu-model>
  Type           : CPU
  Compute units  : 8
  Max WG size    : 4096
  Local memory   : 256 KB
  Global memory  : 16384 MB
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains: `pyopencl`, `numpy`, `pytest`.

---

## Running tests

```bash
pytest tests/ -v
```

### Test results

```
platform darwin -- Python 3.12.4, pytest-9.1.0
collected 207 items

tests/test_framework.py::TestCPUReduce::test_sum PASSED
tests/test_framework.py::TestCPUReduce::test_min PASSED
...
tests/test_reduce.py::TestReduceOpenCL::test_sum_large_array PASSED
...
tests/test_scan.py::TestScanOpenCL::test_multi_block_exclusive PASSED
tests/test_scan.py::TestScanOpenCL::test_multi_block_inclusive PASSED
...
tests/test_sort.py::TestSortOpenCL::test_non_power_of_two PASSED
tests/test_sort.py::TestSortOpenCL::test_large_array PASSED
...
tests/test_gemm.py::TestGemmOpenCL::test_non_tile_multiple PASSED
tests/test_gemm.py::TestGemmOpenCL::test_large PASSED

========================= 207 passed in 0.35s =========================
```

### Test breakdown

| Test file | Suite | Tests | What is verified |
|-----------|-------|------:|-----------------|
| `test_framework.py` | `TestCPUReduce` | 11 | NumPy reference reduce: sum/min/max, identity, sizes |
| `test_framework.py` | `TestCPUScan` | 5 | NumPy reference scan: inclusive, exclusive, zeros |
| `test_framework.py` | `TestCPUSort` | 10 | NumPy reference sort: sorted, permutation, idempotent |
| `test_framework.py` | `TestCPUGemm` | 4 | NumPy reference GEMM: identity, zero, transpose, shape |
| `test_framework.py` | `TestToleranceUtilities` | 6 | `assert_allclose`, `assert_sorted`, `assert_permutation` |
| `test_framework.py` | `TestOpenCLBackend` | 5 | OpenCL device init + each primitive executes |
| `test_infrastructure.py` | — | 5 | Device detection, platform info |
| `test_reduce.py` | `TestReduceProperties` | 26 | Algebraic properties against CPU reference |
| `test_reduce.py` | `TestReduceOpenCL` | 24 | GPU vs CPU reference across all sizes and ops |
| `test_scan.py` | `TestScanProperties` | 17 | Prefix properties, numpy parity, zeros |
| `test_scan.py` | `TestScanOpenCL` | 19 | GPU vs CPU across sizes, multi-block, single element |
| `test_sort.py` | `TestSortProperties` | 18 | Sorted, permutation, idempotent, edge cases |
| `test_sort.py` | `TestSortOpenCL` | 25 | GPU sort: sorted, permutation, non-power-of-two, n=4096 |
| `test_gemm.py` | `TestGemmProperties` | 14 | Identity, zero, transpose, associativity, shape |
| `test_gemm.py` | `TestGemmOpenCL` | 18 | GPU vs CPU: square, non-square, non-tile-multiple, 128×128 |
| **Total** | | **207** | |

Test sizes exercised: `1, 2, 16, 64, 256, 1024` for 1D primitives;
`4, 8, 16, 32, 64` for GEMM dimensions.

---

## Running the benchmark

```bash
python bench/benchmark.py
```

### Benchmark results (Apple M2 Pro via POCL, best of 10 runs)

```
==========================================================
Device : Apple M2 Pro
CUs    : 19
Freq   : 1000 MHz
==========================================================
(best of 10 runs, warmup=3, all sizes in elements of float32)

Reduce  (GB/s = bytes read / time)
  sum  n=    4,096                   0.03 GB/s  (  0.550 ms)
  sum  n=   65,536                   0.71 GB/s  (  0.368 ms)
  sum  n=1,048,576                   4.10 GB/s  (  1.023 ms)
  sum  n=4,194,304                   3.42 GB/s  (  4.907 ms)

Scan    (GB/s = 2 × bytes / time — read + write)
  excl n=    4,096                   0.05 GB/s  (  0.682 ms)
  excl n=   65,536                   0.69 GB/s  (  0.762 ms)
  excl n=1,048,576                   3.82 GB/s  (  2.196 ms)
  excl n=4,194,304                   4.26 GB/s  (  7.880 ms)

Sort    (Melem/s)
  bitonic n=  4,096               0.28 Melem/s  ( 14.495 ms)
  bitonic n= 65,536               2.54 Melem/s  ( 25.825 ms)
  bitonic n=524,288              12.42 Melem/s  ( 42.210 ms)

GEMM    (GFLOP/s = 2·M·K·N / time)
  C=64x64   = A·B                 1.08 GFLOP/s  (  0.484 ms)
  C=128x128 = A·B                 6.57 GFLOP/s  (  0.638 ms)
  C=256x256 = A·B                52.10 GFLOP/s  (  0.644 ms)
  C=512x512 = A·B               208.66 GFLOP/s  (  1.286 ms)
```

These numbers reflect POCL running on CPU cores. On Mali Immortalis-G715
(e.g. Dimensity 9000 at 51.2 GB/s LPDDR5 bandwidth), reduce and scan
would be 10–15× higher, and GEMM would approach the GPU's theoretical
FP32 peak of ~2.9 TFLOP/s.

---

## Repository layout

```
kernels/            OpenCL kernel sources (.cl)
  reduce.cl         two-pass tree reduction — sum, min, max
  scan.cl           Blelloch exclusive prefix scan + offset fixup
  sort.cl           Batcher bitonic compare-swap step
  gemm.cl           tiled 16×16 shared-memory matrix multiply

include/waveprim/   C++ public headers
  reduce.h
  scan.h
  sort.h
  gemm.h

host/               C++ host wrappers + standalone validation drivers
  device_query.cpp
  reduce.cpp
  scan.cpp
  sort.cpp
  gemm.cpp

reference/          NumPy reference implementations (ground truth)
  reduce.py
  scan.py
  sort.py
  gemm.py

tests/
  framework/        Backend abstraction, CPUReference, OpenCLBackend, fixtures
  test_framework.py
  test_infrastructure.py
  test_reduce.py
  test_scan.py
  test_sort.py
  test_gemm.py

bench/
  benchmark.py      Throughput benchmark — GB/s, Melem/s, GFLOP/s
```

---

## Design choices for ARM Mali

### Work-group size — 256

Mali Valhall (G710, G715, Immortalis-G715) executes 16-wide SIMD per shader
core. A work-group of 256 threads = 16 waves fills one execution slot cleanly
and gives the driver enough in-flight work to hide memory latency through
wave switching without over-subscribing the register file.

### Blelloch scan over Hillis-Steele

Hillis-Steele runs in O(n log n) total operations — every doubling step
reads and writes all n elements. Blelloch completes the same exclusive scan
in O(2n) operations (one up-sweep + one down-sweep), matching the sequential
work bound. For memory-bound kernels on Mali's LPDDR5 bus, fewer total
memory transactions matter more than raw occupancy.

### Tiled GEMM — 16×16 tiles

A 16×16 tile occupies 256 × 4 bytes = 1 KB of local (shared) memory.
Two tiles (for A and B) use 2 KB, comfortably within Mali's 32 KB local
memory per shader core. Each work item reuses 16 elements from each tile,
achieving a 16× reduction in global memory bandwidth versus a naive
element-per-thread implementation.

`#pragma unroll` on the 16-iteration inner loop, combined with
`__attribute__((reqd_work_group_size(16, 16, 1)))`, lets the Mali compiler
schedule a straight static FMA chain with no loop-control overhead and
full software pipelining of loads from local memory.

### Bitonic sort

Every comparison-swap at a given step is independent — no element
participates in more than one pair per step. This maps perfectly to
SIMT: each work item does exactly one compare-swap, and a single
`barrier(CLK_LOCAL_MEM_FENCE)` per step is all the synchronisation
needed. The array is padded to the next power of two with `MAXFLOAT`
so the sort always operates on a power-of-two length regardless of
input size.

### `reqd_work_group_size` annotations

All kernels carry `__attribute__((reqd_work_group_size(...)))`. This:

- Lets the Mali compiler treat local size as a compile-time constant
- Enables full static unrolling of the 8-step reduce loop (log₂256)
  and the 9-step scan loops (log₂512)
- Eliminates dynamic `get_local_size()` calls from hot paths
- Allows the driver to validate the launch configuration at enqueue time

---

## CI

GitHub Actions runs on every push to `main`:

1. Install POCL (CPU OpenCL — no GPU)
2. Build all C++ targets with CMake
3. Run each C++ validation driver (`reduce_test`, `scan_test`, `sort_test`, `gemm_test`)
4. Run the full Python test suite (207 tests)

No physical GPU is required at any stage of the pipeline.

---

## License

This project is licensed under the [MIT License](LICENSE).
