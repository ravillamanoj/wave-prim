# wave-prim

[![CI](https://github.com/ravillamanoj/wave-prim/actions/workflows/ci.yml/badge.svg)](https://github.com/ravillamanoj/wave-prim/actions/workflows/ci.yml)

OpenCL parallel primitives for ARM Mali GPU — reduce, prefix scan, bitonic sort,
and tiled GEMM — with a property-based correctness testing framework that runs
on CPU via [POCL](http://portablecl.org/) without GPU hardware.

---

## Primitives

| Primitive | Algorithm | Work complexity | Depth |
|-----------|-----------|----------------|-------|
| **Reduce** | Two-pass tree reduction | O(n) | O(log n) |
| **Scan**   | Blelloch work-efficient | O(2n) | O(log n) |
| **Sort**   | Batcher bitonic sort | O(n log²n) | O(log²n) |
| **GEMM**   | Tiled shared-memory matmul | O(2MKN) | O(K/TILE) |

Each primitive ships with:

- An OpenCL kernel (`.cl`) tuned for Mali Valhall/Immortalis work-group geometry
- A C++ host wrapper in `include/waveprim/` + `host/`
- A Python NumPy reference implementation in `reference/`
- Property-based tests that verify the full algebraic contract, not just one output

---

## Repository layout

```
kernels/          OpenCL kernel sources
  reduce.cl       two-pass tree reduction  (sum / min / max)
  scan.cl         Blelloch exclusive prefix scan
  sort.cl         Batcher bitonic sort
  gemm.cl         tiled 16×16 matrix multiply

include/waveprim/ C++ public headers
host/             C++ host wrappers + standalone validation drivers
reference/        NumPy reference implementations (ground truth)

tests/
  framework/      abstract Backend, CPUReference, OpenCLBackend, fixtures
  test_reduce.py
  test_scan.py
  test_sort.py
  test_gemm.py
  test_framework.py
  test_infrastructure.py

bench/
  benchmark.py    wall-clock throughput benchmark (GB/s, GFLOP/s)
```

---

## Design choices for ARM Mali

### Work-group size — 256

Mali Valhall (G710, G715, Immortalis-G715) executes 16-wide SIMD per shader
core. A work-group of 256 threads = 16 waves fills one execution slot cleanly
and gives the driver enough work to hide memory latency through wave switching.

### Blelloch scan over Hillis-Steele

Hillis-Steele runs in O(n log n) operations — every doubling step touches all
elements. Blelloch completes the same inclusive scan in O(2n) operations
(up-sweep + down-sweep), matching the sequential work bound. For memory-bound
kernels on Mali's 64-bit LPDDR5 bus, fewer total memory accesses matter more
than occupancy.

### Tiled GEMM — 16×16 tiles

A 16×16 tile holds 256 floats × 4 bytes = 1 KB in local (shared) memory.
Two tiles (A and B) consume 2 KB, well within Mali's 32 KB local memory per
shader core. Each thread reuses 16 elements from each tile across the inner
product loop, achieving a 16× reduction in global memory traffic versus a
naive implementation.

The `#pragma unroll` on the 16-iteration inner loop, together with
`__attribute__((reqd_work_group_size(16, 16, 1)))`, lets the Mali compiler
emit a static FMA chain without loop-control overhead.

### Bitonic sort

All comparisons at each step are independent — no element touches two
different pairs in the same step. This maps directly to SIMT execution:
every work item does exactly one compare-swap with no synchronisation
between peers in the same step, and only one `barrier` is needed per step.

### `reqd_work_group_size` annotations

All kernels carry `__attribute__((reqd_work_group_size(...)))`:

- Allows the Mali compiler to treat local size as a compile-time constant
- Enables full unrolling of logarithmic loops in reduce and scan
- Eliminates dynamic `get_local_size()` calls in the inner paths

---

## Building

**Requirements:** CMake 3.16+, C++17 compiler, OpenCL headers

```bash
# Install POCL (CPU OpenCL — no GPU needed)
sudo apt-get install pocl-opencl-icd ocl-icd-opencl-dev   # Ubuntu/Debian
brew install pocl                                           # macOS

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Verify OpenCL is reachable
./build/device_query
```

---

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

207 tests across four primitives, covering:

- Algebraic properties (identity, commutativity, decomposability)
- Edge cases (n=1, reverse-sorted, all-equal, non-power-of-two dimensions)
- OpenCL output vs CPU reference with tolerance calibrated for float32 accumulation

---

## Running the benchmark

```bash
python bench/benchmark.py
```

Example output on an 8-core CPU via POCL:

```
Device : pthread-Intel(R) Core(TM) i7-...
CUs    : 8
Freq   : 2400 MHz
==========================================================

Reduce  (GB/s = bytes read / time)
  sum  n=    4,096              0.05 GB/s  (  0.319 ms)
  sum  n=   65,536              0.38 GB/s  (  0.662 ms)
  sum  n=1,048,576              2.11 GB/s  (  1.994 ms)
  sum  n=4,194,304              4.89 GB/s  (  3.438 ms)

GEMM    (GFLOP/s = 2·M·K·N / time)
  C=64x64 = A·B  (64x64)        0.042 GFLOP/s  (  0.013 ms)
  C=128x128 = A·B  (128x128)    0.147 GFLOP/s  (  0.029 ms)
  C=256x256 = A·B  (256x256)    0.632 GFLOP/s  (  0.053 ms)
  C=512x512 = A·B  (512x512)    1.840 GFLOP/s  (  0.146 ms)
```

On Mali GPU the numbers are 10-50× higher due to dedicated compute units and
higher memory bandwidth (LPDDR5 at 51.2 GB/s on Dimensity 9000).

---

## CI

GitHub Actions runs on every push:

1. Installs POCL (CPU OpenCL)
2. Builds all C++ targets with CMake
3. Runs each C++ validation driver (`reduce_test`, `scan_test`, `sort_test`, `gemm_test`)
4. Runs the full Python test suite (207 tests)

No GPU hardware is required at any point.
