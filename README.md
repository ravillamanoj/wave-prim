# wave-prim

OpenCL parallel primitives (reduce, scan, sort, GEMM) with property-based correctness
testing targeting ARM Mali GPU. Runs without GPU hardware via POCL.

## Overview

wave-prim implements four foundational parallel algorithms in OpenCL:

| Primitive | Description |
|---|---|
| **Reduce** | Collapse an array to a single value — sum, min, max |
| **Scan** | Parallel prefix sum (inclusive and exclusive) |
| **Sort** | Bitonic sort — parallel comparison network |
| **GEMM** | Tiled general matrix multiply using local memory |

Each primitive ships with:
- An OpenCL kernel (`.cl`) optimized for Mali Valhall/Immortalis work-group sizes
- A C++ host wrapper
- A Python NumPy reference implementation (ground truth)
- Property-based tests that validate mathematical correctness — not just one output, but the full algebraic contract

## Why property-based testing?

Most GPU projects have no tests at all. Benchmarks measure speed; they don't catch wrong answers.
wave-prim tests mathematical properties of each primitive:

- **Reduce**: identity element, decomposability, single-element case
- **Scan**: last element equals total reduction, per-element definition
- **Sort**: output is sorted, output is a permutation of input, idempotent
- **GEMM**: identity matrix, transpose property, scalar associativity

The test framework runs against both a CPU NumPy reference and the OpenCL backend,
comparing outputs with configurable numerical tolerance.

## Running without GPU hardware

wave-prim uses [POCL](http://portablecl.org/) to run OpenCL kernels on CPU.
No GPU or physical hardware required for development or CI.

```bash
# Ubuntu
sudo apt-get install pocl-opencl-icd ocl-icd-opencl-dev
```

## Build

**Prerequisites:** CMake 3.16+, a C++17 compiler, OpenCL headers (via POCL or SDK)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Device query

Verify OpenCL is available on your machine:

```bash
./build/device_query
```

Example output (POCL on CPU):
```
Found 1 OpenCL platform(s):

Platform : Portable Computing Language
Version  : OpenCL 3.0 POCL ...

  Device         : pthread-<cpu-name>
  Type           : CPU
  Compute units  : 8
  Max WG size    : 4096
  Local memory   : 256 KB
  Global memory  : 16384 MB
```

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Target hardware

Optimized work-group sizes and local memory layouts target:
- ARM Mali-G710 / G715 (Valhall architecture)
- ARM Mali-G715 / Immortalis-G715 (Immortalis architecture)

The same kernels run on any OpenCL-capable device.

## Project status

| Phase | Status |
|---|---|
| Foundation & infrastructure | ✅ Complete |
| Testing framework | 🔲 Upcoming |
| Reduce primitive | 🔲 Upcoming |
| Scan primitive | 🔲 Upcoming |
| Sort primitive | 🔲 Upcoming |
| GEMM primitive | 🔲 Upcoming |
| Mali optimization | 🔲 Upcoming |
