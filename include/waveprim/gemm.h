#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace waveprim {

// Tiled matrix multiply: C = A * B
//   A: M x K row-major float array
//   B: K x N row-major float array
// Returns an M x N row-major float array.
// program must be built from kernels/gemm.cl.
std::vector<float> gemm(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     A,
    const float*     B,
    int              M,
    int              K,
    int              N
);

} // namespace waveprim
