#pragma once

#include <CL/cl.h>
#include <cstddef>

namespace waveprim {

enum class ReduceOp { Sum, Min, Max };

// Reduce a float array on the GPU.
// program must be built from kernels/reduce.cl.
// Returns the scalar reduced value.
float reduce(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n,
    ReduceOp         op      = ReduceOp::Sum,
    size_t           wg_size = 256
);

} // namespace waveprim
