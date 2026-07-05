#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace waveprim {

// Sort a float array on the GPU using Batcher bitonic sort.
// program must be built from kernels/sort.cl.
// Returns a sorted copy of data in ascending order.
std::vector<float> sort(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n
);

} // namespace waveprim
