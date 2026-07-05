#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <vector>

namespace waveprim {

// Prefix scan of a float array on the GPU.
// program must be built from kernels/scan.cl.
// Returns the exclusive scan; set inclusive=true for an inclusive scan.
std::vector<float> scan(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n,
    bool             inclusive = true,
    size_t           wg_size   = 256
);

} // namespace waveprim
