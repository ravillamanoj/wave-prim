#include "waveprim/sort.h"

#include <CL/cl.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveprim {

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

std::vector<float> sort(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n
) {
    assert(n > 0 && "Cannot sort an empty array");

    size_t n_padded = next_pow2(n);
    int    log2_n   = 0;
    for (size_t p = n_padded; p > 1; p >>= 1) ++log2_n;

    // Pad with +inf so extra slots sort to the tail
    std::vector<float> padded(n_padded, std::numeric_limits<float>::max());
    std::copy(data, data + n, padded.begin());

    cl_int err;
    cl_mem d_data = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   n_padded * sizeof(float), padded.data(), &err);
    if (err) throw std::runtime_error("clCreateBuffer sort failed");

    cl_kernel kernel = clCreateKernel(program, "bitonic_sort_step", &err);
    if (err) throw std::runtime_error("clCreateKernel bitonic_sort_step failed");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);

    for (int stage = 0; stage < log2_n; ++stage) {
        for (int step = stage; step >= 0; --step) {
            cl_int s = stage, t = step;
            clSetKernelArg(kernel, 1, sizeof(cl_int), &s);
            clSetKernelArg(kernel, 2, sizeof(cl_int), &t);
            err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                         &n_padded, nullptr, 0, nullptr, nullptr);
            if (err) throw std::runtime_error("clEnqueueNDRangeKernel bitonic_sort_step failed");
        }
    }

    clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0,
                        n_padded * sizeof(float), padded.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseMemObject(d_data);

    return std::vector<float>(padded.begin(), padded.begin() + n);
}

} // namespace waveprim


// ---------------------------------------------------------------------------
// Standalone driver — validates sort against std::sort
// ---------------------------------------------------------------------------

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    const std::string kernel_path = (argc > 1) ? argv[1] : "kernels/sort.cl";

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);

    char dev_name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    std::cout << "Device: " << dev_name << "\n\n";

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);

    std::string src = load_file(kernel_path);
    const char* src_ptr = src.c_str();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_ptr, nullptr, &err);
    if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        char log[4096] = {};
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build error:\n" << log << "\n";
        return 1;
    }

    const size_t N = 1000; // non-power-of-two to exercise padding
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i)
        data[i] = static_cast<float>(N - i); // reverse order

    std::vector<float> cpu = data;
    std::sort(cpu.begin(), cpu.end());

    std::vector<float> gpu = waveprim::sort(ctx, queue, program, data.data(), N);

    bool sorted_ok = std::is_sorted(gpu.begin(), gpu.end());
    bool match_ok  = (gpu == cpu);

    std::cout << "sorted:  " << (sorted_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "matches: " << (match_ok  ? "PASS" : "FAIL") << "\n";

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return (sorted_ok && match_ok) ? 0 : 1;
}
