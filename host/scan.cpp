#include "waveprim/scan.h"

#include <CL/cl.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveprim {

std::vector<float> scan(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n,
    bool             inclusive,
    size_t           wg_size
) {
    assert(n > 0 && "Cannot scan an empty array");
    assert((wg_size & (wg_size - 1)) == 0 && "wg_size must be a power of 2");

    size_t block_size  = wg_size * 2;
    size_t num_blocks  = (n + block_size - 1) / block_size;
    size_t global_size = num_blocks * wg_size;

    cl_int err;

    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n * sizeof(float), const_cast<float*>(data), &err);
    if (err) throw std::runtime_error("clCreateBuffer input failed");

    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                     n * sizeof(float), nullptr, &err);
    if (err) throw std::runtime_error("clCreateBuffer output failed");

    cl_mem d_block_sums = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                         num_blocks * sizeof(float), nullptr, &err);
    if (err) throw std::runtime_error("clCreateBuffer block_sums failed");

    // Pass 1: exclusive scan within each block
    cl_kernel k_scan = clCreateKernel(program, "scan_exclusive", &err);
    if (err) throw std::runtime_error("clCreateKernel scan_exclusive failed");

    cl_int n_int = static_cast<cl_int>(n);
    clSetKernelArg(k_scan, 0, sizeof(cl_mem), &d_output);
    clSetKernelArg(k_scan, 1, sizeof(cl_mem), &d_input);
    clSetKernelArg(k_scan, 2, sizeof(cl_mem), &d_block_sums);
    clSetKernelArg(k_scan, 3, block_size * sizeof(float), nullptr); // local buf
    clSetKernelArg(k_scan, 4, sizeof(cl_int), &n_int);

    err = clEnqueueNDRangeKernel(queue, k_scan, 1, nullptr,
                                 &global_size, &wg_size, 0, nullptr, nullptr);
    if (err) throw std::runtime_error("clEnqueueNDRangeKernel scan_exclusive failed");

    if (num_blocks > 1) {
        // Pass 2: read block sums to CPU; compute exclusive scan of sums
        std::vector<float> block_sums(num_blocks);
        clEnqueueReadBuffer(queue, d_block_sums, CL_TRUE, 0,
                            num_blocks * sizeof(float), block_sums.data(), 0, nullptr, nullptr);

        std::vector<float> offsets(num_blocks);
        offsets[0] = 0.0f;
        for (size_t i = 1; i < num_blocks; ++i)
            offsets[i] = offsets[i - 1] + block_sums[i - 1];

        cl_mem d_offsets = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          num_blocks * sizeof(float), offsets.data(), &err);
        if (err) throw std::runtime_error("clCreateBuffer offsets failed");

        // Pass 3: add per-block prefix offsets to output
        size_t add_global = ((n + wg_size - 1) / wg_size) * wg_size;
        cl_int bs_int = static_cast<cl_int>(block_size);

        cl_kernel k_add = clCreateKernel(program, "scan_add_offsets", &err);
        if (err) throw std::runtime_error("clCreateKernel scan_add_offsets failed");

        clSetKernelArg(k_add, 0, sizeof(cl_mem), &d_output);
        clSetKernelArg(k_add, 1, sizeof(cl_mem), &d_offsets);
        clSetKernelArg(k_add, 2, sizeof(cl_int), &n_int);
        clSetKernelArg(k_add, 3, sizeof(cl_int), &bs_int);

        err = clEnqueueNDRangeKernel(queue, k_add, 1, nullptr,
                                     &add_global, &wg_size, 0, nullptr, nullptr);
        if (err) throw std::runtime_error("clEnqueueNDRangeKernel scan_add_offsets failed");

        clReleaseKernel(k_add);
        clReleaseMemObject(d_offsets);
    }

    std::vector<float> result(n);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0,
                        n * sizeof(float), result.data(), 0, nullptr, nullptr);

    if (inclusive)
        for (size_t i = 0; i < n; ++i) result[i] += data[i];

    clReleaseKernel(k_scan);
    clReleaseMemObject(d_block_sums);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_input);

    return result;
}

} // namespace waveprim


// ---------------------------------------------------------------------------
// Standalone driver — validates scan against CPU ground truth
// ---------------------------------------------------------------------------

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    const std::string kernel_path = (argc > 1) ? argv[1] : "kernels/scan.cl";

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

    const size_t N = 1200; // spans multiple 512-element blocks
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = 1.0f;

    // CPU exclusive scan
    std::vector<float> cpu_excl(N);
    cpu_excl[0] = 0.0f;
    for (size_t i = 1; i < N; ++i) cpu_excl[i] = cpu_excl[i - 1] + data[i - 1];

    std::vector<float> gpu_excl = waveprim::scan(ctx, queue, program,
                                                  data.data(), N, false);
    std::vector<float> gpu_incl = waveprim::scan(ctx, queue, program,
                                                  data.data(), N, true);

    bool all_pass = true;
    size_t errors = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = std::fabs(cpu_excl[i] - gpu_excl[i]);
        if (diff > 0.5f) { ++errors; all_pass = false; }
    }
    std::cout << "exclusive scan: " << (all_pass ? "PASS" : "FAIL")
              << "  errors=" << errors << "\n";

    // Inclusive: last element should equal N (sum of N ones)
    float last = gpu_incl[N - 1];
    bool incl_ok = std::fabs(last - static_cast<float>(N)) < 1.0f;
    std::cout << "inclusive scan last=" << last
              << " expected=" << N
              << "  " << (incl_ok ? "PASS" : "FAIL") << "\n";

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return (all_pass && incl_ok) ? 0 : 1;
}
