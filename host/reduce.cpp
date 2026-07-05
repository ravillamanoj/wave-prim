#include "waveprim/reduce.h"

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

float reduce(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     data,
    size_t           n,
    ReduceOp         op,
    size_t           wg_size
) {
    assert(n > 0 && "Cannot reduce an empty array");
    assert((wg_size & (wg_size - 1)) == 0 && "wg_size must be a power of 2");

    const char* kernel_name = (op == ReduceOp::Sum) ? "reduce_sum"
                            : (op == ReduceOp::Min) ? "reduce_min"
                                                    : "reduce_max";

    size_t global_size = ((n + wg_size - 1) / wg_size) * wg_size;
    size_t num_groups  = global_size / wg_size;

    cl_int err;

    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n * sizeof(float), const_cast<float*>(data), &err);
    if (err) throw std::runtime_error("clCreateBuffer input failed");

    cl_mem d_partial = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                      num_groups * sizeof(float), nullptr, &err);
    if (err) throw std::runtime_error("clCreateBuffer partial failed");

    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (err) throw std::runtime_error(std::string("clCreateKernel failed: ") + kernel_name);

    cl_int n_int = static_cast<cl_int>(n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem),  &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem),  &d_partial);
    clSetKernelArg(kernel, 2, wg_size * sizeof(float), nullptr); // local memory
    clSetKernelArg(kernel, 3, sizeof(cl_int),  &n_int);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                                 &global_size, &wg_size, 0, nullptr, nullptr);
    if (err) throw std::runtime_error("clEnqueueNDRangeKernel failed");

    std::vector<float> partial(num_groups);
    clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0,
                        num_groups * sizeof(float), partial.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_partial);

    // Second-pass aggregation on CPU
    if (op == ReduceOp::Sum) return std::accumulate(partial.begin(), partial.end(), 0.0f);
    if (op == ReduceOp::Min) return *std::min_element(partial.begin(), partial.end());
    return *std::max_element(partial.begin(), partial.end());
}

} // namespace waveprim


// ---------------------------------------------------------------------------
// Standalone driver — validates reduce against CPU ground truth
// ---------------------------------------------------------------------------

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    const std::string kernel_path = (argc > 1) ? argv[1] : "kernels/reduce.cl";

    // --- OpenCL setup ---
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

    // --- Test data ---
    const size_t N = 1024;
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = static_cast<float>(i + 1);

    float cpu_sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float cpu_min = *std::min_element(data.begin(), data.end());
    float cpu_max = *std::max_element(data.begin(), data.end());

    float gpu_sum = waveprim::reduce(ctx, queue, program, data.data(), N, waveprim::ReduceOp::Sum);
    float gpu_min = waveprim::reduce(ctx, queue, program, data.data(), N, waveprim::ReduceOp::Min);
    float gpu_max = waveprim::reduce(ctx, queue, program, data.data(), N, waveprim::ReduceOp::Max);

    auto check = [](const char* label, float cpu, float gpu) {
        float diff = std::fabs(cpu - gpu) / (std::fabs(cpu) + 1e-6f);
        bool ok = diff < 1e-4f;
        std::cout << label << ": cpu=" << cpu << "  gpu=" << gpu
                  << "  " << (ok ? "PASS" : "FAIL") << "\n";
        return ok;
    };

    bool all_pass = true;
    all_pass &= check("sum", cpu_sum, gpu_sum);
    all_pass &= check("min", cpu_min, gpu_min);
    all_pass &= check("max", cpu_max, gpu_max);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return all_pass ? 0 : 1;
}
