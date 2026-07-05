#include "waveprim/gemm.h"

#include <CL/cl.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace waveprim {

static constexpr int TILE_SIZE = 16;

std::vector<float> gemm(
    cl_context       ctx,
    cl_command_queue queue,
    cl_program       program,
    const float*     A,
    const float*     B,
    int              M,
    int              K,
    int              N
) {
    assert(M > 0 && K > 0 && N > 0);

    size_t global_m = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t global_n = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t local[2]  = { (size_t)TILE_SIZE, (size_t)TILE_SIZE };
    size_t global[2] = { global_m, global_n };

    cl_int err;
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)M * K * sizeof(float), const_cast<float*>(A), &err);
    if (err) throw std::runtime_error("clCreateBuffer A failed");

    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)K * N * sizeof(float), const_cast<float*>(B), &err);
    if (err) throw std::runtime_error("clCreateBuffer B failed");

    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                (size_t)M * N * sizeof(float), nullptr, &err);
    if (err) throw std::runtime_error("clCreateBuffer C failed");

    cl_kernel kernel = clCreateKernel(program, "gemm", &err);
    if (err) throw std::runtime_error("clCreateKernel gemm failed");

    size_t tile_bytes = TILE_SIZE * TILE_SIZE * sizeof(float);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &M);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &K);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &N);
    clSetKernelArg(kernel, 6, tile_bytes,     nullptr); // tileA
    clSetKernelArg(kernel, 7, tile_bytes,     nullptr); // tileB

    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                 global, local, 0, nullptr, nullptr);
    if (err) throw std::runtime_error("clEnqueueNDRangeKernel gemm failed");

    std::vector<float> C((size_t)M * N);
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                        C.size() * sizeof(float), C.data(), 0, nullptr, nullptr);

    clReleaseKernel(kernel);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_A);

    return C;
}

} // namespace waveprim


// ---------------------------------------------------------------------------
// Standalone driver — validates GEMM against CPU ground truth
// ---------------------------------------------------------------------------

static std::string load_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    const std::string kernel_path = (argc > 1) ? argv[1] : "kernels/gemm.cl";

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

    // Test 1: square matrix (multi-tile)
    const int M = 33, K = 29, N = 37; // all non-multiples of 16
    std::vector<float> A(M * K), B(K * N);
    for (int i = 0; i < M * K; ++i) A[i] = (i % 5) - 2.0f;
    for (int i = 0; i < K * N; ++i) B[i] = (i % 7) - 3.0f;

    // CPU reference
    std::vector<float> cpu_C(M * N, 0.0f);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            for (int k = 0; k < K; ++k)
                cpu_C[r * N + c] += A[r * K + k] * B[k * N + c];

    std::vector<float> gpu_C = waveprim::gemm(ctx, queue, program,
                                               A.data(), B.data(), M, K, N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::fabs(cpu_C[i] - gpu_C[i]));

    bool ok = max_err < 1e-2f;
    std::cout << "GEMM " << M << "x" << K << " * " << K << "x" << N
              << ": max_err=" << max_err << "  " << (ok ? "PASS" : "FAIL") << "\n";

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ok ? 0 : 1;
}
