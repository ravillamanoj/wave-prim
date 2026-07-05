"""Sanity checks that verify the development environment is correctly set up."""

import numpy as np
import pyopencl as cl


def test_numpy_available():
    arr = np.array([1, 2, 3], dtype=np.float32)
    assert arr.sum() == 6.0


def test_opencl_platform_available():
    platforms = cl.get_platforms()
    assert len(platforms) > 0, "No OpenCL platforms found — is POCL installed?"


def test_opencl_device_available():
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        devices.extend(p.get_devices())
    assert len(devices) > 0, "No OpenCL devices found"


def test_opencl_context_creation():
    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    ctx = cl.Context([device])
    assert ctx is not None


def test_opencl_simple_kernel():
    """Compile and run a trivial OpenCL kernel to confirm end-to-end execution works."""
    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    kernel_src = """
    __kernel void identity(__global float* out, __global const float* in, int n) {
        int i = get_global_id(0);
        if (i < n) out[i] = in[i];
    }
    """

    program = cl.Program(ctx, kernel_src).build()

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mf = cl.mem_flags
    buf_in  = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=data)
    buf_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    program.identity(queue, (len(data),), None, buf_out, buf_in, np.int32(len(data)))

    result = np.empty_like(data)
    cl.enqueue_copy(queue, result, buf_out)
    queue.finish()

    np.testing.assert_array_equal(result, data)
