#include <CL/cl.h>
#include <iostream>
#include <string>
#include <vector>

static std::string device_type_str(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU)         return "GPU";
    if (type & CL_DEVICE_TYPE_CPU)         return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
    return "Unknown";
}

int main() {
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);

    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    std::cout << "Found " << num_platforms << " OpenCL platform(s):\n\n";

    for (const auto& platform : platforms) {
        char platform_name[256] = {};
        char platform_version[256] = {};
        clGetPlatformInfo(platform, CL_PLATFORM_NAME,    sizeof(platform_name),    platform_name,    nullptr);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);

        std::cout << "Platform : " << platform_name << "\n";
        std::cout << "Version  : " << platform_version << "\n";

        cl_uint num_devices = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        for (const auto& device : devices) {
            char   dev_name[256] = {};
            char   dev_version[256] = {};
            cl_uint  compute_units  = 0;
            size_t   max_wg_size    = 0;
            cl_ulong local_mem      = 0;
            cl_ulong global_mem     = 0;
            cl_device_type dev_type = 0;

            clGetDeviceInfo(device, CL_DEVICE_NAME,               sizeof(dev_name),      dev_name,      nullptr);
            clGetDeviceInfo(device, CL_DEVICE_VERSION,            sizeof(dev_version),   dev_version,   nullptr);
            clGetDeviceInfo(device, CL_DEVICE_TYPE,               sizeof(dev_type),      &dev_type,     nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,  sizeof(compute_units), &compute_units,nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(max_wg_size),   &max_wg_size,  nullptr);
            clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,     sizeof(local_mem),     &local_mem,    nullptr);
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,    sizeof(global_mem),    &global_mem,   nullptr);

            std::cout << "\n  Device         : " << dev_name
                      << "\n  Type           : " << device_type_str(dev_type)
                      << "\n  OpenCL version : " << dev_version
                      << "\n  Compute units  : " << compute_units
                      << "\n  Max WG size    : " << max_wg_size
                      << "\n  Local memory   : " << (local_mem  / 1024) << " KB"
                      << "\n  Global memory  : " << (global_mem / (1024 * 1024)) << " MB\n";
        }
        std::cout << "\n";
    }

    return 0;
}
