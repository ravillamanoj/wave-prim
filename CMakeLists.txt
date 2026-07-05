cmake_minimum_required(VERSION 3.16)
project(wave-prim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

add_executable(device_query host/device_query.cpp)
target_link_libraries(device_query PRIVATE OpenCL::OpenCL)
target_include_directories(device_query PRIVATE include)
