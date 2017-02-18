### CMakeLists.txt for ColorHistograms-cuda project

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static 
    --relocatable-device-code=false 
    -gencode arch=compute_35,code=compute_35 
    -gencode arch=compute_35,code=sm_35 
    )

# Specify include directories
include_directories(
    src
    /usr/include/libpng12
    /usr/local/include
    /usr/local/cuda-8.0/include
    /usr/local/cuda-8.0/samples/common/inc
    )

# Specify library paths
link_directories(
    /usr/lib/x86_64-linux-gnu/
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    ColorHistogram 
    src/main.cpp
    src/histogram_common.h
    src/histogram_cpu.h
    src/histogram_gpu.h
    src/imageio.h
    src/histogram_cpu.cpp
    src/histogram_gpu.cu
    src/imageio.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    ColorHistogram
    -lpng12
    )