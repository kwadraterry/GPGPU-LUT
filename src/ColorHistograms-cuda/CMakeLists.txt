### CMakeLists.txt for ColorHistograms-cuda project

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
find_package(PNG REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --cudart static 
    --relocatable-device-code=false 
    -gencode arch=compute_35,code=compute_35 
    -gencode arch=compute_35,code=sm_35 
    )

find_path(CUDA_COMMON_INCLUDE_DIRS
  helper_cuda.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
    "/usr/local/cuda-8.0"
    "/Developer/NVIDIA/CUDA-8.0"
    "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v8.0/common/inc"
  PATH_SUFFIXES "samples/common/inc"
  DOC "Location of helper_cuda.h"
  NO_DEFAULT_PATH
)
MESSAGE(STATUS("CUDA_COMMON_INCLUDE_DIRS:" ${CUDA_COMMON_INCLUDE_DIRS}))

# Specify include directories
include_directories(
    ${CUDA_COMMON_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
)

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


# Specify library paths
target_link_libraries (
    ColorHistogram 
    ${PNG_LIBRARIES} 
    ${CUDA_LIBRARIES}
)