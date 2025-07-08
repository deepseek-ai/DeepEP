# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR})

add_subdirectory(kernels)

# Link CPP and CUDA together
pybind11_add_module(deep_ep_cpp deep_ep.cpp)
target_link_libraries(deep_ep_cpp PRIVATE ${EP_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
