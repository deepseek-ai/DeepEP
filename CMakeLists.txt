# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.18)
project(deep_ep LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-rdynamic")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# NCCL
# TODO: use `find_package` instead of manual checks
if (NOT NCCL_ROOT_DIR)
    message(FATAL_ERROR "NCCL_ROOT_DIR is not set.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(deep_ep/include third-party/deep_jit/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NCCL_ROOT_DIR}/include .)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NCCL_ROOT_DIR}/lib)

# Add kernels
add_subdirectory(csrc)

# Link CPP and CUDA together
pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)

# Enable kernel code indexing with CMake-based IDEs
cuda_add_library(deep_ep_indexing_cuda STATIC csrc/indexing/main.cu)
