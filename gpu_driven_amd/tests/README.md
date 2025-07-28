## GPU Driven Test Scripts - HIP Version

Below are the test scripts for investigating the feasibility of using multiple CPU proxy processes for GPUDirect communication on AMD GPUs using HIP.

### Converted Test Files

All test files have been converted from CUDA to HIP:

1. **pcie_bench.hip** - PCIe bandwidth benchmarking tool
2. **gpu_to_cpu_bench.hip** - GPU to CPU memory transfer benchmark  
3. **batched_gpu_to_cpu_bench.hip** - Batched GPU to CPU transfer benchmark

