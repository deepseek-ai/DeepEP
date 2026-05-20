#pragma once

#include <cuda.h>
#include <nccl.h>

#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

#include "../../utils/lazy_driver.hpp"

namespace deep_ep::symmetric {

static constexpr int64_t kNumAlignmentBytes = 2097152;

// CPU communicator types: (pid, fd) handle and list of handles from all ranks
using cpu_handle_t = std::pair<int, int>;
using cpu_comm_t = std::vector<cpu_handle_t>;

struct DeviceContext {
    int device_idx;
    CUdevice device;
    int numa_idx;

    DeviceContext() {
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
        CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_idx));
        CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&numa_idx, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
    }

    CUmemAllocationProp gpu_alloc_prop() const { return build_alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device_idx); }
    CUmemAllocationProp cpu_alloc_prop() const { return build_alloc_prop(CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_idx); }

private:
    CUmemAllocationProp build_alloc_prop(const CUmemLocationType& location_type, const int& location_idx) const {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = location_type;
        prop.location.id = location_idx;

        int requested_handle_types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        int flag = 0;
#if CUDART_VERSION >= 13000
        CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
        if (flag)
            requested_handle_types |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif
        prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(requested_handle_types);

        if (location_type == CU_MEM_LOCATION_TYPE_DEVICE) {
            flag = 0;
#if CUDART_VERSION >= 13000
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, device));
#else
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, device));
#endif
            EP_HOST_ASSERT(flag and "GPUDirect RDMA with CUDA VMM is not supported on this device");
            prop.allocFlags.gpuDirectRDMACapable = 1;
        }

        // Check granularity
        size_t num_granularity_bytes = 0;
        CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&num_granularity_bytes, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        EP_HOST_ASSERT((num_granularity_bytes == 0 or kNumAlignmentBytes % num_granularity_bytes == 0) and
                       "Alignment must be a multiple of CUDA allocation granularity");
        return prop;
    }
};

// Try cuMemCreate with FABRIC fallback (mirrors ncclMemAlloc logic)
static void cumem_create_with_fallback(CUmemGenericAllocationHandle* handle,
                                       const int64_t& num_bytes, CUmemAllocationProp* prop) {
    if (prop->requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
        CUresult err = lazy_cuMemCreate(handle, num_bytes, prop, 0);
        if (err == CUDA_ERROR_NOT_PERMITTED or err == CUDA_ERROR_NOT_SUPPORTED) {
            prop->requestedHandleTypes = static_cast<CUmemAllocationHandleType>(
                prop->requestedHandleTypes & ~CU_MEM_HANDLE_TYPE_FABRIC);
            CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0));
        } else {
            CUDA_DRIVER_CHECK(err);
        }
    } else {
        CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0));
    }
}

// Set read/write access for all peer GPUs, and optionally a NUMA node
static void set_access(const CUdeviceptr& addr, const int64_t& num_bytes,
                       const int& device_idx, const int& numa_idx = -1) {
    const bool with_cpu = (numa_idx >= 0);
    CUmemAccessDesc desc[2];
    desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (with_cpu) {
        desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        desc[1].location.id = numa_idx;
        desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    int num_devices;
    CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&num_devices));
    for (int i = 0; i < num_devices; ++ i) {
        int can_access_peer = 0;
        if (i == device_idx or (cudaDeviceCanAccessPeer(&can_access_peer, i, device_idx) == cudaSuccess and can_access_peer)) {
            desc[0].location.id = i;
            CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(addr, num_bytes, desc, with_cpu ? 2 : 1));
        }
    }
}

class SymmetricMemory {
public:
    void* ptr = nullptr;
    int64_t num_bytes = 0;
    int64_t num_gpu_bytes = 0;
    int64_t num_cpu_bytes = 0;

    virtual ~SymmetricMemory() noexcept(0) = default;
};

// Wraps ncclMemAlloc/ncclMemFree (pure GPU, current default)
class GPUSymmetricMemory final : public SymmetricMemory {
public:
    explicit GPUSymmetricMemory(const int64_t& num_bytes) {
        EP_HOST_ASSERT(num_bytes > 0 and num_bytes % kNumAlignmentBytes == 0);
        NCCL_CHECK(ncclMemAlloc(&ptr, num_bytes));
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAlignmentBytes == 0);
        this->num_bytes = num_bytes;
        this->num_gpu_bytes = num_bytes;
    }

    ~GPUSymmetricMemory() override {
        if (ptr != nullptr) {
            NCCL_CHECK(ncclMemFree(ptr));
            ptr = nullptr;
        }
    }
};

// GPU + CPU mixed allocation via CUDA Driver API
// Memory layout: [GPU VRAM (front)] [CPU RAM / NUMA-local (back)]
// The entire contiguous VA range is compatible with `ncclCommWindowRegister`.
class ElasticSymmetricMemory : public SymmetricMemory {
    CUmemGenericAllocationHandle gpu_handle = {};
    CUmemGenericAllocationHandle cpu_handle = {};

public:
    ElasticSymmetricMemory(const int64_t& num_gpu_bytes, const int64_t& num_cpu_bytes) {
        EP_HOST_ASSERT(num_gpu_bytes > 0 and num_gpu_bytes % kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_bytes > 0 and num_cpu_bytes % kNumAlignmentBytes == 0);

        DeviceContext ctx;
        auto gpu_prop = ctx.gpu_alloc_prop();
        auto cpu_prop = ctx.cpu_alloc_prop();

        this->num_gpu_bytes = num_gpu_bytes;
        this->num_cpu_bytes = num_cpu_bytes;
        this->num_bytes = num_gpu_bytes + num_cpu_bytes;

        // Reserve VA and map segments
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, this->num_bytes, kNumAlignmentBytes, 0, 0));
        this->ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));

        cumem_create_with_fallback(&gpu_handle, num_gpu_bytes, &gpu_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_gpu_bytes, 0, gpu_handle, 0));
        set_access(addr, num_gpu_bytes, ctx.device_idx);

        cumem_create_with_fallback(&cpu_handle, num_cpu_bytes, &cpu_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr + num_gpu_bytes, num_cpu_bytes, 0, cpu_handle, 0));
        set_access(addr + num_gpu_bytes, num_cpu_bytes, ctx.device_idx, ctx.numa_idx);

        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAlignmentBytes == 0);
    }

    ~ElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_gpu_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle));
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_gpu_bytes, num_cpu_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(cpu_handle));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, num_bytes));
    }
};

// Each rank creates its own NUMA-local CPU segment;
// CPU segments are imported and mapped contiguously into every rank's VA space.
// Layout: [GPU VRAM (front)] [CPU rank0 | CPU rank1 | ... | CPU rank(N-1) (back)]
class HybridElasticSymmetricMemory final : public SymmetricMemory {
    int num_scaleup_ranks;
    CUmemGenericAllocationHandle gpu_handle = {};
    std::vector<CUmemGenericAllocationHandle> cpu_handles;
    int local_export_fd = -1;

public:
    HybridElasticSymmetricMemory(const cpu_comm_t& cpu_comm,
                                 const int64_t& num_gpu_bytes, const int64_t& num_cpu_bytes,
                                 const int& num_scaleup_ranks, const int& scaleout_rank_idx):
        num_scaleup_ranks(num_scaleup_ranks),
        cpu_handles(num_scaleup_ranks) {
        EP_HOST_ASSERT(num_gpu_bytes > 0 and num_gpu_bytes % kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_bytes > 0 and num_cpu_bytes % kNumAlignmentBytes == 0);

        DeviceContext ctx;
        auto gpu_prop = ctx.gpu_alloc_prop();

        this->num_gpu_bytes = num_gpu_bytes;
        this->num_cpu_bytes = num_cpu_bytes;
        this->num_bytes = num_gpu_bytes + num_cpu_bytes * num_scaleup_ranks;

        // Reserve VA
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, this->num_bytes, kNumAlignmentBytes, 0, 0));
        this->ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));

        // Map GPU segment
        cumem_create_with_fallback(&gpu_handle, num_gpu_bytes, &gpu_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_gpu_bytes, 0, gpu_handle, 0));
        set_access(addr, num_gpu_bytes, ctx.device_idx);

        // Import and map all intra-node CPU segments
        const auto local_pid = getpid();
        for (int i = 0; i < num_scaleup_ranks; ++ i) {
            auto [pid, fd] = cpu_comm[num_scaleup_ranks * scaleout_rank_idx + i];
            int local_fd = fd;
            if (pid != local_pid) {
                int pidfd = syscall(SYS_pidfd_open, pid, 0);
                EP_HOST_ASSERT(pidfd >= 0 and "`pidfd_open` failed");
                local_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0);
                EP_HOST_ASSERT(local_fd >= 0 and "`pidfd_getfd` failed");
                close(pidfd);
            }

            const auto offset = num_gpu_bytes + i * num_cpu_bytes;
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
                &cpu_handles[i],
                reinterpret_cast<void*>(static_cast<uintptr_t>(local_fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(addr + offset, num_cpu_bytes, 0, cpu_handles[i], 0));
            set_access(addr + offset, num_cpu_bytes, ctx.device_idx, ctx.numa_idx);

            if (pid != local_pid) {
                close(local_fd);
            } else {
                local_export_fd = local_fd;
            }
        }

        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAlignmentBytes == 0);
    }

    ~HybridElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));

        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_gpu_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle));

        if (local_export_fd >= 0)
            close(local_export_fd);
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_gpu_bytes, num_cpu_bytes * num_scaleup_ranks));
        for (int i = 0; i < num_scaleup_ranks; ++ i)
            CUDA_DRIVER_CHECK(lazy_cuMemRelease(cpu_handles[i]));

        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, num_bytes));
    }

    // Create a NUMA-local CPU segment and export its POSIX FD handle.
    // Returns (pid, fd) handle for cross-process sharing.
    static cpu_handle_t create_cpu_handle(const int64_t& num_cpu_bytes) {
        EP_HOST_ASSERT(num_cpu_bytes > 0 and num_cpu_bytes % kNumAlignmentBytes == 0);

        DeviceContext ctx;
        auto cpu_prop = ctx.cpu_alloc_prop();

        CUmemGenericAllocationHandle handle;
        cumem_create_with_fallback(&handle, num_cpu_bytes, &cpu_prop);

        int fd = -1;
        CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        // Release the allocation handle -- the POSIX FD keeps the physical memory alive
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));

        return {getpid(), fd};
    }
};

static std::shared_ptr<SymmetricMemory> alloc(const int64_t& num_gpu_bytes, const int64_t& num_cpu_bytes,
                                              const bool& allow_hybrid_mode = false,
                                              const int& num_scaleup_ranks = 0, const int& scaleout_rank_idx = 0,
                                              const cpu_comm_t& cpu_comm = {}) {
    EP_HOST_ASSERT(num_gpu_bytes > 0 and num_gpu_bytes % kNumAlignmentBytes == 0);
    EP_HOST_ASSERT(num_cpu_bytes >= 0 and num_cpu_bytes % kNumAlignmentBytes == 0);

    std::shared_ptr<SymmetricMemory> result;
    if (num_cpu_bytes > 0) {
        if (allow_hybrid_mode) {
            result = std::make_shared<HybridElasticSymmetricMemory>(
                cpu_comm, num_gpu_bytes, num_cpu_bytes,
                num_scaleup_ranks, scaleout_rank_idx);
        } else {
            result = std::make_shared<ElasticSymmetricMemory>(
                num_gpu_bytes, num_cpu_bytes);
        }
    } else {
        result = std::make_shared<GPUSymmetricMemory>(num_gpu_bytes);
    }

    // TODO: move all variables into NCCL runtime
    // Enable NCCL elastic buffer register if the allocator may produce CPU-backed segments
    if (dynamic_cast<ElasticSymmetricMemory*>(result.get()) != nullptr)
        setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);
    // Multi-plane: all ranks share CPU segments, skip proxy re-export for sysmem handles
    if (dynamic_cast<HybridElasticSymmetricMemory*>(result.get()) != nullptr)
        setenv("NCCL_SYM_REUSE_SYSMEM_HANDLES", "1", 0);
    return result;
}

}  // namespace deep_ep::symmetric
