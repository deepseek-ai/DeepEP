#pragma once

#include <cuda.h>
#include <nccl.h>

#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

#include "../../utils/lazy_driver.hpp"

namespace deep_ep::symmetric {

// Shared allocation types: (pid, fd) handle and list of handles from all ranks
using shared_handle_t = std::pair<int, int>;
using shared_comm_t = std::vector<shared_handle_t>;

struct DeviceContext {
    int device_idx;
    CUdevice device;
    int numa_idx;

    DeviceContext() {
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx));
        CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, device_idx));
        CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&numa_idx, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device));
    }

    CUmemAllocationProp gpu_alloc_prop(const bool& skip_fabric = false) const {
        return build_alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device_idx, skip_fabric);
    }

    CUmemAllocationProp cpu_alloc_prop(const bool& skip_fabric = false) const {
        return build_alloc_prop(CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_idx, skip_fabric);
    }

private:
    CUmemAllocationProp build_alloc_prop(const CUmemLocationType& location_type, const int& location_idx, const bool& skip_fabric = false) const {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = location_type;
        prop.location.id = location_idx;

        int requested_handle_types = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        int flag = 0;
#if CUDART_VERSION >= 13000
        if (not skip_fabric) {
            CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
            if (flag)
                requested_handle_types |= CU_MEM_HANDLE_TYPE_FABRIC;
        }
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
        EP_HOST_ASSERT((num_granularity_bytes == 0 or kNumAllocationAlignmentBytes % num_granularity_bytes == 0) and
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

// Set read/write access for local GPU, and optionally a NUMA node
static void set_access(const CUdeviceptr& addr, const int64_t& num_bytes,
                       const int& device_idx, const int& numa_idx = -1) {
    const bool with_cpu = (numa_idx >= 0);
    CUmemAccessDesc desc[2];
    desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc[0].location.id = device_idx;
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (with_cpu) {
        desc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        desc[1].location.id = numa_idx;
        desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(addr, num_bytes, desc, with_cpu ? 2 : 1));
}

class SymmetricMemory {
public:
    void* ptr = nullptr;
    int64_t num_sym_bytes = 0;

    virtual ~SymmetricMemory() noexcept(0) = default;
};

// Wraps ncclMemAlloc/ncclMemFree (pure GPU, current default)
class GPUSymmetricMemory final : public SymmetricMemory {
public:
    explicit GPUSymmetricMemory(const int64_t& num_sym_bytes) {
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0);
        NCCL_CHECK(ncclMemAlloc(&ptr, num_sym_bytes));
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0);
        this->num_sym_bytes = num_sym_bytes;
    }

    ~GPUSymmetricMemory() override {
        if (ptr != nullptr) {
            NCCL_CHECK(ncclMemFree(ptr));
            ptr = nullptr;
        }
    }
};

// GPU main allocation plus a separately backed CPU/GPU storage via CUDA Driver API.
// The two physical handles share one contiguous VA range.
class ElasticSymmetricMemory : public SymmetricMemory {
    int64_t num_rdma_storage_bytes;
    CUmemGenericAllocationHandle gpu_handle = {};
    CUmemGenericAllocationHandle storage_handle = {};

public:
    ElasticSymmetricMemory(const int64_t& num_sym_bytes, const int64_t& num_rdma_storage_bytes,
                           const bool& use_cpu_rdma_storage):
        num_rdma_storage_bytes(num_rdma_storage_bytes) {
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0);

        DeviceContext ctx;
        auto gpu_prop = ctx.gpu_alloc_prop();
        auto storage_prop = use_cpu_rdma_storage ? ctx.cpu_alloc_prop(true) : ctx.gpu_alloc_prop(true);

        this->num_sym_bytes = num_sym_bytes;
        const auto num_allocated_bytes = num_sym_bytes + num_rdma_storage_bytes;

        // Reserve VA and map segments
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, num_allocated_bytes, kNumAllocationAlignmentBytes, 0, 0));
        this->ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));

        cumem_create_with_fallback(&gpu_handle, num_sym_bytes, &gpu_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_sym_bytes, 0, gpu_handle, 0));

        cumem_create_with_fallback(&storage_handle, num_rdma_storage_bytes, &storage_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(
            addr + num_sym_bytes, num_rdma_storage_bytes, 0, storage_handle, 0));

        if (use_cpu_rdma_storage) {
            set_access(addr, num_sym_bytes, ctx.device_idx);
            set_access(addr + num_sym_bytes, num_rdma_storage_bytes,
                       ctx.device_idx, ctx.numa_idx);
        } else {
            set_access(addr, num_allocated_bytes, ctx.device_idx);
        }

        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0);
    }

    ~ElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_sym_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle));
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_sym_bytes, num_rdma_storage_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(storage_handle));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(
            addr, num_sym_bytes + num_rdma_storage_bytes));
    }
};

// Maps the RDMA storage shards exported by intra-node ranks contiguously after the main symmetric-memory segment.
class HybridElasticSymmetricMemory final : public SymmetricMemory {
    int num_scaleup_ranks;
    int64_t num_rdma_storage_bytes;
    CUmemGenericAllocationHandle gpu_handle = {};
    std::vector<CUmemGenericAllocationHandle> storage_handles;
    int local_export_fd = -1;

public:
    HybridElasticSymmetricMemory(const shared_comm_t& shared_comm,
                                 const int64_t& num_sym_bytes, const int64_t& num_rdma_storage_bytes,
                                 const bool& use_cpu_rdma_storage,
                                 const int& num_scaleup_ranks, const int& scaleout_rank_idx):
        num_scaleup_ranks(num_scaleup_ranks),
        num_rdma_storage_bytes(num_rdma_storage_bytes),
        storage_handles(num_scaleup_ranks) {
        EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_rdma_storage_bytes > 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0);

        DeviceContext ctx;
        auto gpu_prop = ctx.gpu_alloc_prop();

        this->num_sym_bytes = num_sym_bytes;
        const auto num_mapped_bytes = num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks;

        // Reserve VA
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, num_mapped_bytes, kNumAllocationAlignmentBytes, 0, 0));
        ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
        EP_HOST_ASSERT(reinterpret_cast<uint64_t>(ptr) % kNumAllocationAlignmentBytes == 0);

        // Map GPU segment
        cumem_create_with_fallback(&gpu_handle, num_sym_bytes, &gpu_prop);
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, num_sym_bytes, 0, gpu_handle, 0));

        // Import and map all intra-node storage segments.
        const auto local_pid = getpid();
        for (int i = 0; i < num_scaleup_ranks; ++ i) {
            auto [pid, fd] = shared_comm[num_scaleup_ranks * scaleout_rank_idx + i];
            int local_fd = fd;
            if (pid != local_pid) {
                int pidfd = syscall(SYS_pidfd_open, pid, 0);
                EP_HOST_ASSERT(pidfd >= 0 and "`pidfd_open` failed");
                local_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0);
                EP_HOST_ASSERT(local_fd >= 0 and "`pidfd_getfd` failed");
                close(pidfd);
            }

            const auto offset = num_sym_bytes + i * num_rdma_storage_bytes;
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
                &storage_handles[i],
                reinterpret_cast<void*>(static_cast<uintptr_t>(local_fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(addr + offset, num_rdma_storage_bytes, 0, storage_handles[i], 0));

            if (pid != local_pid) {
                close(local_fd);
            } else {
                local_export_fd = local_fd;
            }
        }

        // Enable access
        if (use_cpu_rdma_storage) {
            set_access(addr, num_sym_bytes, ctx.device_idx);
            set_access(addr + num_sym_bytes, num_rdma_storage_bytes * num_scaleup_ranks,
                       ctx.device_idx, ctx.numa_idx);
        } else {
            set_access(addr, num_mapped_bytes, ctx.device_idx);
        }
    }

    ~HybridElasticSymmetricMemory() override {
        auto addr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));

        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, num_sym_bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(gpu_handle));

        if (local_export_fd >= 0)
            close(local_export_fd);
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr + num_sym_bytes, num_rdma_storage_bytes * num_scaleup_ranks));
        for (int i = 0; i < num_scaleup_ranks; ++ i)
            CUDA_DRIVER_CHECK(lazy_cuMemRelease(storage_handles[i]));

        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks));
    }

    // Create a CPU/GPU storage segment and export its POSIX FD handle.
    // Returns (pid, fd) handle for cross-process sharing.
    static shared_handle_t create_shared_handle(const int64_t& num_bytes,
                                                const bool& use_cpu_rdma_storage) {
        EP_HOST_ASSERT(num_bytes > 0 and num_bytes % kNumAllocationAlignmentBytes == 0);

        DeviceContext ctx;
        auto prop = use_cpu_rdma_storage ? ctx.cpu_alloc_prop(true) : ctx.gpu_alloc_prop(true);

        CUmemGenericAllocationHandle handle;
        cumem_create_with_fallback(&handle, num_bytes, &prop);

        int fd = -1;
        CUDA_DRIVER_CHECK(lazy_cuMemExportToShareableHandle(
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

        // Release the allocation handle -- the POSIX FD keeps the physical memory alive
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(handle));

        return {getpid(), fd};
    }
};

static std::shared_ptr<SymmetricMemory> alloc(const int64_t& num_sym_bytes,
                                              const int64_t& num_rdma_storage_bytes,
                                              const bool& use_cpu_rdma_storage,
                                              const bool& allow_hybrid_mode = false,
                                              const int& num_scaleup_ranks = 0, const int& scaleout_rank_idx = 0,
                                              const shared_comm_t& shared_comm = {}) {
    EP_HOST_ASSERT(num_sym_bytes > 0 and num_sym_bytes % kNumAllocationAlignmentBytes == 0);
    EP_HOST_ASSERT(num_rdma_storage_bytes >= 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0);

    std::shared_ptr<SymmetricMemory> result;
    if (num_rdma_storage_bytes > 0) {
        if (allow_hybrid_mode) {
            result = std::make_shared<HybridElasticSymmetricMemory>(
                shared_comm, num_sym_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage,
                num_scaleup_ranks, scaleout_rank_idx);
        } else {
            result = std::make_shared<ElasticSymmetricMemory>(
                num_sym_bytes, num_rdma_storage_bytes, use_cpu_rdma_storage);
        }
    } else {
        result = std::make_shared<GPUSymmetricMemory>(num_sym_bytes);
    }

    // TODO: move all variables into NCCL runtime
    // Enable NCCL elastic buffer register if the allocator may produce CPU-backed segments
    if (num_rdma_storage_bytes > 0 and use_cpu_rdma_storage)
        setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);
    return result;
}

}  // namespace deep_ep::symmetric
