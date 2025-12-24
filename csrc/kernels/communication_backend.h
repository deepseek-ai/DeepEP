#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace deep_ep {
namespace internode {

// Backend type enumeration
enum class BackendType {
    NCCL,  // NCCL backend with GIN (GPU-Initiated RDMA) support
    AUTO   // Auto-select based on environment
};

/**
 * Abstract communication backend interface for internode communication.
 *
 * LIFECYCLE AND CALLING ORDER:
 *
 * 1. INITIALIZATION PHASE (on rank 0):
 *    - get_unique_id() - Generate a unique ID for this communication group
 *    - Broadcast unique ID to all ranks (via MPI, PyTorch Distributed, etc.)
 *
 * 2. INITIALIZATION PHASE (on all ranks):
 *    - init() - Initialize the backend with the broadcast unique ID
 *             - Must be called with identical parameters on all ranks
 *             - Blocks until all ranks join
 *    - barrier() - Optional synchronization after init
 *
 * 3. OPERATION PHASE (any order, called as needed):
 *    - alloc() / free() - Memory management for registered buffers
 *    - barrier() - Synchronize all ranks
 *    - get_rank() / get_num_ranks() / get_backend_type() - Query backend state
 *
 * 4. CLEANUP PHASE:
 *    - finalize() - Clean up all resources
 *                 - Must be called on all ranks before destruction
 *                 - No other methods should be called after finalize()
 *
 * GLOBAL BACKEND MANAGEMENT:
 * - initialize_backend() - Creates and initializes the global singleton backend
 * - get_backend() - Retrieves pointer to the global backend (returns nullptr if not initialized)
 * - finalize_backend() - Destroys the global backend singleton
 *
 * THREAD SAFETY: Not thread-safe. All operations must be serialized by the caller.
 */
class CommunicationBackend {
public:
    virtual ~CommunicationBackend() = default;

    // ===== Initialization/Cleanup =====

    // Initialize backend. Must be called on all ranks with same unique_id. Returns local rank.
    virtual int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) = 0;

    // Clean up all resources. Must be called on all ranks before destruction.
    virtual void finalize() = 0;

    // Generate unique ID for communication group. Typically called only on rank 0.
    virtual void get_unique_id(void* unique_id) = 0;

    // ===== Synchronization =====

    // Barrier synchronization across all ranks. Blocks until all ranks reach this point.
    virtual void barrier() = 0;

    // ===== Memory Management =====

    // Allocate registered memory for RDMA. Returns device pointer.
    virtual void* alloc(size_t size, size_t alignment) = 0;

    // Free registered memory previously allocated with alloc().
    virtual void free(void* ptr) = 0;

    // ===== Query Methods =====

    // Get local rank ID (0-based).
    virtual int get_rank() const = 0;

    // Get total number of ranks in the communication group.
    virtual int get_num_ranks() const = 0;

    // Get backend type.
    virtual BackendType get_backend_type() const = 0;

    // Factory method
    static std::unique_ptr<CommunicationBackend> create(BackendType type);
};

// ===== Backend Selection Utilities =====

// Parse backend type from string (e.g., "nccl", "auto").
BackendType parse_backend_type(const std::string& backend_str);

// Auto-detect backend type from environment (checks DEEP_EP_BACKEND env var).
BackendType detect_backend_type();

// Convert backend type enum to string for logging/debugging.
std::string backend_type_to_string(BackendType type);

// ===== Global Backend Management (Singleton Pattern) =====

/**
 * Initialize the global backend singleton.
 * Creates backend instance and calls init() on it.
 * Must be called before any other backend operations.
 * Thread-unsafe: caller must ensure single-threaded access.
 */
void initialize_backend(
    BackendType type, const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank);

/**
 * Get pointer to the global backend singleton.
 * Returns nullptr if initialize_backend() hasn't been called yet.
 * The returned pointer is valid until finalize_backend() is called.
 */
CommunicationBackend* get_backend();

/**
 * Destroy the global backend singleton.
 * Calls finalize() on the backend and releases all resources.
 * After this call, get_backend() will return nullptr.
 */
void finalize_backend();

}  // namespace internode
}  // namespace deep_ep