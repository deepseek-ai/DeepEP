# NCCL GIN Migration Documentation

This document provides an overview of the migration from NVSHMEM to NCCL GIN in the DeepEP framework. Each section details the original NVSHMEM function, its NCCL GIN replacement, and explains the operation's purpose and implementation details.

## Table of Contents

1. [NCCL-GIN Summary](#nccl-gin-summary)
2. [Barrier Operations](#barrier-operations)
3. [Put Operations](#put-operations)
4. [Atomic Operations](#atomic-operations)
5. [Signal Operations](#signal-operations)
6. [Memory Fence Operations](#memory-fence-operations)
7. [Quiet Operations](#quiet-operations)

---

## NCCL-GIN Summary

### Key Design Principles

- **Multiple Communicators**: We use multiple communicators (QPs) to replicate DeepEP's behavior, which uses 12 QPs with 24 SMs in our test configurations
- **Signal-based Atomic Operations**: Atomic operations on head/tail pointers of DeepEP's circular buffers are implemented using signals
- **High-level API Usage**: We use the high-level NCCL GIN device API (`ncclGin` class) for all operations, which provides methods like `put()`, `signal()`, `flush()`, and `readSignal()`
- **Device Communicators**: Operations use `ncclDevComm*` device communicators and `ncclWindow_t` memory windows
- **Per-Channel Context Selection**: Each channel/SM uses a different GIN context (`comm_id = channel_id % num_gin_ctxs`) for load balancing


## Barrier Operations

### Team Barrier

**Purpose**: Synchronizes all processes in the communication team to ensure all operations complete before proceeding.

#### Original NVSHMEM Implementation
```cuda
kLowLatencyMode ? void(nvshmem_barrier(rdma_team)) : nvshmem_barrier_all();
```

#### NCCL GIN Replacement
```cuda
auto dcomm = dcomms[0];
ncclGin net(dcomm, 0);

if (kLowLatencyMode) {
    // Use rank as session ID for symmetric synchronization
    int session_id = dcomm.lsaRank;
    
    // Use GIN barrier session directly with symmetric team
    ncclGinBarrierSession<ncclCoopThread> barrier(
        ncclCoopThread(),
        net,
        ncclTeamTagRail(),
        session_id
    );
    barrier.sync(ncclCoopThread(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
} else {
    // World barrier - synchronizes all ranks
    ncclBarrierSession<ncclCoopThread> barrier(ncclCoopThread(), ncclTeamTagWorld(), net, 0);
    barrier.sync(ncclCoopThread(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
}
```

#### Implementation Details
- **Low Latency Mode**: Uses symmetric team barrier with `session_id = dcomm.lsaRank` for symmetric RDMA ranks
- **Standard Mode**: Uses a barrier that synchronizes all ranks
- **Communicator Selection**: Always uses the first communicator (`dcomms[0]`) for simplicity

---

## Put Operations

### Non-blocking Integer Put

**Purpose**: Transfers token counts as integers to symmetric RDMA ranks using non-blocking operations.

#### Original NVSHMEM Implementation
```cuda
nvshmem_int_put_nbi(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank), 
                    rdma_recv_num_tokens_mixed.send_buffer(thread_id),
                    NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                    translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank));
```

#### NCCL GIN Replacement
```cuda
// GIN put -- only one thread executes it
if (lane_id == 0) {
    // Distribute work across GIN contexts (when sm_id == 0, use first communicator)
    auto comm_id = i % num_gin_ctxs;
    int dst_rank = translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank);
    size_t src_offset = reinterpret_cast<size_t>(rdma_recv_num_tokens_mixed.send_buffer(i)) -
                    reinterpret_cast<size_t>(gin_base_ptr);
    size_t dst_offset = reinterpret_cast<size_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)) -
                    reinterpret_cast<size_t>(gin_base_ptr);
    size_t bytes = (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int);

    ncclGin net(dcomms[comm_id], 0);
    ncclTeam world = ncclTeamWorld(dcomms[comm_id]);
    ncclWindow_t nccl_window = nccl_windows[comm_id];
    net.put(
        world, dst_rank,
        nccl_window, dst_offset,
        nccl_window, src_offset,
        bytes,
        ncclGin_None{},             // no signal
        ncclGin_None{},             // no counter
        ncclCoopThread()
    );
}
__syncwarp(); // Synchronize all warp threads after the operation
```

#### Implementation Details

- **Single-threaded Execution**: Only lane 0 executes the put operation (not warp-collective)
- **Communicator Selection**: Distributes work across GIN contexts (when `sm_id == 0`, uses first communicator)
- **High-level API**: Uses `ncclGin::put()` method with `ncclTeamWorld` and memory windows
- **Data Transfer**: Transfers integer token counts between local and remote buffers
- **Warp Synchronization**: Uses `__syncwarp()` after the operation

### Warp-level Non-blocking Put

**Purpose**: Performs warp-level non-blocking put operations for channel metadata with immediate submission behavior.

#### Original NVSHMEM Implementation
The `nvshmemi_ibgda_put_nbi_warp` function is a warp-level non-blocking RDMA put operation that transfers data from a local buffer to a remote buffer. When `kAlwaysDoPostSend` is set to true, it affects the submission behavior of RDMA work queue entries (WQEs):

- **Immediate Submission**: Always submits RDMA requests immediately after preparing WQEs
- **Producer Index Update**: Updates the producer index (`prod_idx`) immediately
- **Doorbell Ringing**: Notifies InfiniBand hardware to send operations to the network immediately

```cuda
nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                  channel_id, lane_id, 0);
```

#### NCCL GIN Replacement
When using multiple communicators (QPs), we use `channel_id` (`channel_id = sm_id/2`) to select the appropriate communicator.

```cuda
if (lane_id == 0) {  // Only execute on lane 0 to match nvshmemi_ibgda_put_nbi_warp behavior
    // Use a different GIN context and window for each channel/SM
    auto comm_id = channel_id % num_gin_ctxs;
    int dst_rank = translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank);
    size_t src_offset = reinterpret_cast<size_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)) -
                    reinterpret_cast<size_t>(gin_base_ptr);
    size_t dst_offset = reinterpret_cast<size_t>(rdma_channel_meta.recv_buffer(rdma_rank)) -
                    reinterpret_cast<size_t>(gin_base_ptr);
    size_t bytes = sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2);

    ncclGin net(dcomms[comm_id], 0);
    ncclTeam world = ncclTeamWorld(dcomms[comm_id]);
    ncclWindow_t nccl_window = nccl_windows[comm_id];
    net.put(
        world, dst_rank,
        nccl_window, dst_offset,
        nccl_window, src_offset,
        bytes,
        ncclGin_None{},             // no signal
        ncclGin_None{},             // no counter
        ncclCoopThread()
    );
}
__syncwarp(); // Synchronize all warp threads
```

#### Implementation Details
- **Single-threaded Execution**: Only lane 0 executes the put operation (not warp-collective in NCCL GIN)
- **Communicator Selection**: Maps `channel_id` to different communicators for load balancing
- **High-level API**: Uses `ncclGin::put()` method with `ncclTeamWorld` and memory windows
- **Warp Synchronization**: Uses `__syncwarp()` to maintain warp-level semantics

---

## Atomic Operations

### Atomic Add for Tail Updates

**Purpose**: Performs atomic addition on remote memory locations to update tail pointers in circular buffers.

#### Original NVSHMEM Implementation
```cuda
nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank), num_tokens_to_issue,
                                translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), 
                                channel_id, dst_rdma_rank == rdma_rank);
```

#### NCCL GIN Replacement
```cuda
// Use a different GIN context for each channel/SM
auto comm_id = channel_id % num_gin_ctxs;
auto dst_rank = translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank);
auto signal_id = gin_signals_tail + rdma_rank;

ncclGin net(dcomms[comm_id], 0);
ncclTeam world = ncclTeamWorld(dcomms[comm_id]);
net.signal(
    world,                                          // team
    dst_rank,                                       // destination rank
    ncclGin_SignalAdd{signal_id, (uint64_t)num_tokens_to_issue},  // signal + value
    ncclCoopThread(),                               // cooperation scope (default)
    ncclGin_None{},                                 // no descriptor (default)
    cuda::thread_scope_thread,                      // alreadyReleased (default)
    cuda::thread_scope_thread                       // expected_scope (default)
);
```

#### Implementation Details
- **Signal-based Atomic Operations**: Uses `ncclGin::signal()` method with `ncclGin_SignalAdd` to achieve atomic addition
- **Tail Pointer Management**: Updates tail pointers to track available buffer space
- **Communicator Mapping**: Maps `channel_id` to appropriate communicator (QP)
- **No Data Transfer**: Only performs signal operation without transferring data
- **High-level API**: Uses the high-level signal API for atomic operations

### Atomic Add for Head Updates

**Purpose**: Updates head pointers atomically to indicate that buffer space has been freed.

#### Original NVSHMEM Implementation
```cuda
nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank), min_head - last_head,
                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank), 
                                channel_id, lane_id == rdma_rank);
```

#### NCCL GIN Replacement
```cuda
// Use a different GIN context for each channel/SM
auto comm_id = channel_id % num_gin_ctxs;
auto dst_rank = translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank);
auto signal_id = gin_signals_head + rdma_rank;

ncclGin net(dcomms[comm_id], 0);
ncclTeam world = ncclTeamWorld(dcomms[comm_id]);
net.signal(
    world,                                          // team
    dst_rank,                                       // destination rank
    ncclGin_SignalAdd{signal_id, (uint64_t) min_head - (uint64_t) last_head},  // signal + value
    ncclCoopThread(),                               // cooperation scope (default)
    ncclGin_None{},                                 // no descriptor (default)
    cuda::thread_scope_thread,                      // alreadyReleased (default)
    cuda::thread_scope_thread                       // expected_scope (default)
);
```

#### Implementation Details
- **Head Pointer Updates**: Communicates head updates to remote ranks using signal operations
- **Buffer Space Management**: Indicates freed buffer space for reuse
- **Signal-based Communication**: Uses `ncclGin::signal()` method with `ncclGin_SignalAdd`
- **Communicator Selection**: Maps `channel_id` to appropriate communicator (QP)
- **High-level API**: Uses the high-level signal API for atomic operations

---

## Signal Operations

### Signal Layout and Management

**Purpose**: NCCL GIN uses signals to implement atomic operations on head/tail pointers of circular buffers. The signals are organized as follows:

- **Signal Base**: Each buffer type (e.g., internode buffer with index 2) has a base signal ID obtained via `backend->get_signals_base(internode_buffer_idx)`
- **Head Signals**: Located at `signals_base + (channel_id * kNumRDMARanks) + rank_id`
- **Tail Signals**: Located at `signals_base + (kNumRDMARanks * num_channels) + (channel_id * kNumRDMARanks) + rank_id`

This layout ensures that each rank's head and tail pointers for each channel have unique signal IDs.

### Signal Reset

**Purpose**: Resets all head/tail pointers to zero at the beginning of each operation to ensure clean state.

#### Original NVSHMEM Implementation
```cuda
#pragma unroll
for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
    rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
```

#### NCCL GIN Replacement
```cuda
// For each channel we have kNumRDMARanks head and tail signals
int num_signals = kNumRDMARanks * num_channels * 2; 
EP_DEVICE_ASSERT(num_signals <= num_threads);

// Each thread handles one specific signal across all contexts
if (thread_id < num_signals) {
    auto signal_id = signals_base + thread_id;

    // Derive channel_id from signal_id
    // Signal layout: [all head signals][all tail signals]
    int signal_offset = thread_id;
    int head_signal_count = kNumRDMARanks * num_channels;
    int channel_id = (signal_offset < head_signal_count)
        ? signal_offset / kNumRDMARanks
        : (signal_offset - head_signal_count) / kNumRDMARanks;
    // Only reset for the context assigned to this channel
    auto comm_id = channel_id % num_gin_ctxs;
    ncclGin net(dcomms[comm_id], 0);
    net.resetSignal(signal_id);
}
__syncthreads();
```

#### Implementation Details
- **Signal Range**: Each channel has `kNumRDMARanks` head and tail signals
- **Thread Distribution**: Each thread handles one specific signal
- **Channel-specific Reset**: Only resets signal for the communicator assigned to that channel
- **Signal Layout**: Head signals come first, followed by tail signals
- **High-level API**: Uses `ncclGin::resetSignal()` method
- **Synchronization**: Uses `__syncthreads()` to ensure all resets complete

### Signal Reading

**Purpose**: Reads the current value of a signal atomically with proper memory ordering.

#### Original NVSHMEM Implementation
```cuda
cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
```

#### NCCL GIN Replacement
```cuda
// Use different GIN context for each channel/SM
auto comm_id = channel_id % num_gin_ctxs;
auto signal_id = gin_signals_head + lane_id;
ncclGin net(dcomms[comm_id], 0);
uint64_t signal_value = net.readSignal(signal_id);
cached_rdma_channel_head = static_cast<int>(signal_value);
```

#### Implementation Details
- **Context Selection**: Uses different GIN context for each channel/SM
- **Signal Reading**: Uses `ncclGin::readSignal()` method to read signal value
- **High-level API**: Simple method call replaces low-level pointer operations
- **Atomic Operations**: Memory ordering handled internally by the API

---

## Memory Fence Operations

### Memory Fence

**Purpose**: Ensures memory ordering and system-wide visibility of previous operations across all processing elements.

#### Original NVSHMEM Implementation
`nvshmem_fence()` is a memory ordering operation that ensures all previous NVSHMEM operations complete in order before any subsequent 
operations begin. Internally, it works by iterating through all initialized transport layers and performing two types of synchronization: (1) For transports 
without endpoints (like shared memory), it synchronizes all CUDA streams using cudaStreamSynchronize() to ensure GPU operations complete, and (2) For transports 
with endpoints (like InfiniBand or UCX), it calls the transport-specific fence operation for each processing element (PE) to ensure proper ordering of network 
operations. The function essentially acts as a barrier that guarantees that all prior memory operations have been completed and are visible to other PEs before 
any subsequent operations can proceed, providing the necessary memory consistency guarantees for distributed shared memory operations across multiple GPUs and 
nodes.
```cuda
nvshmem_fence();
```

#### NCCL GIN Replacement
```cuda
__threadfence_system();  // Memory barrier
```

#### Implementation Details
- **System-wide Fence**: Uses `__threadfence_system()` for system-wide memory visibility

#### Equivalence Note
The NCCL GIN implementation may not be fully equivalent to NVSHMEM's fence operation, as it uses a different synchronization strategy. Further testing may be required to verify complete functional equivalence.

---

## Quiet Operations

### Quiet Operation

**Purpose**: Ensures all previously issued RDMA operations (puts, atomics, etc.) have completed and their effects are visible to remote PEs. This is critical when reusing buffers to prevent data corruption from in-flight operations.

#### Original NVSHMEM Implementation
`nvshmem_quiet()` is a completion and memory ordering operation that ensures all previously issued NVSHMEM operations (put, get, atomic operations, etc.) targeting all processing elements (PEs) have completed and their effects are visible to remote PEs. Unlike `nvshmem_fence()` which only ensures ordering, 
```cuda
nvshmem_quiet();
```

#### NCCL GIN Replacement
```cuda
// Flush all contexts to ensure all previous inflight operations complete
EP_DEVICE_ASSERT(num_gin_ctxs <= num_threads);
if (thread_id < num_gin_ctxs) {
    auto comm_id = thread_id;
    ncclGin net(dcomms[comm_id], 0);
    net.flush(ncclCoopThread(), cuda::std::memory_order_acquire);
}
__syncthreads();
```

#### Implementation Details
- **Multi-context Flush**: Each thread flushes one GIN context to parallelize the operation
- **Thread Distribution**: Distributes flush operations across threads (one context per thread)
- **High-level API**: Uses `ncclGin::flush()` method for completion
- **Memory Ordering**: Uses `memory_order_acquire` to ensure all previous operations are visible
- **Synchronization**: Uses `__syncthreads()` to ensure all contexts have been flushed before proceeding
- **Buffer Safety**: Critical for preventing data corruption when reusing RDMA buffers

#### Use Case
This operation is typically used before clearing or reusing RDMA buffers to ensure that all previous writes to those buffers have completed. In the DeepEP framework, it's used to wait for all inflight work requests to complete before rewriting the cleared `rdma_buffer`.

---
