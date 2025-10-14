# NCCL GIN Migration Documentation

This document provides a comprehensive overview of the migration from NVSHMEM to NCCL GIN in the DeepEP framework. Each section details the original NVSHMEM function, its NCCL GIN replacement, and explains the operation's purpose and implementation details.

## Table of Contents

1. [NCCL GIN Summary](#nccl-gin-summary)
2. [Barrier Operations](#barrier-operations)
3. [Put Operations](#put-operations)
4. [Atomic Operations](#atomic-operations)
5. [Signal Operations](#signal-operations)
6. [Quiet Operations](#quiet-operations)

---

## NCCL GIN Summary

### Key Design Principles

- **Multiple Communicators**: Multiple communicators (QPs) are used to replicate DeepEP's behavior, which uses 12 QPs with 24 SMs in our test configurations

- **Signal-based Atomic Operations**: Atomic operations on head/tail pointers of DeepEP's circular buffers are implemented using NCCL GIN signals

- **High-level API Usage**: All operations use the high-level NCCL GIN device API (`ncclGin` class), which provides methods including:
  - `put()` for data transfer
  - `signal()` for atomic operations
  - `flush()` for completion
  - `readSignal()` for reading signal values

- **Device Communicators**: Operations use `ncclDevComm*` device communicators and `ncclWindow_t` memory windows for RDMA operations

- **Per-Channel Context Selection**: Each channel/SM uses a different GIN context (`comm_id = channel_id % num_gin_ctxs`) for optimal load balancing across QPs


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

- **Low Latency Mode**: Uses symmetric team barrier (`ncclGinBarrierSession`) with `session_id = dcomm.lsaRank` for symmetric RDMA ranks

- **Standard Mode**: Uses world barrier (`ncclBarrierSession`) that synchronizes all ranks

- **Communicator Selection**: Always uses the first communicator (`dcomms[0]`) for simplicity

- **Memory Ordering**: Uses relaxed memory ordering and fence level for performance

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

- **Single-threaded Execution**: Only lane 0 executes the put operation (not warp-collective in NCCL GIN)

- **Communicator Selection**: Distributes work across GIN contexts using `comm_id = i % num_gin_ctxs` for load balancing

- **High-level API**: Uses `ncclGin::put()` method with `ncclTeamWorld` and memory windows

- **Offset Calculation**: Computes source and destination offsets relative to `gin_base_ptr`

- **Data Transfer**: Transfers `(NUM_MAX_NVL_PEERS + num_rdma_experts + 1)` integer token counts

- **No Signaling**: Put operation completes without signals or counters (`ncclGin_None{}`)

- **Warp Synchronization**: Uses `__syncwarp()` to synchronize all warp threads after the operation

### Warp-level Non-blocking Put

**Purpose**: Performs warp-level non-blocking put operations for channel metadata with immediate submission behavior.

#### Original NVSHMEM Implementation

The `nvshmemi_ibgda_put_nbi_warp` function is a warp-level non-blocking RDMA put operation that transfers data from a local buffer to a remote buffer. When `kAlwaysDoPostSend` is set to `true`, it enables immediate submission behavior:

- **Immediate Submission**: RDMA requests are submitted immediately after preparing work queue entries (WQEs)
- **Producer Index Update**: The producer index (`prod_idx`) is updated immediately
- **Doorbell Ringing**: InfiniBand hardware is notified immediately to send operations to the network

```cuda
nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                  channel_id, lane_id, 0);
```

#### NCCL GIN Replacement

When using multiple communicators (QPs), the `channel_id` (computed as `channel_id = sm_id / 2`) is used to select the appropriate communicator for load balancing.

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

- **Communicator Selection**: Maps `channel_id` to different communicators using `comm_id = channel_id % num_gin_ctxs` for load balancing

- **High-level API**: Uses `ncclGin::put()` method with `ncclTeamWorld` and memory windows

- **Offset Calculation**: Computes source and destination offsets relative to `gin_base_ptr`

- **Data Transfer**: Transfers `sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2)` bytes of channel metadata

- **No Signaling**: Put operation completes without signals or counters (`ncclGin_None{}`)

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

- **Signal-based Atomic Operations**: Uses `ncclGin::signal()` method with `ncclGin_SignalAdd` to perform atomic addition

- **Signal ID Calculation**: Signal ID is computed as `gin_signals_tail + rdma_rank` to identify the target signal

- **Tail Pointer Management**: Atomically updates tail pointers to track available buffer space in circular buffers

- **Communicator Mapping**: Maps `channel_id` to appropriate communicator using `comm_id = channel_id % num_gin_ctxs`

- **No Data Transfer**: Only performs signal operation without transferring data (pure atomic operation)

- **High-level API**: Uses the high-level signal API for atomic operations with default thread scope

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

- **Signal-based Atomic Operations**: Uses `ncclGin::signal()` method with `ncclGin_SignalAdd` to perform atomic addition

- **Signal ID Calculation**: Signal ID is computed as `gin_signals_head + rdma_rank` to identify the target signal

- **Head Pointer Updates**: Atomically communicates head pointer advances to remote ranks to indicate freed buffer space

- **Delta Calculation**: Adds the difference `(min_head - last_head)` to the remote head pointer

- **Buffer Space Management**: Indicates freed buffer space that can be reused by remote ranks

- **Communicator Selection**: Maps `channel_id` to appropriate communicator using `comm_id = channel_id % num_gin_ctxs`

- **High-level API**: Uses the high-level signal API for atomic operations with default thread scope

---

## Signal Operations

### Signal Layout and Management

**Purpose**: NCCL GIN uses signals to implement atomic operations on head/tail pointers of circular buffers.

#### Signal Organization

Signals are organized in a structured layout to ensure each rank's head and tail pointers for each channel have unique signal IDs:

- **Signal Base**: Each buffer type (e.g., internode buffer with index 2) has a base signal ID obtained via:
  ```
  backend->get_signals_base(internode_buffer_idx)
  ```

- **Head Signals**: Located at:
  ```
  signals_base + (channel_id * kNumRDMARanks) + rank_id
  ```

- **Tail Signals**: Located at:
  ```
  signals_base + (kNumRDMARanks * num_channels) + (channel_id * kNumRDMARanks) + rank_id
  ```

This layout ensures unique signal IDs for all head and tail pointers across all channels and ranks.

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

- **Signal Range**: Total of `kNumRDMARanks * num_channels * 2` signals (head + tail for each rank and channel)

- **Thread Distribution**: Each thread handles one specific signal for parallelization

- **Signal ID Calculation**: Signal ID is computed as `signals_base + thread_id`

- **Channel Derivation**: Channel ID is derived from signal offset using the signal layout structure

- **Channel-specific Reset**: Only resets signal using the communicator assigned to that channel (`comm_id = channel_id % num_gin_ctxs`)

- **Signal Layout**: Head signals come first, followed by tail signals (two distinct ranges)

- **High-level API**: Uses `ncclGin::resetSignal()` method for atomic reset to zero

- **Synchronization**: Uses `__syncthreads()` to ensure all resets complete before proceeding

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

- **Context Selection**: Uses different GIN context for each channel/SM via `comm_id = channel_id % num_gin_ctxs`

- **Signal ID Calculation**: Signal ID is computed as `gin_signals_head + lane_id` to read the head pointer for each rank

- **Signal Reading**: Uses `ncclGin::readSignal()` method to atomically read the 64-bit signal value

- **Type Conversion**: Converts the 64-bit signal value to 32-bit integer for head pointer caching

- **High-level API**: Simple method call replaces low-level volatile pointer operations

- **Memory Ordering**: Atomic operations and memory ordering are handled internally by the API

---

## Quiet Operations

### Quiet Operation

**Purpose**: Ensures all previously issued RDMA operations (puts, atomics, signals, etc.) have completed and their effects are visible to remote processing elements (PEs). This is critical when reusing buffers to prevent data corruption from in-flight operations.

#### Original NVSHMEM Implementation

`nvshmem_quiet()` is a completion and memory ordering operation that ensures all previously issued NVSHMEM operations (put, get, atomic operations, etc.) targeting all processing elements have completed and their effects are visible to remote PEs.

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

- **Multi-context Flush**: Each thread flushes one GIN context to parallelize the flush operation

- **Thread Distribution**: Distributes flush operations across threads (one context per thread, where `comm_id = thread_id`)

- **High-level API**: Uses `ncclGin::flush()` method for completion guarantee

- **Memory Ordering**: Uses `cuda::std::memory_order_acquire` to ensure all previous operations are visible

- **Cooperation Scope**: Uses `ncclCoopThread()` to indicate thread-level cooperation

- **Synchronization**: Uses `__syncthreads()` to ensure all contexts have been flushed before proceeding

- **Buffer Safety**: Critical for preventing data corruption when reusing RDMA buffers

#### Use Case

This operation is typically used before clearing or reusing RDMA buffers to ensure that all previous writes to those buffers have completed. In the DeepEP framework, it waits for all inflight work requests to complete before rewriting the cleared `rdma_buffer`, preventing race conditions between in-flight RDMA operations and buffer reuse.

---
