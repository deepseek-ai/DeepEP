#pragma once

#include <deep_ep/common/compiled.cuh>

#define LEGACY_NUM_MAX_NVL_PEERS 8
#define LEGACY_NUM_MAX_RDMA_PEERS 20
#define LEGACY_NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define LEGACY_NUM_MAX_LOCAL_EXPERTS 1024
#define LEGACY_NUM_BUFFER_ALIGNMENT_BYTES 128

#define LEGACY_LOW_LATENCY_SEND_PHASE 1
#define LEGACY_LOW_LATENCY_RECV_PHASE 2

#define LEGACY_FINISHED_SUM_TAG 1024
#define LEGACY_NUM_WAIT_NANOSECONDS 500

// Overridable at build time (e.g. -DLEGACY_NUM_CPU_TIMEOUT_SECS=1800) for consumers
// whose first-step warmup legitimately exceeds the default: JIT kernel compilation
// (e.g. DeepGEMM) on a subset of ranks holds the others inside dispatch past 100 s,
// which the watchdog then misreports as a communication timeout.
#ifndef LEGACY_NUM_CPU_TIMEOUT_SECS
#define LEGACY_NUM_CPU_TIMEOUT_SECS 100
#endif
#ifndef LEGACY_NUM_TIMEOUT_CYCLES
#define LEGACY_NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#endif
