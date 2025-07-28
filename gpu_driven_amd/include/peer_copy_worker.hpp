#pragma once
#include "common.hpp"
#include "ring_buffer.hip.h"
#include <atomic>
#include <thread>
#include <hip/hip_runtime.h>

extern std::atomic<bool> g_run;
#ifdef ENABLE_PROXY_HIP_MEMCPY
extern thread_local uint64_t async_memcpy_count;
extern thread_local uint64_t async_memcpy_total_time;
#endif
extern int src_device;

void peer_copy_worker(CopyRingBuffer& g_ring, int idx);