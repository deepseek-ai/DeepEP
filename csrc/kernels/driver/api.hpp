#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <vector>

namespace deep_ep::driver {

uint64_t increase_seq_idx(const CUstream& stream, uint64_t& seq_idx);

void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts,
                      const std::vector<void*>& srcs,
                      const std::vector<size_t>& sizes);

void copy_engine_sync(const CUstream& stream, const uint64_t& seq_idx,
                      const std::vector<uint64_t*>& write_signals,
                      const std::vector<uint64_t*>& wait_signals);

void copy_engine_wait(const CUstream& stream, const std::vector<uint64_t*>& signals, const uint64_t& target);

}  // namespace deep_ep::driver
