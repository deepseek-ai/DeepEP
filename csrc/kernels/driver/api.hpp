#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <vector>

#include <deep_ep/layout/common/copy_engine.cuh>

namespace deep_ep::comm {
class Context;
}

namespace deep_ep::driver {

void copy_engine_push(const CUstream& stream,
                      const std::vector<void*>& dsts,
                      const std::vector<void*>& srcs,
                      const std::vector<size_t>& sizes);

void copy_engine_wait(const CUstream& stream,
                      const std::vector<uint64_t*>& signals,
                      const uint64_t& target);

void copy_engine_push_synced(const comm::Context& context,
                             layout::CopyEngineSignals& signals,
                             const std::vector<void*>& dsts,
                             const std::vector<void*>& srcs,
                             const std::vector<size_t>& sizes,
                             const CUstream& stream);

}  // namespace deep_ep::driver
