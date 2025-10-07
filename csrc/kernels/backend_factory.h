#pragma once

#include "communication_backend.h"
#include <memory>

namespace deep_ep {
namespace internode {

// Factory function for creating backends
std::unique_ptr<CommunicationBackend> create_backend(BackendType type);

} // namespace internode
} // namespace deep_ep 