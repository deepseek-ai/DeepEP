#pragma once

#include <memory>

#include "communication_backend.h"

namespace deep_ep {
namespace internode {

// Factory function for creating backends
std::unique_ptr<CommunicationBackend> create_backend(BackendType type);

}  // namespace internode
}  // namespace deep_ep