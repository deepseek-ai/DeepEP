#include "backend_factory.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "communication_backend.h"
#ifdef ENABLE_NCCL
#include "nccl_gin_backend.h"
#endif

namespace deep_ep {
namespace internode {

// Global backend instance
static std::unique_ptr<CommunicationBackend> g_backend;

// Factory method implementation
std::unique_ptr<CommunicationBackend> CommunicationBackend::create(BackendType type) {
    return create_backend(type);
}

// Backend selection utilities
BackendType parse_backend_type(const std::string& backend_str) {
    if (backend_str == "nccl" || backend_str == "nccl_gin") {
        return BackendType::NCCL;
    } else if (backend_str == "auto") {
        return BackendType::AUTO;
    } else {
        throw std::runtime_error("Invalid backend string: " + backend_str);
    }
}

BackendType detect_backend_type() {
    const char* env_backend = std::getenv("DEEP_EP_BACKEND");

    if (env_backend) {
        try {
            return parse_backend_type(env_backend);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid DEEP_EP_BACKEND value: " << env_backend << ". Falling back to NCCL." << std::endl;
        }
    }

    return BackendType::NCCL;
}

std::string backend_type_to_string(BackendType type) {
    switch (type) {
        case BackendType::NCCL:
            return "nccl";
        case BackendType::AUTO:
            return "auto";
        default:
            return "unknown";
    }
}

// Global backend management
void initialize_backend(
    BackendType type, const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode, int qps_per_rank) {
    // printf("initialize_backend: %s\n", backend_type_to_string(type).c_str()); fflush(stdout);
    if (g_backend) {
        throw std::runtime_error("Backend already initialized");
    }

    g_backend = CommunicationBackend::create(type);
    if (!g_backend) {
        throw std::runtime_error("Failed to create backend");
    }

    int result = g_backend->init(root_unique_id_val, rank, num_ranks, low_latency_mode, qps_per_rank);
    if (result != rank) {
        throw std::runtime_error("Backend initialization failed");
    }
}

CommunicationBackend* get_backend() {
    if (!g_backend) {
        throw std::runtime_error("Backend not initialized");
    }
    return g_backend.get();
}

void finalize_backend() {
    if (g_backend) {
        g_backend->finalize();
        g_backend.reset();
    }
}

std::unique_ptr<CommunicationBackend> create_backend(BackendType type) {
    switch (type) {
        case BackendType::NCCL:
#ifdef ENABLE_NCCL
            return std::make_unique<NCCLGINBackend>();
#else
            throw std::runtime_error("NCCL backend not compiled in. Set ENABLE_NCCL=1 to enable.");
#endif
        case BackendType::AUTO:
#ifdef ENABLE_NCCL
            return std::make_unique<NCCLGINBackend>();
#else
            throw std::runtime_error("No backend available. Set ENABLE_NCCL=1 to enable NCCL backend.");
#endif
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

}  // namespace internode
}  // namespace deep_ep