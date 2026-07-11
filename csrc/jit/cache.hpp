#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_ep::jit {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    KernelRuntimeCache() = default;

    void clear() {
        cache.clear();
    }

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }

    // Load a runtime from `load_path` and cache it under `key_path`. Used by the losing
    // side of the build rename race: its just-compiled artifacts are equivalent to the
    // winner's (the cache key hashes the kernel name, source, flags, and compiler
    // signature), and the winner's directory may not be visible to this client yet on
    // shared filesystems.
    std::shared_ptr<KernelRuntime> put(const std::filesystem::path& key_path, const std::filesystem::path& load_path) {
        return cache[key_path] = std::make_shared<KernelRuntime>(load_path);
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_ep::jit
