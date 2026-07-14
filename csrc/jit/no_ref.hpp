#pragma once

namespace deep_ep::jit {

// The pointer itself is the kernel argument storage; do not take its address when launching.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace deep_ep::jit
