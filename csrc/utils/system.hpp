#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

#include <deep_ep/common/exception.cuh>

namespace deep_ep {

// ReSharper disable once CppNotAllPathsReturnValue
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    const auto c_str = std::getenv(name.c_str());
    if (c_str == nullptr)
        return default_value;

    // Read the env and convert to the desired type
    if constexpr (std::is_same_v<dtype_t, std::string>) {
        return std::string(c_str);
    } else if constexpr (std::is_same_v<dtype_t, int>) {
        int value;
        std::sscanf(c_str, "%d", &value);
        return value;
    } else {
        EP_HOST_ASSERT(false and "Unexpected type");
    }
}

} // deep_ep
