#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <deep_ep/common/compiled.cuh>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

#include "../utils/system.hpp"

namespace deep_ep {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path, const std::string& nccl_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto library_include_dir = library_root / "include";
    const auto nccl_include_dir = std::filesystem::absolute(
        std::filesystem::path(nccl_root_path) / "include");
    const auto config = deep_jit::Config(
        library_root,
        "EP",
        {},
        {library_include_dir},
        {"deep_ep/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config, nccl_include_dir] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);

        // Keep the NCCL include path in compiler options so it participates in the cache key.
        // DeepEP's own include path stays path-independent because its contents are tracked by the parser.
        auto& nvcc_flags = *runtime->default_compiler_options.nvcc_flags;
        nvcc_flags.emplace_back("--diag-suppress=39,161,174,177,186,940,3012");
        nvcc_flags.emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        nvcc_flags.emplace_back("-I" + nccl_include_dir.string());
        nvcc_flags.emplace_back("-DEP_NUM_TOPK_IDX_BITS=" + std::to_string(EP_NUM_TOPK_IDX_BITS));
        nvcc_flags.emplace_back("-DNCCL_GIN_GDAKI_ENABLE=1");
        nvcc_flags.emplace_back("-DNCCL_GIN_PROXY_ENABLE=0");
        nvcc_flags.emplace_back("-DNCCL_GIN_GPI_ENABLE=0");
        nvcc_flags.emplace_back("-DNCCL_GIN_EFA_GDA_ENABLE=0");
        if (get_env<int>("EP_GIN_GDAKI_DEBUG", 0))
            nvcc_flags.emplace_back("-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1");
        return runtime;
    });
}

inline void register_apis(pybind11::module_& module) {
    deep_jit::register_python_api(module, jit);
    module.def("init_jit", &init_jit);
}

}  // namespace deep_ep
