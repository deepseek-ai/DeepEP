// This file is the real header...

namespace deep_ep {
namespace internode_ll {

// NOTE extracted from `dispatch` body
template <bool kUseFP8, bool kUseNVFP4, int kHidden>
struct DispatchConstsTemplate {
    // FP8 staffs
    static constexpr int kNumPerChannels = kUseNVFP4 ? 16 : 128;
    static constexpr int num_scales = kHidden / kNumPerChannels;
    static constexpr size_t hidden_bytes =
        kUseNVFP4
            ? kHidden * sizeof(__nv_fp8_storage_t) / 2
            : kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    static constexpr size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<
        kUseNVFP4,
        int32_t,
        std::conditional_t<kUseFP8, int2, int4>>;
    using rdma_x_scale_t = std::conditional_t<kUseNVFP4, uint8_t, float>;
    static constexpr size_t num_bytes_per_msg = sizeof(int4) + ((kUseFP8 || kUseNVFP4) ? (hidden_bytes + num_scales * sizeof(rdma_x_scale_t)) : hidden_bytes);
    static constexpr size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
};

} // namespace internode_ll
} // namespace deep_ep
