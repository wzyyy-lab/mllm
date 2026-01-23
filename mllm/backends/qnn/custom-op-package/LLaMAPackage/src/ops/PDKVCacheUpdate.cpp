//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include <cstring>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_PDKVCacheUpdate);

// op execute function declarations
template<typename TensorType, typename ScalarTensorType>
GraphStatus pdKvCacheUpdateImpl(TensorType& out_k_cache,
                                TensorType& out_v_cache,
                                const TensorType& dep,
                                const TensorType& in_k_cache,
                                const TensorType& in_v_cache,
                                const TensorType& present_k,
                                const TensorType& present_v,
                                const ScalarTensorType& n_past,
                                const ScalarTensorType& src_offset,
                                const ScalarTensorType& n_update,
                                const ScalarTensorType& enable);

// forward declaration of sample cost function
static float pdKvCacheUpdateCostFunc(const Op* op);

DEF_PACKAGE_OP((pdKvCacheUpdateImpl<Tensor, Tensor>), "PDKVCacheUpdate")

template<typename TensorType, typename ScalarTensorType>
GraphStatus pdKvCacheUpdateImpl(TensorType& out_k_cache,
                                TensorType& out_v_cache,
                                const TensorType& /*dep*/,
                                const TensorType& in_k_cache,
                                const TensorType& in_v_cache,
                                const TensorType& present_k,
                                const TensorType& present_v,
                                const ScalarTensorType& n_past,
                                const ScalarTensorType& src_offset,
                                const ScalarTensorType& n_update,
                                const ScalarTensorType& enable) {
  /*
   * NOTE:
   * - This op is used as a runtime KV-cache updater for PD-fusion graphs.
   * - To avoid copying the full cache, we only write the updated span.
   * - Runtime is expected to bind out_* buffers to the same underlying memory as in_* buffers (in-place update).
   *
   * Tensor layouts (Qwen3 AOT example):
   * - K cache:      [B, H, D, past_len]
   * - V cache:      [B, H, past_len, D]
   * - present_k:    [B, H, D, S]
   * - present_v:    [B, H, S, D]
   * - Scalars:      int32/uint32, rank=SCALAR or [1]
   */

  // Propagate output shapes.
  out_k_cache.set_dims(in_k_cache);
  out_v_cache.set_dims(in_v_cache);

  const uint32_t enable_ = static_cast<uint32_t>(enable(0, 0, 0, 0));
  if (!enable_) { return GraphStatus::Success; }

  int32_t n_past_ = static_cast<int32_t>(n_past(0, 0, 0, 0));
  int32_t src_offset_ = static_cast<int32_t>(src_offset(0, 0, 0, 0));
  int32_t n_update_ = static_cast<int32_t>(n_update(0, 0, 0, 0));
  if (n_update_ <= 0) { return GraphStatus::Success; }

  auto [b_k, h_k, d_k, past_len_k] = in_k_cache.dims();
  auto [b_v, h_v, past_len_v, d_v] = in_v_cache.dims();
  auto [b_pk, h_pk, d_pk, s_pk] = present_k.dims();
  auto [b_pv, h_pv, s_pv, d_pv] = present_v.dims();

  // Basic consistency checks (best-effort; avoid aborting in production kernels).
  if (b_k != b_pk || h_k != h_pk || d_k != d_pk) { return GraphStatus::Success; }
  if (b_v != b_pv || h_v != h_pv || d_v != d_pv) { return GraphStatus::Success; }
  if (past_len_k != past_len_v) { return GraphStatus::Success; }
  if (s_pk != s_pv) { return GraphStatus::Success; }

  const int32_t past_len = static_cast<int32_t>(past_len_k);
  const int32_t seq_len = static_cast<int32_t>(s_pk);

  if (n_past_ < 0) n_past_ = 0;
  if (src_offset_ < 0) src_offset_ = 0;
  if (src_offset_ >= seq_len) { return GraphStatus::Success; }
  if (n_past_ >= past_len) { return GraphStatus::Success; }
  if (n_past_ + n_update_ > past_len) { n_update_ = past_len - n_past_; }
  if (src_offset_ + n_update_ > seq_len) { n_update_ = seq_len - src_offset_; }
  if (n_update_ <= 0) { return GraphStatus::Success; }

  // Determine element size.
  const DType dtype = present_k.get_dtype();
  size_t elem_bytes = 0;
  if (dtype == DType::QUInt8) {
    elem_bytes = sizeof(uint8_t);
  } else if (dtype == DType::Float16) {
    elem_bytes = sizeof(float) / 2;
  } else if (dtype == DType::Float32) {
    elem_bytes = sizeof(float);
  } else {
    return GraphStatus::Success;
  }

  const uint8_t* pk_ptr = reinterpret_cast<const uint8_t*>(present_k.raw_data_const());
  const uint8_t* pv_ptr = reinterpret_cast<const uint8_t*>(present_v.raw_data_const());
  const uint8_t* k_in_ptr = reinterpret_cast<const uint8_t*>(in_k_cache.raw_data_const());
  const uint8_t* v_in_ptr = reinterpret_cast<const uint8_t*>(in_v_cache.raw_data_const());
  uint8_t* k_out_ptr = reinterpret_cast<uint8_t*>(out_k_cache.raw_data());
  uint8_t* v_out_ptr = reinterpret_cast<uint8_t*>(out_v_cache.raw_data());

  // If runtime couldn't bind out_* to alias in_* (SSA restriction / allocator decision),
  // we must preserve the existing cache content, otherwise we'd lose the untouched prefix/suffix.
  //
  // This is intentionally a full-buffer copy only in the non-alias case; in the common "true in-place"
  // configuration (out == in), this is a zero-cost branch.
  if (k_out_ptr != k_in_ptr) {
    const size_t total_k_bytes = (size_t)b_k * (size_t)h_k * (size_t)d_k * (size_t)past_len * elem_bytes;
    std::memcpy(k_out_ptr, k_in_ptr, total_k_bytes);
  }
  if (v_out_ptr != v_in_ptr) {
    const size_t total_v_bytes = (size_t)b_v * (size_t)h_v * (size_t)past_len * (size_t)d_v * elem_bytes;
    std::memcpy(v_out_ptr, v_in_ptr, total_v_bytes);
  }

  // K cache update: for each (B,H,D), copy n_update along last dim.
  for (Idx b = 0; b < b_k; ++b) {
    for (Idx h = 0; h < h_k; ++h) {
      for (Idx d = 0; d < d_k; ++d) {
        const size_t dst_base = (((b * h_k + h) * d_k + d) * past_len + (size_t)n_past_) * elem_bytes;
        const size_t src_base = (((b * h_k + h) * d_k + d) * (size_t)seq_len + (size_t)src_offset_) * elem_bytes;
        std::memcpy(k_out_ptr + dst_base, pk_ptr + src_base, (size_t)n_update_ * elem_bytes);
      }
    }
  }

  // V cache update: for each (B,H), copy n_update*D contiguous values.
  const size_t row_bytes = (size_t)d_v * elem_bytes;
  for (Idx b = 0; b < b_v; ++b) {
    for (Idx h = 0; h < h_v; ++h) {
      const size_t dst_base = (((b * h_v + h) * (size_t)past_len + (size_t)n_past_) * (size_t)d_v) * elem_bytes;
      const size_t src_base = (((b * h_v + h) * (size_t)seq_len + (size_t)src_offset_) * (size_t)d_v) * elem_bytes;
      std::memcpy(v_out_ptr + dst_base, pv_ptr + src_base, (size_t)n_update_ * row_bytes);
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float pdKvCacheUpdateCostFunc(const Op* /*op*/) { return 0.0f; }

END_PKG_OP_DEFINITION(PKG_PDKVCacheUpdate);
