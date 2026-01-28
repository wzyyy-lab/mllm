//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_FusedPDAttention);

// NOTE: This file currently provides a correctness-first reference implementation.
// On real devices, a float+expf softmax over long histories (e.g. 4k) can be slow enough
// to trigger DSP watchdog timeouts. To make "wiring smoke tests" safer, we cap the
// reference kernel's past length by default. For real performance/accuracy, replace this
// reference with an HVX/HMX optimized kernel and remove/raise the cap as needed.
#ifndef MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST
#define MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST 256
#endif

#ifndef MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ
#define MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ 32
#endif

// op execute function declarations
template<typename TensorType>
GraphStatus fusedPDAttentionImpl(TensorType& out_attn,
                                 const TensorType& q,
                                 const TensorType& k_curr,
                                 const TensorType& v_curr,
                                 const TensorType& past_k_prefill,
                                 const TensorType& past_v_prefill,
                                 const TensorType& past_k_decode,
                                 const TensorType& past_v_decode,
                                 const TensorType& attention_mask,
                                 const TensorType& fusion_ctrl,
                                 const TensorType& q_scale,
                                 const TensorType& q_zp,
                                 const TensorType& k_scale,
                                 const TensorType& k_zp,
                                 const TensorType& v_scale,
                                 const TensorType& v_zp,
                                 const TensorType& out_scale,
                                 const TensorType& out_zp);

// forward declaration of sample cost function
static float fusedPDAttentionCostFunc(const Op* op);

DEF_PACKAGE_OP((fusedPDAttentionImpl<Tensor>), "FusedPDAttention")

template<typename TensorType>
GraphStatus fusedPDAttentionImpl(TensorType& out_attn,
                                 const TensorType& q,
                                 const TensorType& k_curr,
                                 const TensorType& v_curr,
                                 const TensorType& past_k_prefill,
                                 const TensorType& past_v_prefill,
                                 const TensorType& past_k_decode,
                                 const TensorType& past_v_decode,
                                 const TensorType& /*attention_mask*/,
                                 const TensorType& fusion_ctrl,
                                 const TensorType& q_scale,
                                 const TensorType& q_zp,
                                 const TensorType& k_scale,
                                 const TensorType& k_zp,
                                 const TensorType& v_scale,
                                 const TensorType& v_zp,
                                 const TensorType& out_scale,
                                 const TensorType& out_zp) {
  // Correctness-first reference implementation (slow):
  // - Computes PD attention without reading the full attention_mask.
  // - Uses fusion_ctrl to determine active rows and past lengths.
  // - Dequantizes Q/K/V to float, computes masked softmax, accumulates V, then requantizes to uint16.
  //
  // This is NOT an optimized HTP kernel. It is intended to validate end-to-end wiring and semantics
  // before implementing an HVX/HMX-optimized version.

  out_attn.set_dims(q);

  // Expect Q and output to be uint16 (per-tensor asymmetric).
  if (q.get_dtype() != DType::QUInt16 || out_attn.get_dtype() != DType::QUInt16) { return GraphStatus::Success; }
  // Expect K/V to be uint8 (per-tensor symmetric represented as uint8 with zp=128).
  if (k_curr.get_dtype() != DType::QUInt8 || v_curr.get_dtype() != DType::QUInt8) { return GraphStatus::Success; }

  auto [b_q, h_attn, s_q, d_q] = q.dims();  // [B, H_attn, S, D]
  auto [b_k, h_kv, d_k, s_k] = k_curr.dims();  // [B, H_kv, D, S]
  auto [b_v, h_v, s_v, d_v] = v_curr.dims();  // [B, H_kv, S, D]
  auto [b_pk0, h_pk0, d_pk0, past_len0] = past_k_prefill.dims();  // [B, H_kv, D, past_len]
  auto [b_pv0, h_pv0, past_lenv0, d_pv0] = past_v_prefill.dims(); // [B, H_kv, past_len, D]
  auto [b_pk1, h_pk1, d_pk1, past_len1] = past_k_decode.dims();
  auto [b_pv1, h_pv1, past_lenv1, d_pv1] = past_v_decode.dims();

  if (b_q != b_k || b_q != b_v) { return GraphStatus::Success; }
  if (h_kv != h_v) { return GraphStatus::Success; }
  if (d_q != d_k || d_q != d_v) { return GraphStatus::Success; }
  if (s_q != s_k || s_q != s_v) { return GraphStatus::Success; }
  if (h_kv != h_pk0 || h_kv != h_pv0 || h_kv != h_pk1 || h_kv != h_pv1) { return GraphStatus::Success; }
  if (d_k != d_pk0 || d_k != d_pv0 || d_k != d_pk1 || d_k != d_pv1) { return GraphStatus::Success; }
  if (past_len0 != past_lenv0 || past_len1 != past_lenv1) { return GraphStatus::Success; }
  if (past_len0 != past_len1) { return GraphStatus::Success; }
  if (b_pk0 != b_q || b_pv0 != b_q || b_pk1 != b_q || b_pv1 != b_q) { return GraphStatus::Success; }

  const int32_t past_len = static_cast<int32_t>(past_len0);
  const int32_t seq_len = static_cast<int32_t>(s_q);
  const int32_t head_dim = static_cast<int32_t>(d_q);
  const int32_t ctx_len = past_len + seq_len;
  if (seq_len <= 0 || head_dim <= 0 || ctx_len <= 0) { return GraphStatus::Success; }

  // Scalars
  const float q_scale_f = static_cast<float>(q_scale(0, 0, 0, 0));
  const int32_t q_zp_i = static_cast<int32_t>(q_zp(0, 0, 0, 0));
  const float k_scale_f = static_cast<float>(k_scale(0, 0, 0, 0));
  const int32_t k_zp_i = static_cast<int32_t>(k_zp(0, 0, 0, 0));
  const float v_scale_f = static_cast<float>(v_scale(0, 0, 0, 0));
  const int32_t v_zp_i = static_cast<int32_t>(v_zp(0, 0, 0, 0));
  const float out_scale_f = static_cast<float>(out_scale(0, 0, 0, 0));
  const int32_t out_zp_i = static_cast<int32_t>(out_zp(0, 0, 0, 0));
  if (q_scale_f == 0.f || k_scale_f == 0.f || v_scale_f == 0.f || out_scale_f == 0.f) { return GraphStatus::Success; }

  const int32_t prefill_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 0));
  const int32_t decode_active = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 1));
  const int32_t prefill_n_update = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 2));
  // fusion_ctrl[3] is reserved by the runtime and can be used by this reference kernel as a runtime guard:
  // - 0  : use compile-time default (MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ)
  // - >0 : cap allowed seq_len to this value (if seq_len is larger, the op returns early)
  // - <0 : disable seq_len guard entirely
  const int32_t ref_max_seq_override = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 3));
  int32_t prefill_n_past = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 4));
  int32_t decode_n_past = static_cast<int32_t>(fusion_ctrl(0, 0, 0, 5));
  if (prefill_n_past < 0) prefill_n_past = 0;
  if (decode_n_past < 0) decode_n_past = 0;
  if (prefill_n_past > past_len) prefill_n_past = past_len;
  if (decode_n_past > past_len) decode_n_past = past_len;

  const int32_t decode_idx = seq_len - 1;
  const int32_t max_prefill_rows = decode_idx;
  int32_t prefill_rows = prefill_n_update;
  if (prefill_rows < 0) prefill_rows = 0;
  if (prefill_rows > max_prefill_rows) prefill_rows = max_prefill_rows;
  const int32_t prefill_base = decode_idx - prefill_rows;

  const int32_t groups = (h_kv > 0) ? (h_attn / h_kv) : 1;
  if (groups <= 0) { return GraphStatus::Success; }

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(q.raw_data_const());
  const uint8_t* k_curr_ptr = reinterpret_cast<const uint8_t*>(k_curr.raw_data_const());
  const uint8_t* v_curr_ptr = reinterpret_cast<const uint8_t*>(v_curr.raw_data_const());

  const uint8_t* pk0_ptr = reinterpret_cast<const uint8_t*>(past_k_prefill.raw_data_const());
  const uint8_t* pv0_ptr = reinterpret_cast<const uint8_t*>(past_v_prefill.raw_data_const());
  const uint8_t* pk1_ptr = reinterpret_cast<const uint8_t*>(past_k_decode.raw_data_const());
  const uint8_t* pv1_ptr = reinterpret_cast<const uint8_t*>(past_v_decode.raw_data_const());

  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(out_attn.raw_data());
  std::memset(out_ptr, 0, (size_t)b_q * (size_t)h_attn * (size_t)seq_len * (size_t)head_dim * sizeof(uint16_t));

  // Safety guard for the reference implementation:
  // avoid accidentally running a quadratic-time float+expf kernel for large seq_len on DSP.
  int32_t eff_max_seq = MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ;
  if (ref_max_seq_override > 0) eff_max_seq = ref_max_seq_override;
  if (ref_max_seq_override < 0) eff_max_seq = 0;
  if (eff_max_seq > 0 && seq_len > eff_max_seq) {
    if (out_zp_i != 0) {
      const size_t total = (size_t)b_q * (size_t)h_attn * (size_t)seq_len * (size_t)head_dim;
      for (size_t i = 0; i < total; ++i) { out_ptr[i] = (uint16_t)out_zp_i; }
    }
    return GraphStatus::Success;
  }

  // Helper lambdas for indexing.
  auto q_off = [&](int32_t b, int32_t h, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_attn + (size_t)h) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };
  auto out_off = q_off;

  // k cache: [B,H_kv,D,Len]
  auto kcache_off = [&](int32_t b, int32_t hk, int32_t d, int32_t t, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)head_dim + (size_t)d) * (size_t)len + (size_t)t;
  };
  // v cache: [B,H_kv,Len,D]
  auto vcache_off = [&](int32_t b, int32_t hk, int32_t t, int32_t d, int32_t len) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)len + (size_t)t) * (size_t)head_dim + (size_t)d;
  };
  // k_curr: [B,H_kv,D,S]
  auto kcurr_off = [&](int32_t b, int32_t hk, int32_t d, int32_t s) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)head_dim + (size_t)d) * (size_t)seq_len + (size_t)s;
  };
  // v_curr: [B,H_kv,S,D]
  auto vcurr_off = [&](int32_t b, int32_t hk, int32_t s, int32_t d) -> size_t {
    return (((size_t)b * (size_t)h_kv + (size_t)hk) * (size_t)seq_len + (size_t)s) * (size_t)head_dim + (size_t)d;
  };

  const float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

  // NOTE: This reference kernel avoids reading attention_mask to reduce bandwidth.
  // It reconstructs the allowed positions from (n_past, row index) matching PDFusionRunner::prepare_io().

  // Temporary storage (per head/row): scores for allowed positions.
  // Use heap allocation to avoid large stack frames on DSP.
  const int32_t ref_max_past = (MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST > 0) ? MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST : 0;
  const int32_t max_allowed = (ref_max_past > 0) ? std::min(ctx_len, ref_max_past + seq_len) : ctx_len;
  float* scores = (float*)malloc((size_t)max_allowed * sizeof(float));
  float* probs = (float*)malloc((size_t)max_allowed * sizeof(float));
  float* qtmp = (float*)malloc((size_t)head_dim * sizeof(float));
  float* outtmp = (float*)malloc((size_t)head_dim * sizeof(float));
  if (!scores || !probs || !qtmp || !outtmp) {
    if (scores) free(scores);
    if (probs) free(probs);
    if (qtmp) free(qtmp);
    if (outtmp) free(outtmp);
    return GraphStatus::Success;
  }

  auto run_row = [&](int32_t b, int32_t h, int32_t row, bool is_decode_row) {
    const int32_t hk = h / groups;
    if (hk < 0 || hk >= (int32_t)h_kv) { return; }

    // Dequantize q vector.
    for (int32_t d = 0; d < head_dim; ++d) {
      const int32_t q_u = (int32_t)q_ptr[q_off(b, h, row, d)];
      qtmp[d] = ((float)(q_u - q_zp_i)) * q_scale_f;
    }

    int32_t n_past = is_decode_row ? decode_n_past : prefill_n_past;
    if (n_past < 0) n_past = 0;
    if (n_past > past_len) n_past = past_len;
    if (ref_max_past > 0 && n_past > ref_max_past) n_past = ref_max_past;

    // Allowed positions are a subset of [0..ctx_len-1].
    //  - past: [0..n_past-1]
    //  - current: if decode row -> only (past_len + decode_idx)
    //             else -> [past_len + prefill_base .. past_len + row] (right-aligned causal within chunk)
    int32_t allowed_count = 0;
    // past positions
    for (int32_t t = 0; t < n_past; ++t) {
      float dot = 0.f;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)(is_decode_row ? pk1_ptr[kcache_off(b, hk, d, t, past_len)]
                                                   : pk0_ptr[kcache_off(b, hk, d, t, past_len)]);
        const float k_f = ((float)(k_u - k_zp_i)) * k_scale_f;
        dot += qtmp[d] * k_f;
      }
      scores[allowed_count++] = dot * inv_sqrt_d;
    }
    // current positions
    if (is_decode_row) {
      // only decode token itself
      const int32_t s = decode_idx;
      float dot = 0.f;
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t k_u = (int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)];
        const float k_f = ((float)(k_u - k_zp_i)) * k_scale_f;
        dot += qtmp[d] * k_f;
      }
      scores[allowed_count++] = dot * inv_sqrt_d;
    } else {
      for (int32_t s = prefill_base; s <= row; ++s) {
        float dot = 0.f;
        for (int32_t d = 0; d < head_dim; ++d) {
          const int32_t k_u = (int32_t)k_curr_ptr[kcurr_off(b, hk, d, s)];
          const float k_f = ((float)(k_u - k_zp_i)) * k_scale_f;
          dot += qtmp[d] * k_f;
        }
        scores[allowed_count++] = dot * inv_sqrt_d;
      }
    }
    if (allowed_count <= 0) { return; }

    // Softmax over allowed positions.
    float maxv = scores[0];
    for (int32_t i = 1; i < allowed_count; ++i) {
      if (scores[i] > maxv) maxv = scores[i];
    }
    float sum = 0.f;
    for (int32_t i = 0; i < allowed_count; ++i) {
      const float e = expf(scores[i] - maxv);
      probs[i] = e;
      sum += e;
    }
    if (sum == 0.f) { return; }
    const float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < allowed_count; ++i) { probs[i] *= inv_sum; }

    // Weighted sum of V.
    for (int32_t d = 0; d < head_dim; ++d) { outtmp[d] = 0.f; }

    int32_t idx = 0;
    // past V
    for (int32_t t = 0; t < n_past; ++t) {
      const float w = probs[idx++];
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t v_u = (int32_t)(is_decode_row ? pv1_ptr[vcache_off(b, hk, t, d, past_len)]
                                                   : pv0_ptr[vcache_off(b, hk, t, d, past_len)]);
        const float v_f = ((float)(v_u - v_zp_i)) * v_scale_f;
        outtmp[d] += w * v_f;
      }
    }
    // current V
    if (is_decode_row) {
      const int32_t s = decode_idx;
      const float w = probs[idx++];
      for (int32_t d = 0; d < head_dim; ++d) {
        const int32_t v_u = (int32_t)v_curr_ptr[vcurr_off(b, hk, s, d)];
        const float v_f = ((float)(v_u - v_zp_i)) * v_scale_f;
        outtmp[d] += w * v_f;
      }
    } else {
      for (int32_t s = prefill_base; s <= row; ++s) {
        const float w = probs[idx++];
        for (int32_t d = 0; d < head_dim; ++d) {
          const int32_t v_u = (int32_t)v_curr_ptr[vcurr_off(b, hk, s, d)];
          const float v_f = ((float)(v_u - v_zp_i)) * v_scale_f;
          outtmp[d] += w * v_f;
        }
      }
    }

    // Requantize to uint16.
    for (int32_t d = 0; d < head_dim; ++d) {
      const float qv = outtmp[d] / out_scale_f;
      int32_t out_u = (int32_t)lrintf(qv) + out_zp_i;
      if (out_u < 0) out_u = 0;
      if (out_u > 65535) out_u = 65535;
      out_ptr[out_off(b, h, row, d)] = (uint16_t)out_u;
    }
  };

  // Execute requested rows.
  if (prefill_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) {
        for (int32_t t = 0; t < prefill_rows; ++t) { run_row(b, h, prefill_base + t, /*is_decode_row=*/false); }
      }
    }
  }
  if (decode_active) {
    for (int32_t b = 0; b < (int32_t)b_q; ++b) {
      for (int32_t h = 0; h < (int32_t)h_attn; ++h) { run_row(b, h, decode_idx, /*is_decode_row=*/true); }
    }
  }

  free(scores);
  free(probs);
  free(qtmp);
  free(outtmp);
  return GraphStatus::Success;
}

__attribute__((unused)) static float fusedPDAttentionCostFunc(const Op* /*op*/) { return 0.0f; }

END_PKG_OP_DEFINITION(PKG_FusedPDAttention);
