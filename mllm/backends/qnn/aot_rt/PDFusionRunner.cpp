// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/PDFusionRunner.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

namespace {

constexpr const char* kInputIdsName = "input_ids";
constexpr const char* kPositionIdsName = "position_ids";
constexpr const char* kAttentionMaskName = "attention_mask";
constexpr const char* kFusionCtrlName = "fusion_ctrl";

inline std::string pastKeyName(int layer, const char* slot) { return "past_key_" + std::string(slot) + "_" + std::to_string(layer); }
inline std::string pastValueName(int layer, const char* slot) { return "past_value_" + std::string(slot) + "_" + std::to_string(layer); }

inline std::string presentKeyName(int layer) { return "present_key_" + std::to_string(layer); }
inline std::string presentValueName(int layer) { return "present_value_" + std::to_string(layer); }

inline std::string updatedPastKeyName(int layer, const char* slot) {
  if (slot == nullptr || std::string(slot).empty()) { return "updated_past_key_" + std::to_string(layer); }
  return "updated_past_key_" + std::string(slot) + "_" + std::to_string(layer);
}
inline std::string updatedPastValueName(int layer, const char* slot) {
  if (slot == nullptr || std::string(slot).empty()) { return "updated_past_value_" + std::to_string(layer); }
  return "updated_past_value_" + std::string(slot) + "_" + std::to_string(layer);
}

}  // namespace

PDFusionRunner::PDFusionRunner(const QnnAOTConfig& config) : config_(config) {}

bool PDFusionRunner::load() {
  return load({});
}

bool PDFusionRunner::load(const std::vector<int>& pd_total_lens) {
  if (!config_.pd_fusion_enable) {
    MLLM_ERROR("PDFusionRunner::load called but config.pd_fusion_enable is false");
    return false;
  }

  pd_total_lens_.clear();
  if (!pd_total_lens.empty()) {
    pd_total_lens_ = pd_total_lens;
  } else {
    pd_total_lens_ = {config_.pd_total_len};
  }
  std::sort(pd_total_lens_.begin(), pd_total_lens_.end());
  pd_total_lens_.erase(std::unique(pd_total_lens_.begin(), pd_total_lens_.end()), pd_total_lens_.end());
  if (pd_total_lens_.empty()) {
    MLLM_ERROR("PDFusionRunner::load: empty pd_total_lens");
    return false;
  }
  for (int N : pd_total_lens_) {
    if (N <= 1 || N >= config_.context_len) {
      MLLM_ERROR("Invalid PD total_len {} (context_len={})", N, config_.context_len);
      return false;
    }
  }

  // Ensure max_ar_len/max_cache_len are compatible with PD.
  config_.max_ar_len = std::max(config_.max_ar_len, *std::max_element(pd_total_lens_.begin(), pd_total_lens_.end()));
  config_.max_cache_len = config_.context_len - 1;  // keep consistent with existing aot_rt runtime convention

  // 1-token decode-only graph (optional fallback when no prefill work is available).
  decode_module_ = std::make_unique<QnnAOTModule>("model.0.s1");
  decode_module_->to(kQNN);

  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  if (!backend) {
    MLLM_ERROR("QNN Backend not found");
    return false;
  }

  kv_prefill_ = std::make_unique<KVCacheManager<uint8_t>>(config_);
  kv_decode_ = std::make_unique<KVCacheManager<uint8_t>>(config_);

  // Use the max PD total length as the initial "graph ar_len" for both caches.
  kv_prefill_->initCache(backend->allocator().get(), config_.max_ar_len);
  kv_decode_->initCache(backend->allocator().get(), config_.max_ar_len);
  // Share output buffers so both cache managers can consume the fused graph's KV outputs.
  kv_decode_->aliasOutputBuffersFrom(*kv_prefill_);

  // Default bindings: internal caches.
  active_prefill_ = kv_prefill_.get();
  active_decode_ = kv_decode_.get();

  pd_graphs_.clear();
  for (int N : pd_total_lens_) {
    PDGraphIO g;
    g.total_len = N;
    g.past_len = config_.context_len - N;
    g.module = std::make_unique<QnnAOTModule>("model.0.pd.s" + std::to_string(N));
    g.module->to(kQNN);
    init_pd_io(g);
    pd_graphs_.emplace(N, std::move(g));
  }

  init_decode_io();
  return true;
}

PDFusionRunner::PDGraphIO* PDFusionRunner::get_pd_graph(int total_len) {
  auto it = pd_graphs_.find(total_len);
  if (it == pd_graphs_.end()) { return nullptr; }
  return &it->second;
}

void PDFusionRunner::bindCaches(KVCacheManager<uint8_t>* prefill, KVCacheManager<uint8_t>* decode) {
  active_prefill_ = prefill ? prefill : kv_prefill_.get();
  active_decode_ = decode ? decode : kv_decode_.get();

  // Ensure output buffers of the active caches point to the fused graph output buffers (owned by kv_prefill_).
  // updateCache() reads from output_buffer pointers.
  if (active_prefill_) { active_prefill_->aliasOutputBuffersFrom(*kv_prefill_); }
  if (active_decode_) { active_decode_->aliasOutputBuffersFrom(*kv_prefill_); }

  // Rebind PD graph KV input pointers to the active cache buffers for all PD graphs.
  const int L = config_.num_layers;

  const auto& k0 = active_prefill_->getKCache();
  const auto& v0 = active_prefill_->getVCache();
  const auto& k1 = active_decode_->getKCache();
  const auto& v1 = active_decode_->getVCache();

  for (auto& [_, g] : pd_graphs_) {
    const int base = 4;
    for (int l = 0; l < L; ++l) {
      g.input_tensors[base + l].impl()->storage()->ptr_ = k0[l].buffer;
      g.input_tensors[base + L + l].impl()->storage()->ptr_ = v0[l].buffer;
      g.input_tensors[base + 2 * L + l].impl()->storage()->ptr_ = k1[l].buffer;
      g.input_tensors[base + 3 * L + l].impl()->storage()->ptr_ = v1[l].buffer;
    }
    // Rebind output KV pointers (output_buffer) for slot0(prefill) owner.
    for (int l = 0; l < L; ++l) {
      g.output_tensors[1 + l].impl()->storage()->ptr_ = kv_prefill_->getKCache()[l].output_buffer;
      g.output_tensors[1 + L + l].impl()->storage()->ptr_ = kv_prefill_->getVCache()[l].output_buffer;
    }

    if (config_.kv_update_on_device) {
      const int base_upd = 1 + 2 * L;
      // updated prefill K/V
      for (int l = 0; l < L; ++l) { g.output_tensors[base_upd + l].impl()->storage()->ptr_ = k0[l].buffer; }
      for (int l = 0; l < L; ++l) { g.output_tensors[base_upd + L + l].impl()->storage()->ptr_ = v0[l].buffer; }
      // updated decode K/V
      for (int l = 0; l < L; ++l) { g.output_tensors[base_upd + 2 * L + l].impl()->storage()->ptr_ = k1[l].buffer; }
      for (int l = 0; l < L; ++l) { g.output_tensors[base_upd + 3 * L + l].impl()->storage()->ptr_ = v1[l].buffer; }
    }
  }

  // Rebind decode-only graph KV input pointers to the active decode cache buffers.
  const int decode_base = 3;
  if ((int)decode_input_tensors_.size() >= decode_base + 2 * L) {
    for (int l = 0; l < L; ++l) {
      decode_input_tensors_[decode_base + l].impl()->storage()->ptr_ = k1[l].buffer;
      decode_input_tensors_[decode_base + L + l].impl()->storage()->ptr_ = v1[l].buffer;
    }
  }

  // Rebind decode-only graph KV output pointers to the canonical KV output buffers (owned by kv_prefill_).
  // Both PD and decode-only executions write their "present_*" tensors into the same output buffers so that
  // KVCacheManager::updateCache() can always read from aliasOutputBuffersFrom(*kv_prefill_).
  if ((int)decode_output_tensors_.size() >= 1 + 2 * L) {
    for (int l = 0; l < L; ++l) {
      decode_output_tensors_[1 + l].impl()->storage()->ptr_ = kv_prefill_->getKCache()[l].output_buffer;
      decode_output_tensors_[1 + L + l].impl()->storage()->ptr_ = kv_prefill_->getVCache()[l].output_buffer;
    }
  }

  if (config_.kv_update_on_device) {
    const int base_upd = 1 + 2 * L;
    if ((int)decode_output_tensors_.size() >= base_upd + 2 * L) {
      for (int l = 0; l < L; ++l) { decode_output_tensors_[base_upd + l].impl()->storage()->ptr_ = k1[l].buffer; }
      for (int l = 0; l < L; ++l) { decode_output_tensors_[base_upd + L + l].impl()->storage()->ptr_ = v1[l].buffer; }
    }
  }
}

void PDFusionRunner::init_pd_io(PDGraphIO& g) {
  g.input_tensors.clear();
  g.output_tensors.clear();

  g.input_tensors.reserve(4 + 4 * config_.num_layers);

  const int total_len = g.total_len;
  const int32_t past_len = g.past_len;

  // 1) input_ids: [1, total_len]
  auto input_ids = Tensor::empty({1, total_len}, kInt32, kQNN).alloc();
  input_ids.setName(kInputIdsName);
  g.input_tensors.push_back(input_ids);

  // 2) position_ids: [total_len]
  auto pos_ids = Tensor::empty({total_len}, kInt32, kQNN).alloc();
  pos_ids.setName(kPositionIdsName);
  g.input_tensors.push_back(pos_ids);

  // 3) attention_mask: [1, 1, total_len, context_len]
  auto attn_mask = Tensor::empty({1, 1, total_len, config_.context_len}, kUInt16, kQNN).alloc();
  attn_mask.setName(kAttentionMaskName);
  g.input_tensors.push_back(attn_mask);

  // 4) fusion_ctrl: [6]
  //   [0]=is_prefill_active, [1]=is_decode_active,
  //   [2]=prefill_n_update (used by PDKVCacheUpdate),
  //   [3]=reserved,
  //   [4]=prefill_n_past (for future fused attention kernels),
  //   [5]=decode_n_past  (for future fused attention kernels).
  auto fusion_ctrl = Tensor::empty({6}, kInt32, kQNN).alloc();
  fusion_ctrl.setName(kFusionCtrlName);
  g.input_tensors.push_back(fusion_ctrl);

  // 5) KV caches for slot0(prefill) and slot1(decode)
  const auto& k_prefill = kv_prefill_->getKCache();
  const auto& v_prefill = kv_prefill_->getVCache();
  const auto& k_decode = kv_decode_->getKCache();
  const auto& v_decode = kv_decode_->getVCache();

  for (int l = 0; l < config_.num_layers; ++l) {
    // slot0 K
    // Scheme A: token-major K cache [B, H, past_len, D] to make per-token D contiguous.
    auto k0 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    k0.impl()->storage()->ptr_ = k_prefill[l].buffer;
    k0.impl()->storage()->mem_type_ = kManual;
    k0.setName(pastKeyName(l, "prefill"));
    g.input_tensors.push_back(k0);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    // slot0 V
    auto v0 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    v0.impl()->storage()->ptr_ = v_prefill[l].buffer;
    v0.impl()->storage()->mem_type_ = kManual;
    v0.setName(pastValueName(l, "prefill"));
    g.input_tensors.push_back(v0);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    // slot1 K
    auto k1 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    k1.impl()->storage()->ptr_ = k_decode[l].buffer;
    k1.impl()->storage()->mem_type_ = kManual;
    k1.setName(pastKeyName(l, "decode"));
    g.input_tensors.push_back(k1);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    // slot1 V
    auto v1 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    v1.impl()->storage()->ptr_ = v_decode[l].buffer;
    v1.impl()->storage()->mem_type_ = kManual;
    v1.setName(pastValueName(l, "decode"));
    g.input_tensors.push_back(v1);
  }

  // Outputs: logits + present_key/value (packed pd_total_len tokens),
  // optionally followed by on-device KV-cache update outputs.
  const bool kv_update = config_.kv_update_on_device;
  g.output_tensors.reserve(1 + 2 * config_.num_layers + (kv_update ? 4 * config_.num_layers : 0));

  // P0-1: PD graphs only need the last 2 rows of logits (prefill_last + decode).
  // After right-aligned packing, runtime consumes fixed rows: [N-2, N-1] -> logits shape [1,1,2,vocab].
  auto logits = Tensor::empty({1, 1, 2, config_.vocab_size}, kUInt16, kQNN).alloc();
  logits.setName("logits");
  g.output_tensors.push_back(logits);

  // Use kv_prefill_ output buffers as the canonical "present_*" storage.
  for (int l = 0; l < config_.num_layers; ++l) {
    auto k_out = Tensor::empty({1, config_.num_heads, config_.head_dim, total_len}, config_.kv_dtype, kQNN);
    k_out.impl()->storage()->ptr_ = kv_prefill_->getKCache()[l].output_buffer;
    k_out.impl()->storage()->mem_type_ = kManual;
    k_out.setName(presentKeyName(l));
    g.output_tensors.push_back(k_out);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    auto v_out = Tensor::empty({1, config_.num_heads, total_len, config_.head_dim}, config_.kv_dtype, kQNN);
    v_out.impl()->storage()->ptr_ = kv_prefill_->getVCache()[l].output_buffer;
    v_out.impl()->storage()->mem_type_ = kManual;
    v_out.setName(presentValueName(l));
    g.output_tensors.push_back(v_out);
  }

  if (kv_update) {
    // Updated caches for slot0(prefill) and slot1(decode). Runtime binds these outputs to the same shared buffers
    // as the corresponding inputs (in-place update) to avoid CPU memcpy.
    for (int l = 0; l < config_.num_layers; ++l) {
      auto k_upd0 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
      k_upd0.impl()->storage()->ptr_ = kv_prefill_->getKCache()[l].buffer;
      k_upd0.impl()->storage()->mem_type_ = kManual;
      k_upd0.setName(updatedPastKeyName(l, "prefill"));
      g.output_tensors.push_back(k_upd0);
    }
    for (int l = 0; l < config_.num_layers; ++l) {
      auto v_upd0 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
      v_upd0.impl()->storage()->ptr_ = kv_prefill_->getVCache()[l].buffer;
      v_upd0.impl()->storage()->mem_type_ = kManual;
      v_upd0.setName(updatedPastValueName(l, "prefill"));
      g.output_tensors.push_back(v_upd0);
    }
    for (int l = 0; l < config_.num_layers; ++l) {
      auto k_upd1 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
      k_upd1.impl()->storage()->ptr_ = kv_decode_->getKCache()[l].buffer;
      k_upd1.impl()->storage()->mem_type_ = kManual;
      k_upd1.setName(updatedPastKeyName(l, "decode"));
      g.output_tensors.push_back(k_upd1);
    }
    for (int l = 0; l < config_.num_layers; ++l) {
      auto v_upd1 = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
      v_upd1.impl()->storage()->ptr_ = kv_decode_->getVCache()[l].buffer;
      v_upd1.impl()->storage()->mem_type_ = kManual;
      v_upd1.setName(updatedPastValueName(l, "decode"));
      g.output_tensors.push_back(v_upd1);
    }
  }
}

void PDFusionRunner::init_decode_io() {
  decode_input_tensors_.clear();
  decode_output_tensors_.clear();

  decode_input_tensors_.reserve(3 + 2 * config_.num_layers);

  // 1) input_ids: [1, 1]
  auto input_ids = Tensor::empty({1, 1}, kInt32, kQNN).alloc();
  input_ids.setName(kInputIdsName);
  decode_input_tensors_.push_back(input_ids);

  // 2) position_ids: [1]
  auto pos_ids = Tensor::empty({1}, kInt32, kQNN).alloc();
  pos_ids.setName(kPositionIdsName);
  decode_input_tensors_.push_back(pos_ids);

  // 3) attention_mask: [1, 1, 1, context_len]
  auto attn_mask = Tensor::empty({1, 1, 1, config_.context_len}, kUInt16, kQNN).alloc();
  attn_mask.setName(kAttentionMaskName);
  decode_input_tensors_.push_back(attn_mask);

  // 4) KV cache for decode slot only (past_len = context_len - 1)
  const auto& k_decode = kv_decode_->getKCache();
  const auto& v_decode = kv_decode_->getVCache();
  const int32_t past_len = config_.context_len - 1;

  for (int l = 0; l < config_.num_layers; ++l) {
    auto k = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    k.impl()->storage()->ptr_ = k_decode[l].buffer;
    k.impl()->storage()->mem_type_ = kManual;
    k.setName("past_key_" + std::to_string(l));
    decode_input_tensors_.push_back(k);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    auto v = Tensor::empty({1, config_.num_heads, past_len, config_.head_dim}, config_.kv_dtype, kQNN);
    v.impl()->storage()->ptr_ = v_decode[l].buffer;
    v.impl()->storage()->mem_type_ = kManual;
    v.setName("past_value_" + std::to_string(l));
    decode_input_tensors_.push_back(v);
  }

  const bool kv_update = config_.kv_update_on_device;
  decode_output_tensors_.reserve(1 + 2 * config_.num_layers + (kv_update ? 2 * config_.num_layers : 0));
  auto logits = Tensor::empty({1, 1, 1, config_.vocab_size}, kUInt16, kQNN).alloc();
  logits.setName("logits");
  decode_output_tensors_.push_back(logits);

  for (int l = 0; l < config_.num_layers; ++l) {
    auto k_out = Tensor::empty({1, config_.num_heads, config_.head_dim, 1}, config_.kv_dtype, kQNN);
    k_out.impl()->storage()->ptr_ = kv_decode_->getKCache()[l].output_buffer;
    k_out.impl()->storage()->mem_type_ = kManual;
    k_out.setName(presentKeyName(l));
    decode_output_tensors_.push_back(k_out);
  }
  for (int l = 0; l < config_.num_layers; ++l) {
    auto v_out = Tensor::empty({1, config_.num_heads, 1, config_.head_dim}, config_.kv_dtype, kQNN);
    v_out.impl()->storage()->ptr_ = kv_decode_->getVCache()[l].output_buffer;
    v_out.impl()->storage()->mem_type_ = kManual;
    v_out.setName(presentValueName(l));
    decode_output_tensors_.push_back(v_out);
  }

  if (kv_update) {
    // Updated single-slot caches.
    const int32_t past_len_upd = config_.context_len - 1;
    for (int l = 0; l < config_.num_layers; ++l) {
      auto k_upd = Tensor::empty({1, config_.num_heads, past_len_upd, config_.head_dim}, config_.kv_dtype, kQNN);
      k_upd.impl()->storage()->ptr_ = kv_decode_->getKCache()[l].buffer;
      k_upd.impl()->storage()->mem_type_ = kManual;
      k_upd.setName(updatedPastKeyName(l, ""));
      decode_output_tensors_.push_back(k_upd);
    }
    for (int l = 0; l < config_.num_layers; ++l) {
      auto v_upd = Tensor::empty({1, config_.num_heads, past_len_upd, config_.head_dim}, config_.kv_dtype, kQNN);
      v_upd.impl()->storage()->ptr_ = kv_decode_->getVCache()[l].buffer;
      v_upd.impl()->storage()->mem_type_ = kManual;
      v_upd.setName(updatedPastValueName(l, ""));
      decode_output_tensors_.push_back(v_upd);
    }
  }
}

void PDFusionRunner::prepare_io(PDGraphIO& g, const PrefillSlot& prefill, const DecodeSlot& decode, int32_t prefill_n_update) {
  const int32_t total_len = g.total_len;
  const int32_t prefill_len = total_len - 1;
  const int32_t decode_idx = total_len - 1;
  const int32_t context_len = config_.context_len;
  const int32_t past_len = g.past_len;

  if (config_.kv_update_on_device && !warned_kv_aliasing_) {
    const int L = config_.num_layers;
    const int in_base = 4;
    const int out_base_upd = 1 + 2 * L;
    bool all_aliased = true;
    if ((int)g.input_tensors.size() >= in_base + 4 * L && (int)g.output_tensors.size() >= out_base_upd + 4 * L) {
      for (int l = 0; l < L; ++l) {
        auto* in_k0 = g.input_tensors[in_base + l].impl()->storage()->ptr_;
        auto* in_v0 = g.input_tensors[in_base + L + l].impl()->storage()->ptr_;
        auto* in_k1 = g.input_tensors[in_base + 2 * L + l].impl()->storage()->ptr_;
        auto* in_v1 = g.input_tensors[in_base + 3 * L + l].impl()->storage()->ptr_;

        auto* out_k0 = g.output_tensors[out_base_upd + l].impl()->storage()->ptr_;
        auto* out_v0 = g.output_tensors[out_base_upd + L + l].impl()->storage()->ptr_;
        auto* out_k1 = g.output_tensors[out_base_upd + 2 * L + l].impl()->storage()->ptr_;
        auto* out_v1 = g.output_tensors[out_base_upd + 3 * L + l].impl()->storage()->ptr_;

        all_aliased = all_aliased && (in_k0 == out_k0) && (in_v0 == out_v0) && (in_k1 == out_k1) && (in_v1 == out_v1);
      }
    } else {
      all_aliased = false;
    }
    if (!all_aliased) {
      MLLM_WARN("PDKVCacheUpdate: KV cache input/output are not fully aliased; kernel will preserve cache by copying in_* -> out_* "
                "(correct but bandwidth-heavy). Consider enabling true in-place aliasing for performance.");
    }
    warned_kv_aliasing_ = true;
  }

  int32_t* input_ids_ptr = g.input_tensors[0].ptr<int32_t>();
  int32_t* pos_ids_ptr = g.input_tensors[1].ptr<int32_t>();
  uint16_t* attn_mask_ptr = g.input_tensors[2].ptr<uint16_t>();
  int32_t* ctrl_ptr = g.input_tensors[3].ptr<int32_t>();

  // ctrl layout:
  // - ctrl[0], ctrl[1] as enable flags
  // - ctrl[2] as prefill_n_update (used by PDKVCacheUpdate custom op in the traced graph)
  // - ctrl[4], ctrl[5] as past lengths (n_past) for slot0/slot1 (future fused attention kernels)
  ctrl_ptr[0] = prefill.active ? 1 : 0;
  ctrl_ptr[1] = decode.active ? 1 : 0;
  ctrl_ptr[2] = prefill.active ? prefill_n_update : 0;
  ctrl_ptr[3] = 0;
  // Will be filled after clampPast is defined.

  // P0-2: right-align prefill chunk into [base .. N-2], keeping decode fixed at N-1.
  // This makes the two "useful" rows fixed: prefill_last=N-2, decode_last=N-1.
  const int32_t m = (prefill.active ? prefill_n_update : 0);
  const int32_t base = (m > 0) ? (decode_idx - m) : decode_idx;

  // Zero-fill everything first.
  for (int i = 0; i < total_len; ++i) {
    input_ids_ptr[i] = 0;
    pos_ids_ptr[i] = 0;
  }

  // Prefill tokens placed at [base .. decode_idx-1].
  if (prefill.active && m > 0) {
    for (int32_t t = 0; t < m; ++t) {
      const int32_t row = base + t;
      if (row < 0 || row >= decode_idx) continue;
      const int64_t src = prefill.prompt_pos + (int64_t)t;
      if (src >= 0 && src < (int64_t)prefill.prompt_tokens.size()) {
        input_ids_ptr[row] = (int32_t)prefill.prompt_tokens[src];
      }
      pos_ids_ptr[row] = (int32_t)(prefill.start_pos + (int64_t)t);
    }
  }

  // Decode token fixed at last row.
  if (decode.active) {
    input_ids_ptr[decode_idx] = (int32_t)decode.cur_token;
    pos_ids_ptr[decode_idx] = (int32_t)decode.start_pos;
  }

  // Attention mask: [1, 1, total_len, context_len] (flattened as [total_len, context_len]).
  //
  // Convention matches examples/qwen3_qnn_aot/modeling_qwen_qnn_aot.hpp:
  //   allowed = 65535, masked = 0
  //
  // Layout matches KVCacheManager::initAttentionMask:
  //   [0 .. past_len-1]                 -> past KV cache slots
  //   [past_len .. past_len+total_len-1] -> current tokens (packed prefill+decode)
  constexpr uint16_t kMasked = 0;
  constexpr uint16_t kAllowed = 65535;
  // P1-1: avoid rewriting the entire [N * context_len] mask every step.
  // Only rows that are actually consumed this step need correct masking; unused rows can remain stale
  // because they never contribute to KV updates nor logits sampling after right-aligned packing.
  if (!g.mask_initialized) {
    std::fill_n(attn_mask_ptr, total_len * context_len, kMasked);
    g.mask_initialized = true;
  }

  auto clampPast = [&](int64_t n_past) -> int32_t {
    if (n_past <= 0) return 0;
    if (n_past >= past_len) return past_len;
    return (int32_t)n_past;
  };
  ctrl_ptr[4] = prefill.active ? clampPast(prefill.start_pos) : 0;
  ctrl_ptr[5] = decode.active ? clampPast(decode.start_pos) : 0;

  // Prefill rows: only [base .. decode_idx-1] are active, causal within that interval,
  // and must NOT see decode token (last column in current window).
  if (prefill.active && m > 0) {
    const int32_t n_past0 = clampPast(prefill.start_pos);
    for (int32_t t = 0; t < m; ++t) {
      const int32_t r = base + t;
      if (r < 0 || r >= decode_idx) continue;
      uint16_t* row = attn_mask_ptr + r * context_len;
      std::fill_n(row, context_len, kMasked);
      std::fill_n(row, n_past0, kAllowed);
      for (int32_t j = base; j <= r; ++j) { row[past_len + j] = kAllowed; }
    }
  }

  // Decode row: attends to its own past + itself only (must NOT see prefill current tokens).
  if (decode.active) {
    const int32_t n_past1 = clampPast(decode.start_pos);
    uint16_t* row = attn_mask_ptr + decode_idx * context_len;
    std::fill_n(row, context_len, kMasked);
    std::fill_n(row, n_past1, kAllowed);
    row[past_len + decode_idx] = kAllowed;
  }
}

void PDFusionRunner::prepare_decode_io(const DecodeSlot& decode) {
  if (config_.kv_update_on_device && !warned_kv_aliasing_) {
    const int L = config_.num_layers;
    const int in_base = 3;
    const int out_base_upd = 1 + 2 * L;
    bool all_aliased = true;
    if ((int)decode_input_tensors_.size() >= in_base + 2 * L && (int)decode_output_tensors_.size() >= out_base_upd + 2 * L) {
      for (int l = 0; l < L; ++l) {
        auto* in_k = decode_input_tensors_[in_base + l].impl()->storage()->ptr_;
        auto* in_v = decode_input_tensors_[in_base + L + l].impl()->storage()->ptr_;
        auto* out_k = decode_output_tensors_[out_base_upd + l].impl()->storage()->ptr_;
        auto* out_v = decode_output_tensors_[out_base_upd + L + l].impl()->storage()->ptr_;
        all_aliased = all_aliased && (in_k == out_k) && (in_v == out_v);
      }
    } else {
      all_aliased = false;
    }
    if (!all_aliased) {
      MLLM_WARN("PDKVCacheUpdate (decode-only): KV cache input/output are not fully aliased; kernel will preserve cache by copying in_* -> out_* "
                "(correct but bandwidth-heavy).");
    }
    warned_kv_aliasing_ = true;
  }

  int32_t* input_ids_ptr = decode_input_tensors_[0].ptr<int32_t>();
  int32_t* pos_ids_ptr = decode_input_tensors_[1].ptr<int32_t>();
  uint16_t* attn_mask_ptr = decode_input_tensors_[2].ptr<uint16_t>();

  input_ids_ptr[0] = decode.active ? (int32_t)decode.cur_token : 0;
  pos_ids_ptr[0] = (int32_t)decode.start_pos;

  // attention_mask: [1, 1, 1, context_len], allowed=65535, masked=0
  constexpr uint16_t kMasked = 0;
  constexpr uint16_t kAllowed = 65535;
  std::fill_n(attn_mask_ptr, config_.context_len, kMasked);

  const int32_t past_len = config_.context_len - 1;
  const int32_t n_past = std::min<int32_t>(std::max<int32_t>(0, (int32_t)decode.start_pos), past_len);
  // Allow past KV prefix.
  std::fill_n(attn_mask_ptr, n_past, kAllowed);
  // Allow self at the last column (new token column in fixed layout).
  attn_mask_ptr[past_len] = kAllowed;
}

void PDFusionRunner::update_kv_after_run(int total_len, const PrefillSlot& prefill, const DecodeSlot& decode, int32_t prefill_n_update) {
  if (config_.kv_update_on_device) {
    // KV-cache update is handled inside the graph by PDKVCacheUpdate custom ops.
    return;
  }
  // Ensure both KV caches are arranged for pd_total_len.
  active_prefill_->rearrangeCache(total_len);
  active_decode_->rearrangeCache(total_len);

  // Slot0: write the prefill_n_update tokens for this step.
  // With right-aligned packing, prefill tokens occupy [base .. total_len-2].
  if (prefill.active && prefill_n_update > 0) {
    const int32_t decode_idx = total_len - 1;
    const int32_t base = decode_idx - prefill_n_update;
    std::vector<bool> selected(total_len, false);
    for (int32_t i = 0; i < prefill_n_update; ++i) {
      const int32_t idx = base + i;
      if (idx >= 0 && idx < decode_idx) { selected[idx] = true; }
    }
    active_prefill_->updateCache(total_len, (int32_t)prefill.start_pos, prefill_n_update, selected);
  }

  // Slot1: write exactly 1 token from the last row (idx=pd_total_len-1).
  if (decode.active) {
    std::vector<bool> selected(total_len, false);
    selected[total_len - 1] = true;
    active_decode_->updateCache(total_len, (int32_t)decode.start_pos, /*n_update=*/1, selected);
  }
}

void PDFusionRunner::update_decode_kv_after_run(const DecodeSlot& decode) {
  if (config_.kv_update_on_device) { return; }
  if (decode.active) {
    active_decode_->updateCache(1, (int32_t)decode.start_pos, /*n_update=*/1, {});
  }
}

int64_t PDFusionRunner::sample_token_from_logits(const mllm::Tensor& logits) {
  // logits: [1, 1, 1, vocab] or [1,1,N,vocab] after slicing/squeezing.
  auto x = logits.to(kCPU);
  if (x.rank() == 4) { x = x.squeeze(0).squeeze(0); }
  if (x.rank() == 3) { x = x.squeeze(0); }
  // This path is only used by decode-only execution (model.0.s1).
  return decode_module_->sampleGreedy(x);
}

static int64_t sample_token_at_idx_from_logits_tensor(QnnAOTModule* module, const mllm::Tensor& logits4d, int32_t token_idx) {
  auto logits_ = logits4d.to(kCPU).squeeze(0).squeeze(0);
  auto logits = logits_[{token_idx, kAll}];
  return module->sampleGreedy(logits);
}

PDFusionRunner::StepResult PDFusionRunner::step(const PrefillSlot& prefill, const DecodeSlot& decode) {
  if (!active_prefill_ || !active_decode_) { return {}; }
  if (pd_total_lens_.empty()) { return {}; }

  // Determine how many prefill tokens are valid for this step.
  int32_t prefill_n_update = 0;
  if (prefill.active) {
    auto remain = (int64_t)prefill.prompt_tokens.size() - prefill.prompt_pos;
    const int64_t max_prefill = (int64_t)pd_total_lens_.back() - 1;
    prefill_n_update = (int32_t)std::max<int64_t>(0, std::min<int64_t>(remain, max_prefill));
  }

  // Choose the smallest PD graph that can hold prefill_n_update tokens (plus 1 decode row).
  int total_len = pd_total_lens_.back();
  if (prefill.active && prefill_n_update > 0) {
    for (int N : pd_total_lens_) {
      if ((N - 1) >= prefill_n_update) {
        total_len = N;
        break;
      }
    }
  }
  auto* g = get_pd_graph(total_len);
  if (!g || !g->module) { return {}; }

  // Ensure KV cache layout matches the PD graph input shapes before execution.
  active_prefill_->rearrangeCache(total_len);
  active_decode_->rearrangeCache(total_len);

  prepare_io(*g, prefill, decode, prefill_n_update);
  g->module->setOutputTensors(g->output_tensors);

  auto module_input = g->input_tensors;
  auto outs = (*g->module)(module_input);
  if (outs.empty() || outs.size() != g->output_tensors.size()) {
    MLLM_ERROR("PDFusionRunner: PD graph execution failed or returned unexpected outputs (graph='{}', expected={}, got={})",
               g->module->getModuleName(), g->output_tensors.size(), outs.size());
    return {};
  }
  g->output_tensors = std::move(outs);

  update_kv_after_run(total_len, prefill, decode, prefill_n_update);

  StepResult ret;
  if (prefill.active && prefill_n_update > 0) {
    // P0-1/P0-2: logits only contains 2 rows: [prefill_last, decode_last].
    ret.prefill_next_token = sample_token_at_idx_from_logits_tensor(g->module.get(), g->output_tensors[0], /*token_idx=*/0);
  }
  if (decode.active) {
    ret.decode_next_token = sample_token_at_idx_from_logits_tensor(g->module.get(), g->output_tensors[0], /*token_idx=*/1);
  }
  return ret;
}

std::optional<int64_t> PDFusionRunner::decodeOnly(const DecodeSlot& decode) {
  if (!decode_module_ || !active_decode_) { return std::nullopt; }
  if (!decode.active) { return std::nullopt; }

  prepare_decode_io(decode);
  decode_module_->setOutputTensors(decode_output_tensors_);

  // Ensure decode cache layout matches 1-token graph input shapes.
  active_decode_->rearrangeCache(1);

  auto module_input = decode_input_tensors_;
  auto outs = (*decode_module_)(module_input);
  if (outs.empty() || outs.size() != decode_output_tensors_.size()) {
    MLLM_ERROR("PDFusionRunner: decode-only graph execution failed or returned unexpected outputs (graph='{}', expected={}, got={})",
               decode_module_->getModuleName(), decode_output_tensors_.size(), outs.size());
    return std::nullopt;
  }
  decode_output_tensors_ = std::move(outs);

  update_decode_kv_after_run(decode);

  // decode logits: [1,1,1,vocab]
  return sample_token_from_logits(decode_output_tensors_[0]);
}

void PDFusionRunner::clearCaches() {
  if (!kv_prefill_ || !kv_decode_) return;
  clearPrefillCache();
  clearDecodeCache();
}

void PDFusionRunner::clearPrefillCache() {
  if (!kv_prefill_) return;
  auto clear_one = [](KVCacheManager<uint8_t>& kv) {
    for (const auto& item : kv.getKCache()) {
      if (item.buffer_storage && item.buffer_storage->ptr_ && item.buffer_storage->size_) {
        std::memset(item.buffer_storage->ptr_, 0, item.buffer_storage->size_);
      }
    }
    for (const auto& item : kv.getVCache()) {
      if (item.buffer_storage && item.buffer_storage->ptr_ && item.buffer_storage->size_) {
        std::memset(item.buffer_storage->ptr_, 0, item.buffer_storage->size_);
      }
    }
  };
  clear_one(*kv_prefill_);
}

void PDFusionRunner::clearDecodeCache() {
  if (!kv_decode_) return;
  auto clear_one = [](KVCacheManager<uint8_t>& kv) {
    for (const auto& item : kv.getKCache()) {
      if (item.buffer_storage && item.buffer_storage->ptr_ && item.buffer_storage->size_) {
        std::memset(item.buffer_storage->ptr_, 0, item.buffer_storage->size_);
      }
    }
    for (const auto& item : kv.getVCache()) {
      if (item.buffer_storage && item.buffer_storage->ptr_ && item.buffer_storage->size_) {
        std::memset(item.buffer_storage->ptr_, 0, item.buffer_storage->size_);
      }
    }
  };
  clear_one(*kv_decode_);
}

void PDFusionRunner::swapSlots() {
  if (!kv_prefill_ || !kv_decode_) return;

  std::swap(kv_prefill_, kv_decode_);
  // Ensure slot1(decode) reads KV outputs from the fused graph execution buffers owned by slot0(prefill).
  kv_decode_->aliasOutputBuffersFrom(*kv_prefill_);

  // swapSlots() is intended for the runner's internal 2-slot pipeline usage.
  // After swapping, reset bindings to internal caches and rebind pointers.
  bindCaches(nullptr, nullptr);
}

}  // namespace mllm::qnn::aot
