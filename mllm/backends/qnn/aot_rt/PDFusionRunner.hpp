// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::qnn::aot {

// A minimal runtime-side "dual-slot" PD-fusion executor.
//
// This class only provides IO packing + KV updates for a *single* fused static graph execution.
// It assumes a PD-fused QNN graph exists (compiled separately) and follows the IO naming conventions used here.
class PDFusionRunner {
 public:
  struct PrefillSlot {
    // Up to config.pd_prefill_len tokens will be consumed from prompt_tokens.
    std::vector<int64_t> prompt_tokens;
    int64_t prompt_pos = 0;
    int64_t start_pos = 0;  // KV cache start position for this request
    bool active = false;
  };

  struct DecodeSlot {
    uint64_t cur_token = 0;
    int64_t start_pos = 0;  // KV cache position for this token
    bool active = false;
  };

  struct StepResult {
    // If active, returns sampled next token for each slot (greedy sampling).
    std::optional<int64_t> prefill_next_token;
    std::optional<int64_t> decode_next_token;
  };

  explicit PDFusionRunner(const QnnAOTConfig& config);

  // Allocate buffers and load the fused graph module.
  bool load();
  // Load multiple PD graphs (e.g. N=32/64/128) so runtime can pick the smallest that fits the current prefill chunk.
  // If empty, defaults to {config.pd_total_len}.
  bool load(const std::vector<int>& pd_total_lens);

  // Bind external KV caches for the next execution.
  // If nullptr is passed, the runner's internal cache for that slot is used.
  // NOTE: the fused graph's KV outputs are always read from the runner's internal output buffers, and copied into
  // the bound caches via KVCacheManager::updateCache().
  void bindCaches(KVCacheManager<uint8_t>* prefill, KVCacheManager<uint8_t>* decode);

  // Execute exactly one PD-fused step (one graphExecute).
  StepResult step(const PrefillSlot& prefill, const DecodeSlot& decode);

  // Decode-only step using the 1-token graph (e.g. `model.0.s1`) to avoid wasting PD rows when prefill is absent.
  // Returns the sampled next token (greedy) if decode.active=true.
  std::optional<int64_t> decodeOnly(const DecodeSlot& decode);

  // Clear both KV caches.
  void clearCaches();
  void clearPrefillCache();
  void clearDecodeCache();

  // Swap slot0(prefill) and slot1(decode) KV caches in-place.
  // Useful for a 2-slot pipeline where a just-prefilled request becomes the next decode request.
  void swapSlots();

 private:
  struct PDGraphIO {
    int total_len = 0;
    int past_len = 0;
    std::unique_ptr<QnnAOTModule> module;
    std::vector<mllm::Tensor> input_tensors;
    std::vector<mllm::Tensor> output_tensors;
    bool mask_initialized = false;
  };

  PDGraphIO* get_pd_graph(int total_len);
  void init_pd_io(PDGraphIO& g);
  void init_decode_io();
  void prepare_io(PDGraphIO& g, const PrefillSlot& prefill, const DecodeSlot& decode, int32_t prefill_n_update);
  void prepare_decode_io(const DecodeSlot& decode);

  // Copy KV outputs into the two KV caches.
  void update_kv_after_run(int total_len, const PrefillSlot& prefill, const DecodeSlot& decode, int32_t prefill_n_update);
  void update_decode_kv_after_run(const DecodeSlot& decode);

  int64_t sample_token_from_logits(const mllm::Tensor& logits);

  QnnAOTConfig config_;
  std::unique_ptr<QnnAOTModule> decode_module_;

  // Two independent KV caches (slot0=prefill, slot1=decode), but sharing the same *output* buffers
  // to allow both to consume KV results from the single fused graph execution.
  std::unique_ptr<KVCacheManager<uint8_t>> kv_prefill_;
  std::unique_ptr<KVCacheManager<uint8_t>> kv_decode_;

  // Active bindings (may point to external caches).
  KVCacheManager<uint8_t>* active_prefill_ = nullptr;
  KVCacheManager<uint8_t>* active_decode_ = nullptr;

  std::vector<int> pd_total_lens_;
  std::unordered_map<int, PDGraphIO> pd_graphs_;

  std::vector<mllm::Tensor> decode_input_tensors_;
  std::vector<mllm::Tensor> decode_output_tensors_;

  bool warned_kv_aliasing_ = false;
};

}  // namespace mllm::qnn::aot
