// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"

namespace mllm::qnn::aot {

struct QnnAOTConfig {
  int num_layers = 28;
  // Number of KV heads (num_key_value_heads for GQA/MQA models).
  int num_heads = 12;
  int head_dim = 128;
  int vocab_size = 151936;

  int context_len = 4096;
  int ar_len = 128;  // Chunk size for prefill
  int sliding_window = 0;

  // Prefill-Decode Fusion (PD Fusion)
  // A fixed-shape dual-slot execution: [0..pd_prefill_len-1] for prefill, [pd_total_len-1] for decode(1 token).
  bool pd_fusion_enable = false;
  int pd_total_len = 128;
  int pd_prefill_len = 127;

  // If true, graphs are expected to contain on-device KV-cache update ops (e.g. LLaMAPackage::PDKVCacheUpdate),
  // and runtime should NOT run KVCacheManager::updateCache() on CPU.
  bool kv_update_on_device = false;

  // Derived/Computed
  int max_ar_len = 128;
  int max_cache_len = 4096;

  DataTypes kv_dtype = kUInt8;
  bool use_int64_token = true;
};

}  // namespace mllm::qnn::aot
