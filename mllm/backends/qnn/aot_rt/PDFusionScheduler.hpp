// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/PDFusionRunner.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"

namespace mllm::qnn::aot {

// A minimal ActionFlow-style "packer" for PD fusion:
// - prefill_queue: requests needing prompt prefill (chunked)
// - decode_queue: active decode requests (1 token per step)
//
// Each tick packs at most 1 prefill request and 1 decode request into one PD graph execution.
// If there is no prefill work, it falls back to the 1-token decode graph to avoid wasting PD rows.
class PDFusionScheduler {
 public:
  struct RequestState {
    int64_t id = 0;
    std::vector<int64_t> prompt_tokens;
    int64_t prompt_pos = 0;    // how many prompt tokens are already in KV cache
    int64_t decode_pos = 0;    // KV position of cur_token
    uint64_t cur_token = 0;    // input token for decode step
    bool prefill_done = false;
    bool finished = false;

    std::unique_ptr<KVCacheManager<uint8_t>> kv;
  };

  struct TickResult {
    // Token IDs produced in this tick (decode path). One per decode request.
    std::optional<std::pair<int64_t, uint64_t>> decode_emitted;
    // If a request completes prefill (TTFT), its first token is produced here.
    std::optional<std::pair<int64_t, uint64_t>> prefill_emitted;
  };

  explicit PDFusionScheduler(QnnAOTConfig config);
  PDFusionScheduler(QnnAOTConfig config, std::vector<int> pd_total_lens);

  bool load();

  // Add a new request into prefill_queue and return its id.
  int64_t submit(std::vector<int64_t> prompt_tokens);

  void setEosIds(std::unordered_set<uint64_t> eos_ids) { eos_ids_ = std::move(eos_ids); }

  // Execute one scheduling tick.
  TickResult tick();

  const std::deque<int64_t>& prefillQueue() const { return prefill_queue_; }
  const std::deque<int64_t>& decodeQueue() const { return decode_queue_; }

  const RequestState* get(int64_t id) const;

 private:
  int64_t pick_prefill_id_not_equal(int64_t decode_id) const;

  void ensure_request_cache(RequestState& req);
  void on_prefill_progress(RequestState& req, int32_t n_update, const std::optional<int64_t>& next_token);
  void on_decode_progress(RequestState& req, const std::optional<int64_t>& next_token);

  QnnAOTConfig config_;
  PDFusionRunner runner_;
  std::vector<int> pd_total_lens_;

  mllm::Allocator* allocator_ = nullptr;
  int64_t next_id_ = 1;
  std::unordered_set<uint64_t> eos_ids_;

  std::unordered_map<int64_t, RequestState> requests_;
  std::deque<int64_t> prefill_queue_;
  std::deque<int64_t> decode_queue_;
};

}  // namespace mllm::qnn::aot
