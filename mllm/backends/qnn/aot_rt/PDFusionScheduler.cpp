// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/PDFusionScheduler.hpp"

#include <algorithm>

#include "mllm/engine/Context.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

namespace {
QnnAOTConfig normalizeConfig(QnnAOTConfig cfg) {
  cfg.pd_fusion_enable = true;
  if (cfg.pd_total_len <= 1) { cfg.pd_total_len = 128; }
  if (cfg.pd_prefill_len <= 0 || cfg.pd_prefill_len >= cfg.pd_total_len) { cfg.pd_prefill_len = cfg.pd_total_len - 1; }
  cfg.max_ar_len = std::max(cfg.max_ar_len, std::max(cfg.pd_total_len, 1));
  cfg.max_cache_len = cfg.context_len - 1;
  return cfg;
}

QnnAOTConfig normalizeConfigWithPdLens(QnnAOTConfig cfg, const std::vector<int>& pd_total_lens) {
  cfg = normalizeConfig(cfg);
  if (!pd_total_lens.empty()) {
    const int max_len = *std::max_element(pd_total_lens.begin(), pd_total_lens.end());
    if (max_len > 1) {
      cfg.pd_total_len = max_len;
      cfg.pd_prefill_len = max_len - 1;
      cfg.max_ar_len = std::max(cfg.max_ar_len, max_len);
    }
  }
  return cfg;
}
}  // namespace

PDFusionScheduler::PDFusionScheduler(QnnAOTConfig config) : config_(normalizeConfig(config)), runner_(config_) {}

PDFusionScheduler::PDFusionScheduler(QnnAOTConfig config, std::vector<int> pd_total_lens)
    : config_(normalizeConfigWithPdLens(config, pd_total_lens)), runner_(config_), pd_total_lens_(std::move(pd_total_lens)) {
  std::sort(pd_total_lens_.begin(), pd_total_lens_.end());
  pd_total_lens_.erase(std::unique(pd_total_lens_.begin(), pd_total_lens_.end()), pd_total_lens_.end());
}

bool PDFusionScheduler::load() {
  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  if (!backend) {
    MLLM_ERROR("QNN Backend not found");
    return false;
  }
  allocator_ = backend->allocator().get();

  return pd_total_lens_.empty() ? runner_.load() : runner_.load(pd_total_lens_);
}

int64_t PDFusionScheduler::submit(std::vector<int64_t> prompt_tokens) {
  const int64_t id = next_id_++;
  RequestState req;
  req.id = id;
  req.prompt_tokens = std::move(prompt_tokens);
  req.prompt_pos = 0;
  req.prefill_done = false;
  req.finished = false;
  requests_.emplace(id, std::move(req));
  prefill_queue_.push_back(id);
  return id;
}

const PDFusionScheduler::RequestState* PDFusionScheduler::get(int64_t id) const {
  auto it = requests_.find(id);
  if (it == requests_.end()) return nullptr;
  return &it->second;
}

int64_t PDFusionScheduler::pick_prefill_id_not_equal(int64_t decode_id) const {
  for (auto id : prefill_queue_) {
    if (id != decode_id) return id;
  }
  return -1;
}

void PDFusionScheduler::ensure_request_cache(RequestState& req) {
  if (req.kv) return;
  req.kv = std::make_unique<KVCacheManager<uint8_t>>(config_);
  // Initialize with pd_total_len layout; we will rearrangeCache() as needed before each execution.
  req.kv->initCache(allocator_, config_.max_ar_len);
}

void PDFusionScheduler::on_prefill_progress(RequestState& req, int32_t n_update, const std::optional<int64_t>& next_token) {
  req.prompt_pos += n_update;
  if (req.prompt_pos >= (int64_t)req.prompt_tokens.size()) {
    req.prefill_done = true;
    if (next_token.has_value()) {
      req.cur_token = (uint64_t)*next_token;
      req.decode_pos = (int64_t)req.prompt_tokens.size();
    }
  }
}

void PDFusionScheduler::on_decode_progress(RequestState& req, const std::optional<int64_t>& next_token) {
  if (!next_token.has_value()) return;
  const uint64_t next = (uint64_t)*next_token;
  req.cur_token = next;
  req.decode_pos += 1;
  if (!eos_ids_.empty() && eos_ids_.count(next)) { req.finished = true; }
}

PDFusionScheduler::TickResult PDFusionScheduler::tick() {
  TickResult out;
  if (prefill_queue_.empty() && decode_queue_.empty()) { return out; }

  // Pick decode (oldest) if any.
  const int64_t decode_id = decode_queue_.empty() ? -1 : decode_queue_.front();
  // Pick prefill (oldest) but not equal to decode_id.
  const int64_t prefill_id = prefill_queue_.empty() ? -1 : (decode_id >= 0 ? pick_prefill_id_not_equal(decode_id) : prefill_queue_.front());

  // Case A: decode only (no prefill candidate) -> run s1 graph.
  if (decode_id >= 0 && prefill_id < 0) {
    auto& dec = requests_.at(decode_id);
    ensure_request_cache(dec);

    runner_.bindCaches(nullptr, dec.kv.get());
    PDFusionRunner::DecodeSlot d;
    d.active = true;
    d.cur_token = dec.cur_token;
    d.start_pos = dec.decode_pos;

    auto next = runner_.decodeOnly(d);
    on_decode_progress(dec, next);
    if (next.has_value()) { out.decode_emitted = std::make_pair(decode_id, (uint64_t)*next); }

    if (dec.finished) { decode_queue_.pop_front(); }
    return out;
  }

  // Case B: prefill only (no decode) -> run PD graph with decode inactive.
  if (decode_id < 0 && prefill_id >= 0) {
    auto& pre = requests_.at(prefill_id);
    ensure_request_cache(pre);

    runner_.bindCaches(pre.kv.get(), nullptr);
    PDFusionRunner::PrefillSlot p;
    p.active = true;
    p.prompt_tokens = pre.prompt_tokens;
    p.prompt_pos = pre.prompt_pos;
    p.start_pos = pre.prompt_pos;

    PDFusionRunner::DecodeSlot d;
    d.active = false;

    const int32_t remain = (int32_t)std::max<int64_t>(0, (int64_t)pre.prompt_tokens.size() - pre.prompt_pos);
    const int32_t n_update = std::min<int32_t>(config_.pd_prefill_len, remain);

    auto res = runner_.step(p, d);
    on_prefill_progress(pre, n_update, res.prefill_next_token);

    if (pre.prefill_done) {
      // move prefill -> decode
      auto it = std::find(prefill_queue_.begin(), prefill_queue_.end(), prefill_id);
      if (it != prefill_queue_.end()) prefill_queue_.erase(it);
      decode_queue_.push_back(prefill_id);
      if (res.prefill_next_token.has_value()) { out.prefill_emitted = std::make_pair(prefill_id, (uint64_t)*res.prefill_next_token); }
    }
    return out;
  }

  // Case C: fused PD (one prefill + one decode).
  if (decode_id >= 0 && prefill_id >= 0) {
    auto& pre = requests_.at(prefill_id);
    auto& dec = requests_.at(decode_id);
    ensure_request_cache(pre);
    ensure_request_cache(dec);

    runner_.bindCaches(pre.kv.get(), dec.kv.get());

    PDFusionRunner::PrefillSlot p;
    p.active = true;
    p.prompt_tokens = pre.prompt_tokens;
    p.prompt_pos = pre.prompt_pos;
    p.start_pos = pre.prompt_pos;

    PDFusionRunner::DecodeSlot d;
    d.active = true;
    d.cur_token = dec.cur_token;
    d.start_pos = dec.decode_pos;

    const int32_t remain = (int32_t)std::max<int64_t>(0, (int64_t)pre.prompt_tokens.size() - pre.prompt_pos);
    const int32_t n_update = std::min<int32_t>(config_.pd_prefill_len, remain);
    const bool will_finish = (pre.prompt_pos + n_update) >= (int64_t)pre.prompt_tokens.size();

    auto res = runner_.step(p, d);

    on_prefill_progress(pre, n_update, res.prefill_next_token);
    on_decode_progress(dec, res.decode_next_token);

    if (res.decode_next_token.has_value()) { out.decode_emitted = std::make_pair(decode_id, (uint64_t)*res.decode_next_token); }
    if (will_finish && res.prefill_next_token.has_value()) { out.prefill_emitted = std::make_pair(prefill_id, (uint64_t)*res.prefill_next_token); }

    if (dec.finished) { decode_queue_.pop_front(); }

    if (pre.prefill_done) {
      auto it = std::find(prefill_queue_.begin(), prefill_queue_.end(), prefill_id);
      if (it != prefill_queue_.end()) prefill_queue_.erase(it);
      decode_queue_.push_back(prefill_id);
    }
    return out;
  }

  return out;
}

}  // namespace mllm::qnn::aot
