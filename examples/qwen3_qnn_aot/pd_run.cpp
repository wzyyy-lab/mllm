#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/core.h>
#include <mllm/mllm.hpp>

#include "mllm/backends/qnn/aot_rt/PDFusionRunner.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"

using mllm::Argparse;
using namespace mllm::qnn::aot;  // NOLINT

static std::vector<int64_t> tensorToI64Vec(const mllm::Tensor& t) {
  MLLM_RT_ASSERT(t.rank() == 2 && t.dtype() == mllm::kInt64);
  std::vector<int64_t> out;
  out.reserve(t.shape()[1]);
  for (int i = 0; i < t.shape()[1]; ++i) { out.push_back(t.ptr<int64_t>()[i]); }
  return out;
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model").help("QNN context path (e.g. *.bin)").def("qwen3_pd_context.bin");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("Tokenizer path").def("tokenizer.json");
  auto& config_path = Argparse::add<std::string>("-c|--config").help("Config path").required(true);
  auto& total_len = Argparse::add<int>("--total_len").help("PD total length (prefill+decode)").def(128);
  auto& context_len = Argparse::add<int>("--context_len").help("Context length").def(1024);
  auto& max_gen = Argparse::add<int>("--max_gen").help("Max decode tokens for overlapped demo").def(32);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  mllm::initQnnBackend(model_path.get());

  auto qwen3_cfg = mllm::models::qwen3::Qwen3Config(config_path.get());
  auto tokenizer = mllm::models::qwen3::Qwen3Tokenizer(tokenizer_path.get());

  QnnAOTConfig cfg;
  cfg.num_layers = qwen3_cfg.num_hidden_layers;
  // KV cache uses num_key_value_heads (GQA).
  cfg.num_heads = qwen3_cfg.num_key_value_heads;
  cfg.head_dim = qwen3_cfg.head_dim;
  cfg.vocab_size = qwen3_cfg.vocab_size;
  cfg.context_len = context_len.get();

  cfg.pd_fusion_enable = true;
  cfg.pd_total_len = total_len.get();
  cfg.pd_prefill_len = cfg.pd_total_len - 1;

  // Keep consistent with existing aot_rt usage.
  cfg.use_int64_token = false;
  cfg.sliding_window = cfg.context_len;

  // Demo-only: assumes you compiled a PD graph named model.0.pd.s{total_len}.
  PDFusionRunner runner(cfg);
  if (!runner.load()) {
    std::cerr << "Failed to load PD fused graph\n";
    return 1;
  }

  std::string prompt_a;
  std::string prompt_b;
  fmt::print("Prompt A (decode target): ");
  std::getline(std::cin, prompt_a);
  fmt::print("Prompt B (prefill to overlap): ");
  std::getline(std::cin, prompt_b);

  auto input_a = tokenizer.convertMessage({.prompt = prompt_a});
  auto input_b = tokenizer.convertMessage({.prompt = prompt_b});
  auto tokens_a = tensorToI64Vec(input_a["sequence"]);
  auto tokens_b = tensorToI64Vec(input_b["sequence"]);

  // 1) Prefill prompt A into slot0(prefill), decode inactive.
  int64_t a_pos = 0;
  int64_t a_start = 0;
  std::optional<int64_t> a_next_token;
  while (a_pos < (int64_t)tokens_a.size()) {
    PDFusionRunner::PrefillSlot p;
    p.active = true;
    p.prompt_tokens = tokens_a;
    p.prompt_pos = a_pos;
    p.start_pos = a_start;

    PDFusionRunner::DecodeSlot d;
    d.active = false;

    auto res = runner.step(p, d);
    const int64_t n_update = std::min<int64_t>(cfg.pd_prefill_len, (int64_t)tokens_a.size() - a_pos);
    a_pos += n_update;
    a_start += n_update;
    if (a_pos >= (int64_t)tokens_a.size()) { a_next_token = res.prefill_next_token; }
  }
  if (!a_next_token.has_value()) {
    std::cerr << "Prefill A failed to produce next token\n";
    return 1;
  }
  {
    uint64_t tok = (uint64_t)*a_next_token;
    std::wstring wstr = tokenizer.detokenize(tok);
    std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
    std::cout << str << std::flush;
  }

  // 2) Promote A to decode slot.
  runner.swapSlots();

  // 3) Overlap: prefill prompt B while decoding tokens for A.
  uint64_t cur_token = (uint64_t)*a_next_token;
  int64_t decode_pos = (int64_t)tokens_a.size();

  std::unordered_set<uint64_t> eos_ids;
  eos_ids.insert(151643);
  eos_ids.insert(151645);

  int64_t b_pos = 0;
  int64_t b_start = 0;
  std::optional<uint64_t> b_first_token;
  bool a_done = false;

  for (int step = 0; step < max_gen.get(); ++step) {
    PDFusionRunner::PrefillSlot p;
    p.active = b_pos < (int64_t)tokens_b.size();
    if (p.active) {
      p.prompt_tokens = tokens_b;
      p.prompt_pos = b_pos;
      p.start_pos = b_start;
    }

    PDFusionRunner::DecodeSlot d;
    d.active = true;
    d.cur_token = cur_token;
    d.start_pos = decode_pos;

    // If we have no prefill work to fuse, fall back to the 1-token graph to avoid wasting PD rows.
    std::optional<int64_t> decode_next;
    std::optional<int64_t> prefill_next;
    if (p.active) {
      const int64_t remain = (int64_t)tokens_b.size() - b_pos;
      const int64_t n_update = std::min<int64_t>(cfg.pd_prefill_len, remain);
      const bool will_finish = (b_pos + n_update) >= (int64_t)tokens_b.size();

      auto res = runner.step(p, d);
      decode_next = res.decode_next_token;
      prefill_next = res.prefill_next_token;

      b_pos += n_update;
      b_start += n_update;
      if (will_finish && prefill_next.has_value() && !b_first_token.has_value()) {
        b_first_token = (uint64_t)*prefill_next;
      }
    } else {
      decode_next = runner.decodeOnly(d);
    }

    if (decode_next.has_value()) {
      uint64_t next = (uint64_t)*decode_next;
      std::wstring wstr = tokenizer.detokenize(next);
      std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
      std::cout << str << std::flush;

      cur_token = next;
      decode_pos += 1;
      if (eos_ids.count(next)) {
        a_done = true;
        break;
      }
    } else {
      break;
    }
  }

  // If B has been fully prefilling, promote it to decode after A finishes and decode-only generate a few tokens.
  if (a_done && b_pos >= (int64_t)tokens_b.size()) {
    // Promote B KV into decode slot.
    runner.swapSlots();
    runner.clearPrefillCache();

    // B's first token was computed during its prefill (TTFT); we delay printing until after A finishes in this demo.
    if (!b_first_token.has_value()) {
      std::cout << "\n";
      return 0;
    }

    uint64_t b_cur = *b_first_token;
    int64_t b_decode_pos = (int64_t)tokens_b.size();

    // Emit B's first token.
    {
      std::wstring wstr = tokenizer.detokenize(b_cur);
      std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
      std::cout << "\n" << str << std::flush;
    }

    for (int step = 0; step < max_gen.get(); ++step) {
      PDFusionRunner::DecodeSlot d;
      d.active = true;
      d.cur_token = b_cur;
      d.start_pos = b_decode_pos;

      auto next = runner.decodeOnly(d);
      if (!next.has_value()) break;
      uint64_t tok = (uint64_t)*next;
      std::wstring wstr = tokenizer.detokenize(tok);
      std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
      std::cout << str << std::flush;
      b_cur = tok;
      b_decode_pos += 1;
      if (eos_ids.count(tok)) break;
    }
  }

  std::cout << "\n";
  return 0;
});
