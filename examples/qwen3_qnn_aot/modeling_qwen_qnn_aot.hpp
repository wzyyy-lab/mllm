// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/backends/qnn/CustomLayers.hpp"

namespace mllm::models::qwen3 {

namespace ptq {

Tensor QDQ(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  std::string scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  std::string zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  if (m->getModuleName().empty()) {
    scale_name = qdq_name_in_pytorch + ".fake_quant.scale";
    zp_name = qdq_name_in_pytorch + ".fake_quant.zero_point";
  } else {
    scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
    zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";
  }

  switch (in.dtype()) {
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    // For Constant!
    case kFloat32: {
      MLLM_RT_ASSERT_EQ(in.rank(), 1);
      MLLM_RT_ASSERT_EQ(in.size(-1), 1);
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

Tensor QDQ_KV(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  // The inputs is int8 sym. which means zero_point should be changed.
  switch (in.dtype()) {
    case kUInt8PerTensorSym: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 0);

      // Is 128! not 127!
      auto new_zp = Tensor::constant(128, kInt32).setName(zp_name).setMemType(kParamsNormal);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", new_zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

Tensor QDQ_ROPE(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  (void)in.__unsafeSetDType(kUInt16PerTensorAsy);

  switch (in.dtype()) {
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }

  return in;
}

}  // namespace ptq

Tensor rotateHalf(Tensor x, nn::Module* m, const std::string& qdq_name_in_pytorch) {  // NOLINT
  // X is [x, x, x, D]
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({ptq::QDQ(m, -x2, qdq_name_in_pytorch), x1}, -1);
}

using vi32 = std::vector<int32_t>;
#define CONV2D_PROPERTY vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G32

// Using Conv2D to replace Linear.
// Conv2D Filter Weight is [1, 1, In, Out]
// Conv2D Activation is [N, H=1, W=Seq, In]

class Qwen3MLP final : public nn::Module {
  nn::Conv2D gate_proj_;
  nn::Conv2D up_proj_;
  nn::Conv2D down_proj_;
  nn::SiLU silu_;
  int hidden_size_;
  int intermediate_size_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Conv2D>("gate_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Conv2D>("up_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY);
    down_proj_ = reg<nn::Conv2D>("down_proj", cfg.intermediate_size, cfg.hidden_size, CONV2D_PROPERTY);
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = ptq::QDQ(this, x, "up_proj_input_qdq");
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto up_result = ptq::QDQ(this, up_proj_(x), "up_proj_output_qdq").view({1, -1, intermediate_size_}, true);
    auto gate_result = ptq::QDQ(this, gate_proj_(x), "gate_proj_output_qdq").view({1, -1, intermediate_size_}, true);

    // SiLU
    gate_result = ptq::QDQ(this, (gate_result * ptq::QDQ(this, nn::functional::sigmoid(gate_result), "sigmoid_output_qdq")),
                           "act_output_qdq");

    auto o = ptq::QDQ(this, gate_result * up_result, "down_proj_input_qdq");
    o = o.view({1, 1, -1, intermediate_size_}, true);
    o = down_proj_(o).view({1, -1, hidden_size_}, true);

    return {o};
  }
};

class Qwen3Attention final : public nn::Module {
  nn::Conv2D q_proj_;
  nn::Conv2D k_proj_;
  nn::Conv2D v_proj_;
  nn::Conv2D o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;
  nn::qnn::FusedPDAttention fused_pd_attn_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;

 public:
  Qwen3Attention() = default;

  Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = (1.f / sqrtf((float)head_dim_));

    // clang-format off
    q_proj_ = reg<nn::Conv2D>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, CONV2D_PROPERTY);
    k_proj_ = reg<nn::Conv2D>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY);
    v_proj_ = reg<nn::Conv2D>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY);
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, CONV2D_PROPERTY);
    // clang-format on

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
    fused_pd_attn_ = reg<nn::qnn::FusedPDAttention>("fused_pd_attn");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];

    // PD fusion optional IO:
    // inputs[4]=fusion_ctrl, [5..8]=prefill/decode KV
    const bool pd_fusion = inputs.size() >= 9;
    auto fusion_ctrl = pd_fusion ? inputs[4] : Tensor::nil();
    auto past_key = inputs[pd_fusion ? 5 : 4];
    auto past_value = inputs[pd_fusion ? 6 : 5];
    auto past_key_decode = pd_fusion ? inputs[7] : Tensor::nil();
    auto past_value_decode = pd_fusion ? inputs[8] : Tensor::nil();

    const int32_t seq_len = hidden_states.size(1);

    // [B, S, D]
    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    // [B, S, H * D]
    auto query_states = q_proj_(hidden_states);
    auto key_states = k_proj_(hidden_states);
    auto value_states = v_proj_(hidden_states);

    // [B, H, S, D]
    query_states = query_states.view({1, -1, num_attention_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    key_states = key_states.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    value_states = value_states.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);

    // [B, H, S, D]
    query_states = rms_norm_q_(ptq::QDQ(this, query_states, "q_norm_input_qdq"));
    key_states = rms_norm_k_(ptq::QDQ(this, key_states, "k_norm_input_qdq"));

    query_states = ptq::QDQ(this, query_states, "q_norm_output_qdq");
    key_states = ptq::QDQ(this, key_states, "k_norm_output_qdq");

    // [B, H, S, D]
    auto cos = llm_embedding_cos.unsqueeze(1);
    auto sin = llm_embedding_sin.unsqueeze(1);
    query_states =
        ptq::QDQ(this,
                 ptq::QDQ(this, query_states * cos, "q_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(query_states, this, "q_rope_neg_half_qdq") * sin, "q_rope_mul_1_output_qdq"),
                 "q_rope_add_0_output_qdq");
    key_states =
        ptq::QDQ(this,
                 ptq::QDQ(this, key_states * cos, "k_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(key_states, this, "k_rope_neg_half_qdq") * sin, "k_rope_mul_1_output_qdq"),
                 "k_rope_add_0_output_qdq");

    // De-quantization and quantization again
    key_states = key_states.to(kFloat32);
    key_states = key_states.to(kUInt8PerTensorSym);
    key_states = ptq::QDQ_KV(this, key_states, "k_cast_to_int8_qdq");

    // [B, H, D, S]
    key_states = key_states.transpose(2, 3);

    // Handle KV Cache
    value_states = ptq::QDQ(this, value_states, "v_cast_to_int16_qdq");
    value_states = value_states.to(kFloat32);
    value_states = value_states.to(kUInt8PerTensorSym);
    value_states = ptq::QDQ_KV(this, value_states, "v_cast_to_int8_qdq");

    if (!pd_fusion) {
      // past_key is token-major cache: [B, H_kv, L, D], but attention math expects [B, H_kv, D, L]
      auto past_key_dm = past_key.transpose(2, 3);
      auto kh = nn::functional::concat({past_key_dm, key_states}, -1);  // [B, H_kv, D, S]
      auto vh = nn::functional::concat({past_value, value_states}, 2);  // [B, H, S, D]

      // Repeat
      kh = kh.repeat(num_key_value_groups_, 1);
      vh = vh.repeat(num_key_value_groups_, 1);

      // Attn
      auto attn = ptq::QDQ(this, nn::functional::matmul(query_states, kh), "qk_matmul_output_qdq");
      auto scale = Tensor::constant(scale_, kFloat32);
      scale = ptq::QDQ(this, scale, "scaling_qdq");
      attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq");

      // Masked Softmax
      // NOTE: attention_mask convention from aot_rt/KVCacheManager:
      //   allowed = 65535, masked = 0
      auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq");
      auto minus_value = Tensor::constant(-20, kFloat32);
      minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq");
      auto attn_vv = ptq::QDQ(this, attn_min.addConstant(minus_value), "minus_0_output_qdq");
      attn = nn::functional::where(causal_mask.equal(0.f), attn_vv, attn);
      attn = ptq::QDQ(this, attn, "where_attn_qdq");
      attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq");
      auto y = ptq::QDQ(this, nn::functional::matmul(attn, vh), "attn_value_matmul_output_qdq");
      y = y.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
      y = o_proj_(y).view({1, -1, hidden_size_}, true);
      return {y, key_states, value_states};
    }

    // Fused PD attention (Kernel C) can be enabled at build time.
    // Default is off because the reference kernel is correctness-first and not performance-optimized.
#ifndef MLLM_ENABLE_FUSED_PD_ATTENTION
#define MLLM_ENABLE_FUSED_PD_ATTENTION 0
#endif
    constexpr bool kEnableFusedPDAttention = (MLLM_ENABLE_FUSED_PD_ATTENTION != 0);
    if (kEnableFusedPDAttention) {
      // Provide explicit quant params to avoid implicit dequant/cast ops in QNN.
      auto pf = this->getTopParameterFile();
      const std::string prefix = this->getModuleName().empty() ? std::string("") : (this->getModuleName() + ".");

      auto q_scale = pf->pull(prefix + "q_rope_add_0_output_qdq.fake_quant.scale");
      auto q_zp = pf->pull(prefix + "q_rope_add_0_output_qdq.fake_quant.zero_point");

      auto k_scale = pf->pull(prefix + "k_cast_to_int8_qdq.fake_quant.scale");
      auto v_scale = pf->pull(prefix + "v_cast_to_int8_qdq.fake_quant.scale");
      auto k_zp = Tensor::constant(128, kInt32);
      auto v_zp = Tensor::constant(128, kInt32);

      auto out_scale = pf->pull(prefix + "attn_value_matmul_output_qdq.fake_quant.scale");
      auto out_zp = pf->pull(prefix + "attn_value_matmul_output_qdq.fake_quant.zero_point");

      // q: [B, H_attn, S, D]
      // k_curr: [B, H_kv, D, S]
      // v_curr: [B, H_kv, S, D]
      // past_k/v_*: from inputs (two slots)
      // attention_mask: inputs[3] (runtime-provided attention_mask)
      auto attn_out =
          fused_pd_attn_(query_states, key_states, value_states, past_key, past_value, past_key_decode, past_value_decode, causal_mask,
                         fusion_ctrl, q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp)[0];
      // Attach output quant params for downstream QNN quant recipe (even though the kernel already outputs quantized data).
      attn_out = ptq::QDQ(this, attn_out, "attn_value_matmul_output_qdq");
      // Shape expected by downstream o_proj: [1,1,S,H*D]
      auto y_cat = attn_out.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
      auto y = o_proj_(y_cat).view({1, -1, hidden_size_}, true);
      return {y, key_states, value_states};
    }

    // PD fusion: split queries into [0..S-2] (prefill) and [S-1] (decode), each with its own past KV.
    MLLM_RT_ASSERT(seq_len >= 2);
    const int32_t prefill_len = seq_len - 1;
    const int32_t decode_idx = seq_len - 1;

    // Prefill branch
    auto q0 = query_states.slice({kAll, kAll, {0, prefill_len}, kAll}, /*ssa=*/true);
    auto m0 = causal_mask.slice({kAll, kAll, {0, prefill_len}, kAll}, /*ssa=*/true);
    // past_key is token-major cache: [B, H_kv, L, D], but attention math expects [B, H_kv, D, L]
    auto past_key_dm = past_key.transpose(2, 3);
    auto kh0 = nn::functional::concat({past_key_dm, key_states}, -1).repeat(num_key_value_groups_, 1);
    auto vh0 = nn::functional::concat({past_value, value_states}, 2).repeat(num_key_value_groups_, 1);

    auto attn0 = ptq::QDQ(this, nn::functional::matmul(q0, kh0), "qk_matmul_output_qdq");
    auto scale = Tensor::constant(scale_, kFloat32);
    scale = ptq::QDQ(this, scale, "scaling_qdq");
    attn0 = ptq::QDQ(this, attn0.mulConstant(scale), "mul_0_output_qdq");
    auto attn0_min = ptq::QDQ(this, attn0.min(-1, true), "reduce_min_output_qdq");
    auto minus_value = Tensor::constant(-20, kFloat32);
    minus_value = ptq::QDQ(this, minus_value, "neg_20_qdq");
    auto attn0_vv = ptq::QDQ(this, attn0_min.addConstant(minus_value), "minus_0_output_qdq");
    attn0 = nn::functional::where(m0.equal(0.f), attn0_vv, attn0);
    attn0 = ptq::QDQ(this, attn0, "where_attn_qdq");
    attn0 = ptq::QDQ(this, nn::functional::softmax(attn0, -1), "softmax_output_qdq");
    auto y0 = ptq::QDQ(this, nn::functional::matmul(attn0, vh0), "attn_value_matmul_output_qdq");
    y0 = y0.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);

    // Decode branch
    auto q1 = query_states.slice({kAll, kAll, {decode_idx, decode_idx + 1}, kAll}, /*ssa=*/true);
    auto m1 = causal_mask.slice({kAll, kAll, {decode_idx, decode_idx + 1}, kAll}, /*ssa=*/true);
    // past_key_decode is token-major cache: [B, H_kv, L, D], but attention math expects [B, H_kv, D, L]
    auto past_key_decode_dm = past_key_decode.transpose(2, 3);
    auto kh1 = nn::functional::concat({past_key_decode_dm, key_states}, -1).repeat(num_key_value_groups_, 1);
    auto vh1 = nn::functional::concat({past_value_decode, value_states}, 2).repeat(num_key_value_groups_, 1);

    auto attn1 = ptq::QDQ(this, nn::functional::matmul(q1, kh1), "qk_matmul_output_qdq");
    attn1 = ptq::QDQ(this, attn1.mulConstant(scale), "mul_0_output_qdq");
    auto attn1_min = ptq::QDQ(this, attn1.min(-1, true), "reduce_min_output_qdq");
    auto attn1_vv = ptq::QDQ(this, attn1_min.addConstant(minus_value), "minus_0_output_qdq");
    attn1 = nn::functional::where(m1.equal(0.f), attn1_vv, attn1);
    attn1 = ptq::QDQ(this, attn1, "where_attn_qdq");
    attn1 = ptq::QDQ(this, nn::functional::softmax(attn1, -1), "softmax_output_qdq");
    auto y1 = ptq::QDQ(this, nn::functional::matmul(attn1, vh1), "attn_value_matmul_output_qdq");
    y1 = y1.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);

    // Pack [prefill_len + 1] back to a single [1, 1, S, H*D] tensor, then apply o_proj once.
    auto y_cat = nn::functional::concat({y0, y1}, 2);
    auto y = o_proj_(y_cat).view({1, -1, hidden_size_}, true);

    // Keep fusion_ctrl in the attention subgraph for future custom-kernel lowering (no-op numerically).
    auto ctrl0 = fusion_ctrl[{{0}}].to(kFloat32).view({1});
    auto ctrl_zero = ctrl0 - ctrl0;
    y = y.addConstant(ctrl_zero);

    return {y, key_states, value_states};
  }

  int layer_idx_;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Attention self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3Decoder() = default;

  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];

    const bool pd_fusion = inputs.size() >= 9;
    auto fusion_ctrl = pd_fusion ? inputs[4] : Tensor::nil();
    auto past_key = inputs[pd_fusion ? 5 : 4];
    auto past_value = inputs[pd_fusion ? 6 : 5];
    auto past_key_decode = pd_fusion ? inputs[7] : Tensor::nil();
    auto past_value_decode = pd_fusion ? inputs[8] : Tensor::nil();

    auto hidden_states = inputs[0];
    hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq");
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto _ = pd_fusion ? self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, fusion_ctrl, past_key,
                                   past_value, past_key_decode, past_value_decode)
                       : self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, past_key, past_value);
    hidden_states = _[0];
    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, hidden_states, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    hidden_states = mlp_(hidden_states)[0];
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");
    return {hidden_states, _[1], _[2]};
  }
};

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;
  nn::Param rope_sin_;
  nn::Param rope_cos_;
  std::vector<nn::qnn::PDKVCacheUpdate> kv_update_prefill_;
  std::vector<nn::qnn::PDKVCacheUpdate> kv_update_decode_;
  std::vector<nn::qnn::PDKVCacheUpdate> kv_update_single_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;

 public:
  Qwen3Text() = default;

  Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    rope_sin_ = reg<nn::Param>("mllm_max_sin_embedding", "model.mllm_max_sin_embedding");
    rope_cos_ = reg<nn::Param>("mllm_max_cos_embedding", "model.mllm_max_cos_embedding");

    kv_update_prefill_.reserve(cfg.num_hidden_layers);
    kv_update_decode_.reserve(cfg.num_hidden_layers);
    kv_update_single_.reserve(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      kv_update_prefill_.push_back(reg<nn::qnn::PDKVCacheUpdate>("pd_kv_update_prefill_" + std::to_string(i)));
      kv_update_decode_.push_back(reg<nn::qnn::PDKVCacheUpdate>("pd_kv_update_decode_" + std::to_string(i)));
      kv_update_single_.push_back(reg<nn::qnn::PDKVCacheUpdate>("kv_update_" + std::to_string(i)));
    }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = embedding_(inputs[0]);

    // Quantization
    x = x.to(kUInt16PerTensorAsy);

    const auto& position_ids = inputs[1];
    auto causal_mask = inputs[2];

    // PD fusion IO layout (optional):
    //   inputs = [sequence, position_ids, causal_mask, fusion_ctrl,
    //             prefill_keys(L), prefill_values(L), decode_keys(L), decode_values(L)]
    // Legacy IO layout:
    //   inputs = [sequence, position_ids, causal_mask, keys(L), values(L)]
    const bool has_pd_fusion_io = (int32_t)inputs.size() >= 4 + 4 * num_hidden_layers_;
    const int32_t base_kv = has_pd_fusion_io ? 4 : 3;
    const int32_t base_prefill_k = base_kv;
    const int32_t base_prefill_v = base_prefill_k + num_hidden_layers_;
    const int32_t base_decode_k = base_prefill_v + num_hidden_layers_;
    const int32_t base_decode_v = base_decode_k + num_hidden_layers_;

    // clang-format off
    auto llm_embedding_sin = nn::functional::gather(ptq::QDQ_ROPE(this, rope_sin_(), "sin_embedding_input_qdq"), 1, position_ids);
    auto llm_embedding_cos = nn::functional::gather(ptq::QDQ_ROPE(this, rope_cos_(), "cos_embedding_input_qdq"), 1, position_ids);
    // clang-format on

    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    std::vector<Tensor> updated_kv;
    Tensor fusion_ctrl = has_pd_fusion_io ? inputs[3] : Tensor::nil();
    for (auto [index, block] : enumerate(blocks)) {
      auto pk0 = inputs[base_prefill_k + index];
      auto pv0 = inputs[base_prefill_v + index];
      auto pk1 = has_pd_fusion_io ? inputs[base_decode_k + index] : Tensor::nil();
      auto pv1 = has_pd_fusion_io ? inputs[base_decode_v + index] : Tensor::nil();
      auto _ = has_pd_fusion_io ? block(x, llm_embedding_sin, llm_embedding_cos, causal_mask, fusion_ctrl, pk0, pv0, pk1, pv1)
                                : block(x, llm_embedding_sin, llm_embedding_cos, causal_mask, pk0, pv0);
      x = _[0];
      keys.push_back(_[1]);
      values.push_back(_[2]);

      // KV cache update via custom op (lowered to LLaMAPackage::PDKVCacheUpdate).
      const int32_t seq_len = x.size(1);
      const int32_t decode_idx = seq_len - 1;

      if (has_pd_fusion_io) {
        auto prefill_enable = fusion_ctrl[{{0}}];
        auto decode_enable = fusion_ctrl[{{1}}];
        auto prefill_n_update = fusion_ctrl[{{2}}];

        // Use fusion_ctrl lengths instead of position_ids so runtime can freely choose packing layout.
        // fusion_ctrl[4]=prefill_n_past, fusion_ctrl[5]=decode_n_past
        auto prefill_n_past = fusion_ctrl[{{4}}];
        auto decode_n_past = fusion_ctrl[{{5}}];

        // If prefill tokens are right-aligned within [0..decode_idx-1], their source offset is:
        //   src0 = decode_idx - prefill_n_update
        // This keeps the last prefill token at (decode_idx-1), which enables fixed "last-2 rows" gathering later.
        auto src0 = Tensor::constant(decode_idx, kInt32) - prefill_n_update;
        auto src1 = Tensor::constant(decode_idx, kInt32);
        auto one = Tensor::constant(1, kInt32);

        auto [k0_out, v0_out] = kv_update_prefill_[index](x, pk0, pv0, keys.back(), values.back(), prefill_n_past, src0,
                                                          prefill_n_update, prefill_enable);
        k0_out.setName("updated_past_key_prefill_" + std::to_string(index));
        v0_out.setName("updated_past_value_prefill_" + std::to_string(index));
        updated_kv.push_back(k0_out);
        updated_kv.push_back(v0_out);

        auto [k1_out, v1_out] =
            kv_update_decode_[index](x, pk1, pv1, keys.back(), values.back(), decode_n_past, src1, one, decode_enable);
        k1_out.setName("updated_past_key_decode_" + std::to_string(index));
        v1_out.setName("updated_past_value_decode_" + std::to_string(index));
        updated_kv.push_back(k1_out);
        updated_kv.push_back(v1_out);
      } else {
        auto enable = Tensor::constant(1, kInt32);
        auto n_past = position_ids[{{0}}];
        auto src = Tensor::constant(0, kInt32);
        auto n_update = Tensor::constant(seq_len, kInt32);

        auto [k_out, v_out] = kv_update_single_[index](x, pk0, pv0, keys.back(), values.back(), n_past, src, n_update, enable);
        k_out.setName("updated_past_key_" + std::to_string(index));
        v_out.setName("updated_past_value_" + std::to_string(index));
        updated_kv.push_back(k_out);
        updated_kv.push_back(v_out);
      }
    }

    x = norm_(ptq::QDQ(this, x, "norm_input_qdq"));
    x = x.view({1, 1, -1, hidden_size_}, true);

    // Keep fusion_ctrl in the traced graph (even if not used by kernels yet).
    if (has_pd_fusion_io) {
      auto ctrl0 = inputs[3][{{0}}].to(kFloat32).view({1});
      auto ctrl_zero = ctrl0 - ctrl0;
      x = x.addConstant(ctrl_zero);
    }

    auto ret = std::vector<Tensor>{x};
    for (const auto& item : keys) { ret.push_back(item); }
    for (const auto& item : values) { ret.push_back(item); }
    for (const auto& item : updated_kv) { ret.push_back(item); }

    return ret;
  }
};

class Qwen3ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen3Text>("model", cfg);

    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, CONV2D_PROPERTY);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    // Things we need to return
    ir::IRContext::ptr_t llm_ir = nullptr;

    const bool use_runtime_io_names = input.count("input_ids") && input.count("attention_mask");
    auto sequence = use_runtime_io_names ? input.at("input_ids") : input.at("sequence");
    auto causal_mask = use_runtime_io_names ? input.at("attention_mask") : input.at("causal_mask");

    std::vector<Tensor> kv_caches;
    std::vector<Tensor> kv_caches_decode;
    const bool has_pd_fusion_io = input.count("fusion_ctrl") > 0;

    // Optional fusion control tensor (shape [6], int32)
    Tensor fusion_ctrl = Tensor::nil();
    if (has_pd_fusion_io) { fusion_ctrl = input.at("fusion_ctrl"); }

    if (has_pd_fusion_io) {
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto k0 = "past_key_prefill_" + std::to_string(i);
        auto v0 = "past_value_prefill_" + std::to_string(i);
        auto k1 = "past_key_decode_" + std::to_string(i);
        auto v1 = "past_value_decode_" + std::to_string(i);
        if (!input.count(k0) || !input.count(v0) || !input.count(k1) || !input.count(v1)) {
          throw std::runtime_error("Missing PD fusion KV cache for layer " + std::to_string(i));
        }
        kv_caches.push_back(input.at(k0));
      }
      for (int i = 0; i < cfg.num_hidden_layers; ++i) { kv_caches.push_back(input.at("past_value_prefill_" + std::to_string(i))); }
      for (int i = 0; i < cfg.num_hidden_layers; ++i) { kv_caches_decode.push_back(input.at("past_key_decode_" + std::to_string(i))); }
      for (int i = 0; i < cfg.num_hidden_layers; ++i) { kv_caches_decode.push_back(input.at("past_value_decode_" + std::to_string(i))); }
    } else {
      // Legacy single-KV-cache inputs.
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto past_key_name = "past_key_" + std::to_string(i);
        if (!input.count(past_key_name)) { throw std::runtime_error("Missing KV cache for layer " + std::to_string(i)); }
        kv_caches.push_back(input.at(past_key_name));
      }
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto past_value_name = "past_value_" + std::to_string(i);
        if (!input.count(past_value_name)) { throw std::runtime_error("Missing KV cache for layer " + std::to_string(i)); }
        kv_caches.push_back(input.at(past_value_name));
      }
    }

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      // Use existing position_ids (runtime-provided).
      position_ids = input.at("position_ids");
      // Accept both [S] and [1, S]. Normalize to [S].
      if (position_ids.rank() == 2 && position_ids.size(0) == 1) { position_ids = position_ids.squeeze(0); }
    } else {
      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({seq_len}, kInt32, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int32_t>();
      for (int s = 0; s < seq_len; ++s) { position_ids_ptr[s] = s; }
    }

    ir::lowlevel::traceStart();

    // Build inputs for llm: sequence, llm_embedding_sin, llm_embedding_cos, causal_mask, then all KV caches
    std::vector<Tensor> llm_inputs = {sequence, position_ids, causal_mask};
    if (has_pd_fusion_io) {
      llm_inputs.push_back(fusion_ctrl);
      llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());
      llm_inputs.insert(llm_inputs.end(), kv_caches_decode.begin(), kv_caches_decode.end());
    } else {
      llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());
    }

    sequence = llm(llm_inputs)[0];
    // P0-1: reduce lm_head compute and logits bandwidth for PD graphs.
    // For PD graphs (seq_len=N>=2), only the last two rows are ever consumed by runtime:
    // - row N-2: prefill_last (fixed after right-aligned packing)
    // - row N-1: decode
    if (has_pd_fusion_io) {
      MLLM_RT_ASSERT(sequence.rank() == 4);
      const int32_t S = (int32_t)sequence.size(2);
      MLLM_RT_ASSERT(S >= 2);
      sequence = sequence.slice({kAll, kAll, {S - 2, S}, kAll}, /*ssa=*/true);
    }
    sequence = lm_head_(ptq::QDQ(this, sequence, "lm_head_input_qdq"));
    sequence = ptq::QDQ(this, sequence, "lm_head_output_qdq");
    ir::lowlevel::traceComment("    ╔═════╗   ");
    ir::lowlevel::traceComment("   ║  o o  ║  ");
    ir::lowlevel::traceComment("   ║   ▽   ║  ");
    ir::lowlevel::traceComment("   ╚═════╝   ");
    ir::lowlevel::traceComment("    ║   ║     ");
    ir::lowlevel::traceComment("   ╱╩╦╦╩╲    ");
    llm_ir = ir::lowlevel::traceStop();

    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const Qwen3Config& cfg;
  Qwen3Text llm;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
};

}  // namespace mllm::models::qwen3
