// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/nn/Layer.hpp"

// -------------------- Custom QNN Layers --------------------
namespace mllm::nn::qnn {

struct DequantizeAddOpOptions : public BaseOpOptions<DequantizeAddOpOptions> {
  DataTypes dtype;
  int32_t out_channels;
};

struct PDKVCacheUpdateOpOptions : public BaseOpOptions<PDKVCacheUpdateOpOptions> {};
struct FusedPDAttentionOpOptions : public BaseOpOptions<FusedPDAttentionOpOptions> {};
struct FusedPDAttentionK4OpOptions : public BaseOpOptions<FusedPDAttentionK4OpOptions> {};

/**
 * @brief QNN Custom Layer: DequantizeAdd
 *
 * This layer performs dequantization of the input tensor followed by an element-wise addition with a bias tensor.
 * The bias is the previous linear layer's bias. This layer MUST be named with the name of the previous Linear plus
 * ".dequantize" to correctly load the bias during model loading.
 *
 */
class DequantizeAdd : public Layer {
 public:
  DequantizeAdd();

  explicit DequantizeAdd(DataTypes dtype, int32_t out_channels);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

/**
 * @brief QNN Custom Layer: PDKVCacheUpdate
 *
 * Device-side KV cache update used by PD fusion runtime.
 *
 * Expected inputs:
 *   dep, in_k_cache, in_v_cache, present_k, present_v, n_past, src_offset, n_update, enable
 *
 * Outputs:
 *   out_k_cache, out_v_cache
 */
class PDKVCacheUpdate : public Layer {
 public:
  PDKVCacheUpdate();

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

/**
 * @brief QNN Custom Layer: FusedPDAttention
 *
 * A placeholder for a future fused PD attention kernel (prefill+decode in one op).
 *
 * Expected inputs (v0):
 *   q, k_curr, v_curr, past_k_prefill, past_v_prefill, past_k_decode, past_v_decode, attention_mask, fusion_ctrl,
 *   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
 *
 * fusion_ctrl is expected to be shape [6] int32:
 *   [0]=is_prefill_active, [1]=is_decode_active, [2]=prefill_n_update, [3]=reserved,
 *   [4]=prefill_n_past, [5]=decode_n_past
 *
 * Outputs:
 *   attn_out  (same shape/dtype as q)
 */
class FusedPDAttention : public Layer {
 public:
  FusedPDAttention();

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

/**
 * @brief QNN Custom Layer: FusedPDAttentionK4
 *
 * Kernel C v3 variant for PD-k graphs (k=4 decode rows).
 *
 * Expected inputs:
 *   q, k_curr, v_curr,
 *   past_k_prefill, past_v_prefill,
 *   past_k_dec0, past_v_dec0, past_k_dec1, past_v_dec1, past_k_dec2, past_v_dec2, past_k_dec3, past_v_dec3,
 *   attention_mask, fusion_ctrl, decode_past_lens,
 *   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
 *
 * Outputs:
 *   attn_out  (same shape/dtype as q)
 */
class FusedPDAttentionK4 : public Layer {
 public:
  FusedPDAttentionK4();

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn::qnn

// -------------------- Custom QNN Ops --------------------
namespace mllm::qnn {

class DequantizeAddOp final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit DequantizeAddOp(const nn::qnn::DequantizeAddOpOptions& options) : CustomizedOp("DequantizeAdd"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override {}

  // Public accessors for QNN pattern matching
  const Tensor& getWeightTensor() const { return weight_; }
  const nn::qnn::DequantizeAddOpOptions& getOptions() const { return options_; }

 protected:
  nn::qnn::DequantizeAddOpOptions options_;
  Tensor weight_;
};

class DequantizeAddFactory final : public mllm::plugin::interface::CustomizedOpFactory<nn::qnn::DequantizeAddOpOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const nn::qnn::DequantizeAddOpOptions& cargo) override {
    auto p = std::make_shared<DequantizeAddOp>(cargo);
    p->setOpType(opType());
    return p;
  }
};

class PDKVCacheUpdateOp final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit PDKVCacheUpdateOp(const nn::qnn::PDKVCacheUpdateOpOptions& options)
      : CustomizedOp("PDKVCacheUpdate"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& /*ploader*/) override {}

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

 protected:
  nn::qnn::PDKVCacheUpdateOpOptions options_;
};

class PDKVCacheUpdateFactory final : public mllm::plugin::interface::CustomizedOpFactory<nn::qnn::PDKVCacheUpdateOpOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const nn::qnn::PDKVCacheUpdateOpOptions& cargo) override {
    auto p = std::make_shared<PDKVCacheUpdateOp>(cargo);
    p->setOpType(opType());
    return p;
  }
};

class FusedPDAttentionOp final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit FusedPDAttentionOp(const nn::qnn::FusedPDAttentionOpOptions& options)
      : CustomizedOp("FusedPDAttention"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& /*ploader*/) override {}

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

 protected:
  nn::qnn::FusedPDAttentionOpOptions options_;
};

class FusedPDAttentionFactory final
    : public mllm::plugin::interface::CustomizedOpFactory<nn::qnn::FusedPDAttentionOpOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const nn::qnn::FusedPDAttentionOpOptions& cargo) override {
    auto p = std::make_shared<FusedPDAttentionOp>(cargo);
    p->setOpType(opType());
    return p;
  }
};

class FusedPDAttentionK4Op final : public mllm::plugin::interface::CustomizedOp {
 public:
  explicit FusedPDAttentionK4Op(const nn::qnn::FusedPDAttentionK4OpOptions& options)
      : CustomizedOp("FusedPDAttentionK4"), options_(options) {}

  void load(const mllm::ParameterFile::ptr_t& /*ploader*/) override {}

  void trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void forward(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

  void reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) override;

  void setup(const std::vector<mllm::Tensor>& /*inputs*/, std::vector<mllm::Tensor>& /*outputs*/) override {}

 protected:
  nn::qnn::FusedPDAttentionK4OpOptions options_;
};

class FusedPDAttentionK4Factory final
    : public mllm::plugin::interface::CustomizedOpFactory<nn::qnn::FusedPDAttentionK4OpOptions> {
 public:
  inline std::shared_ptr<mllm::BaseOp> createOpImpl(const nn::qnn::FusedPDAttentionK4OpOptions& cargo) override {
    auto p = std::make_shared<FusedPDAttentionK4Op>(cargo);
    p->setOpType(opType());
    return p;
  }
};

}  // namespace mllm::qnn
