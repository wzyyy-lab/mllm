// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "CustomLayers.hpp"
#include <memory>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

// -------------------- Custom QNN Layers --------------------
namespace mllm::nn::qnn {
DequantizeAdd::DequantizeAdd()
    : Layer(OpTypes::kDynamicOp_Start, DequantizeAddOpOptions{.dtype = kFloat32, .out_channels = 0}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"));
  this->impl()->__forceSetDevice(kQNN);
}

DequantizeAdd::DequantizeAdd(DataTypes dtype, int32_t out_channels)
    : Layer(OpTypes::kDynamicOp_Start, DequantizeAddOpOptions{.dtype = dtype, .out_channels = out_channels}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"));
  this->impl()->__forceSetDevice(kQNN);
}

PDKVCacheUpdate::PDKVCacheUpdate() : Layer(OpTypes::kDynamicOp_Start, PDKVCacheUpdateOpOptions{}) {
  // NOTE: For AOT compilation/tracing we do not require a live QNN backend in Context.
  // Register this customized op under the CPU backend (see examples) so it can be traced, then
  // AOT lowering will emit it as a QNN custom op (LLaMAPackage) via QnnAOTPDKVCacheUpdatePattern.
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "PDKVCacheUpdate"));
  this->impl()->__forceSetDevice(kCPU);
}

FusedPDAttention::FusedPDAttention() : Layer(OpTypes::kDynamicOp_Start, FusedPDAttentionOpOptions{}) {
  // Same tracing trick as PDKVCacheUpdate: create the op under CPU so AOT can trace without a live QNN backend.
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "FusedPDAttention"));
  this->impl()->__forceSetDevice(kCPU);
}

FusedPDAttentionK4::FusedPDAttentionK4() : Layer(OpTypes::kDynamicOp_Start, FusedPDAttentionK4OpOptions{}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "FusedPDAttentionK4"));
  this->impl()->__forceSetDevice(kCPU);
}

FusedPDAttentionNoMask::FusedPDAttentionNoMask() : Layer(OpTypes::kDynamicOp_Start, FusedPDAttentionNoMaskOpOptions{}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "FusedPDAttentionNoMask"));
  this->impl()->__forceSetDevice(kCPU);
}

FusedPDAttentionK4NoMask::FusedPDAttentionK4NoMask() : Layer(OpTypes::kDynamicOp_Start, FusedPDAttentionK4NoMaskOpOptions{}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kCPU, "FusedPDAttentionK4NoMask"));
  this->impl()->__forceSetDevice(kCPU);
}

}  // namespace mllm::nn::qnn

// -------------------- Custom QNN Ops --------------------
namespace mllm::qnn {
void DequantizeAddOp::load(const mllm::ParameterFile::ptr_t& ploader) {
  std::string weight_name = getName();
  // find the ".dequantize" suffix and replace it with ".bias"
  auto pos = weight_name.find("dequantize");
  if (pos != -1) { weight_name.erase(pos, 10); }
  weight_name += "bias";

  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(weight_name);
      weight_ = weight_.view({1, 1, 1, options_.out_channels});
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(weight_name);
      weight_ = weight_.view({1, 1, 1, options_.out_channels});
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void DequantizeAddOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}
void DequantizeAddOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // CastType operation maintains the same shape
  assert(inputs.size() == 1);
  const auto& input = inputs[0];

  outputs.emplace_back(Tensor::empty(input.shape(), options_.dtype, input.device()));
}

void PDKVCacheUpdateOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}

void PDKVCacheUpdateOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // Inputs:
  //   dep, in_k_cache, in_v_cache, present_k, present_v, n_past, src_offset, n_update, enable
  MLLM_RT_ASSERT_EQ((int)inputs.size(), 9);
  outputs.emplace_back(Tensor::empty(inputs[1].shape(), inputs[1].dtype(), inputs[1].device()));
  outputs.emplace_back(Tensor::empty(inputs[2].shape(), inputs[2].dtype(), inputs[2].device()));
}

void FusedPDAttentionOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}

void FusedPDAttentionOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // Inputs (v2):
  //   q, k_curr, v_curr, past_k_prefill, past_v_prefill, past_k_decode, past_v_decode, attention_mask, fusion_ctrl,
  //   ref_max_seq_override, ref_max_past_override,
  //   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
  MLLM_RT_ASSERT_EQ((int)inputs.size(), 19);
  const auto& q = inputs[0];
  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
}

void FusedPDAttentionK4Op::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}

void FusedPDAttentionK4Op::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // Inputs (v3, k=4):
  //   q, k_curr, v_curr,
  //   past_k_prefill, past_v_prefill,
  //   past_k_dec0, past_v_dec0, past_k_dec1, past_v_dec1, past_k_dec2, past_v_dec2, past_k_dec3, past_v_dec3,
  //   attention_mask, fusion_ctrl, decode_past_lens, ref_max_seq_override, ref_max_past_override,
  //   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
  MLLM_RT_ASSERT_EQ((int)inputs.size(), 26);
  const auto& q = inputs[0];
  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
}

void FusedPDAttentionNoMaskOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}

void FusedPDAttentionNoMaskOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // Inputs (v2, no-mask):
  //   q, k_curr, v_curr, past_k_prefill, past_v_prefill, past_k_decode, past_v_decode, fusion_ctrl,
  //   ref_max_seq_override, ref_max_past_override,
  //   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
  MLLM_RT_ASSERT_EQ((int)inputs.size(), 18);
  const auto& q = inputs[0];
  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
}

void FusedPDAttentionK4NoMaskOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}

void FusedPDAttentionK4NoMaskOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // Inputs (v3, k=4, no-mask):
  //   q, k_curr, v_curr,
  //   past_k_prefill, past_v_prefill,
  //   past_k_dec0, past_v_dec0, past_k_dec1, past_v_dec1, past_k_dec2, past_v_dec2, past_k_dec3, past_v_dec3,
  //   fusion_ctrl, decode_past_lens, ref_max_seq_override, ref_max_past_override,
  //   q_scale, q_zp, k_scale, k_zp, v_scale, v_zp, out_scale, out_zp
  MLLM_RT_ASSERT_EQ((int)inputs.size(), 25);
  const auto& q = inputs[0];
  outputs.emplace_back(Tensor::empty(q.shape(), q.dtype(), q.device()));
}

}  // namespace mllm::qnn
