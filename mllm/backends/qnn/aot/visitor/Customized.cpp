// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/visitor/Customized.hpp"

#include "mllm/backends/base/PluginInterface.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

static bool isPDKVCacheUpdate(const mllm::ir::linalg::CustomizedOp::ptr_t& op) {
  auto* base = op->getAOp();
  auto* customized = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(base);
  if (!customized) { return false; }
  return customized->getCustomOpTypeName() == std::string("PDKVCacheUpdate");
}

static bool isFusedPDAttention(const mllm::ir::linalg::CustomizedOp::ptr_t& op) {
  auto* base = op->getAOp();
  auto* customized = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(base);
  if (!customized) { return false; }
  const auto t = customized->getCustomOpTypeName();
  return t == std::string("FusedPDAttention") || t == std::string("FusedPDAttentionNoMask");
}

static bool isFusedPDAttentionK4(const mllm::ir::linalg::CustomizedOp::ptr_t& op) {
  auto* base = op->getAOp();
  auto* customized = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(base);
  if (!customized) { return false; }
  const auto t = customized->getCustomOpTypeName();
  return t == std::string("FusedPDAttentionK4") || t == std::string("FusedPDAttentionK4NoMask");
}

bool QnnAOTPDKVCacheUpdatePattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (!op->isa_<mllm::ir::linalg::CustomizedOp>()) { return false; }
  if (op->getAttr("using_qnn") == nullptr) { return false; }
  return isPDKVCacheUpdate(op->cast_<mllm::ir::linalg::CustomizedOp>());
}

bool QnnAOTPDKVCacheUpdatePattern::rewrite(ir::IRWriter& /*writer*/, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto cust = op->cast_<mllm::ir::linalg::CustomizedOp>();
  MLLM_RETURN_FALSE_IF_NOT(isPDKVCacheUpdate(cust));

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto qnn_op_node = QnnAOTNodeOperation::create("PDKVCacheUpdate");
  qnn_op_node->setPackageName("LLaMAPackage")->setName(cust->getAOp()->getName());

  for (auto& in_v : op->inputs()) {
    auto t = in_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }
  for (auto& out_v : op->outputs()) {
    auto t = out_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

bool QnnAOTFusedPDAttentionPattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (!op->isa_<mllm::ir::linalg::CustomizedOp>()) { return false; }
  if (op->getAttr("using_qnn") == nullptr) { return false; }
  return isFusedPDAttention(op->cast_<mllm::ir::linalg::CustomizedOp>());
}

bool QnnAOTFusedPDAttentionPattern::rewrite(ir::IRWriter& /*writer*/, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto cust = op->cast_<mllm::ir::linalg::CustomizedOp>();
  MLLM_RETURN_FALSE_IF_NOT(isFusedPDAttention(cust));

  auto* base = cust->getAOp();
  auto* customized = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(base);
  MLLM_RETURN_FALSE_IF_NOT(customized);
  const std::string type_name = customized->getCustomOpTypeName();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto qnn_op_node = QnnAOTNodeOperation::create(type_name);
  qnn_op_node->setPackageName("LLaMAPackage")->setName(cust->getAOp()->getName());

  for (auto& in_v : op->inputs()) {
    auto t = in_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }
  for (auto& out_v : op->outputs()) {
    auto t = out_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

bool QnnAOTFusedPDAttentionK4Pattern::isMatch(const mllm::ir::op_ptr_t& op) {
  if (!op->isa_<mllm::ir::linalg::CustomizedOp>()) { return false; }
  if (op->getAttr("using_qnn") == nullptr) { return false; }
  return isFusedPDAttentionK4(op->cast_<mllm::ir::linalg::CustomizedOp>());
}

bool QnnAOTFusedPDAttentionK4Pattern::rewrite(ir::IRWriter& /*writer*/, const ir::op_ptr_t& op) {
  auto env = AOTCompileContext::getInstance().getEnv();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("quant_recipe"));
  auto cust = op->cast_<mllm::ir::linalg::CustomizedOp>();
  MLLM_RETURN_FALSE_IF_NOT(isFusedPDAttentionK4(cust));

  auto* base = cust->getAOp();
  auto* customized = dynamic_cast<mllm::plugin::interface::CustomizedOp*>(base);
  MLLM_RETURN_FALSE_IF_NOT(customized);
  const std::string type_name = customized->getCustomOpTypeName();

  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_graph_name"));
  auto qnn_graph_name = op->getAttr("qnn_graph_name")->cast_<ir::StrAttr>()->data();
  MLLM_RETURN_FALSE_IF_NOT(op->getAttr("qnn_context_name"));
  auto qnn_context_name = op->getAttr("qnn_context_name")->cast_<ir::StrAttr>()->data();

  auto qnn_op_node = QnnAOTNodeOperation::create(type_name);
  qnn_op_node->setPackageName("LLaMAPackage")->setName(cust->getAOp()->getName());

  for (auto& in_v : op->inputs()) {
    auto t = in_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceInput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }
  for (auto& out_v : op->outputs()) {
    auto t = out_v->cast_<ir::tensor::TensorValue>();
    qnn_op_node->emplaceOutput(env->captureQnnAOTNodeTensor(qnn_context_name, qnn_graph_name, t));
  }

  env->captureAOTNodeOp(qnn_context_name, qnn_graph_name, qnn_op_node);
  return true;
}

}  // namespace mllm::qnn::aot
