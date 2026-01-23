// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/PDFusionPass.hpp"

#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

uint8_t PDFusionPass::run(const ir::node_ptr_t& op) {
  auto cfg = AOTCompileContext::getInstance().getConfig();
  if (!cfg.contains("pd_fusion") || !cfg["pd_fusion"].is_object()) { return ir::PASS_RET_SUCCESS; }
  const bool enable = cfg["pd_fusion"].value("enable", false);
  if (!enable) { return ir::PASS_RET_SUCCESS; }

  const int total_len = cfg["pd_fusion"].value("total_len", 128);
  const int prefill_len = cfg["pd_fusion"].value("prefill_len", 127);
  MLLM_INFO("PDFusionPass enabled (total_len={}, prefill_len={})", total_len, prefill_len);
  if (prefill_len <= 0 || prefill_len >= total_len) {
    MLLM_ERROR("PDFusionPass: invalid lengths (total_len={}, prefill_len={})", total_len, prefill_len);
    return ir::PASS_RET_FAILURE;
  }

  // The top op should be ModuleOp
  if (!op->isa_<ir::ModuleOp>()) {
    MLLM_ERROR("PDFusionPass expects ModuleOp as top level op");
    return ir::PASS_RET_FAILURE;
  }

  const std::string old_graph_name = "model.0.s" + std::to_string(total_len);
  const std::string new_graph_name = "model.0.pd.s" + std::to_string(total_len);

  auto ir_ctx = getCtx();
  auto old_graph = ir_ctx->lookupSymbolTable(old_graph_name);
  if (!old_graph) {
    MLLM_WARN("PDFusionPass: subgraph '{}' not found, skip renaming", old_graph_name);
    return ir::PASS_RET_SUCCESS;
  }

  // 1) Rename the subgraph symbol.
  if (old_graph->isa_<ir::graph::SubGraphOp>()) {
    auto sub_g = old_graph->cast_<ir::graph::SubGraphOp>();
    sub_g->setSymbolAttr(ir_ctx->create<ir::SymbolAttr>(new_graph_name));
  } else {
    MLLM_WARN("PDFusionPass: symbol '{}' is not a SubGraphOp, skip renaming", old_graph_name);
    return ir::PASS_RET_SUCCESS;
  }

  // 2) Update call sites: any CallGraphOp referencing model.0.s{N} -> model.0.pd.s{N}.
  auto module_op = op->cast_<ir::ModuleOp>();
  auto writer = ir::IRWriter(ir_ctx, module_op->getTopRegion());
  writer.walk<ir::graph::CallGraphOp>([&](ir::IRWriter& /*w*/, const ir::graph::CallGraphOp::ptr_t& call_op) {
    if (call_op->getSymbolAttr() && call_op->getSymbolAttr()->str() == old_graph_name) {
      call_op->setSymbolAttr(ir_ctx->create<ir::SymbolAttr>(new_graph_name));
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // 3) Best-effort update: ops that carry "qnn_graph_name" attr.
  writer.walk<ir::Op>([&](ir::IRWriter& /*w*/, const ir::Op::ptr_t& any_op) {
    auto attr = any_op->getAttr("qnn_graph_name");
    if (attr && attr->isa_<ir::StrAttr>() && attr->cast_<ir::StrAttr>()->data() == old_graph_name) {
      any_op->setAttr("qnn_graph_name", ir_ctx->create<ir::StrAttr>(new_graph_name));
    }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // NOTE: Actual PD fusion rewrites (dual-slot attention/kv-cache) are TODO and will be implemented
  // once the custom ops and model-specific IR patterns are defined.

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createPDFusionPass() { return std::make_shared<PDFusionPass>(); }

}  // namespace mllm::qnn::aot
