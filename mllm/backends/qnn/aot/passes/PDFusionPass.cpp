// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/PDFusionPass.hpp"

#include <cctype>
#include <string>
#include <unordered_map>

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

  auto ir_ctx = getCtx();
  auto module_op = op->cast_<ir::ModuleOp>();

  // In a "multi-bucket" build (pd.s32/s64/s128...), build_context_pd.cpp runs the pipeline multiple times with the same
  // aot_config. The old implementation only renamed exactly one graph "model.0.s{total_len}" -> "model.0.pd.s{total_len}",
  // leaving other buckets as "model.0.s{N}" and making runtime unable to load them.
  //
  // Fix: rename all subgraphs matching "model.0.s{N}" (N>1) to "model.0.pd.s{N}", and update all call sites/attrs.

  std::unordered_map<std::string, std::string> rename_map;
  auto parse_bucket_len = [](const std::string& name) -> int {
    const std::string prefix = "model.0.s";
    if (name.rfind(prefix, 0) != 0) return -1;
    if (name.size() == prefix.size()) return -1;
    for (size_t i = prefix.size(); i < name.size(); ++i) {
      if (!std::isdigit((unsigned char)name[i])) return -1;
    }
    return std::stoi(name.substr(prefix.size()));
  };

  {
    auto writer = ir::IRWriter(ir_ctx, module_op->getTopRegion());
    writer.walk<ir::graph::SubGraphOp>([&](ir::IRWriter& /*w*/, const ir::graph::SubGraphOp::ptr_t& sub_g) {
      const std::string old_name = sub_g->getSymbolAttr()->str();
      const int N = parse_bucket_len(old_name);
      if (N <= 1) return ir::IRWriter::WalkResult::WALK_CONTINUE;
      const std::string new_name = "model.0.pd.s" + std::to_string(N);
      rename_map.emplace(old_name, new_name);
      sub_g->setSymbolAttr(ir_ctx->create<ir::SymbolAttr>(new_name));
      return ir::IRWriter::WalkResult::WALK_CONTINUE;
    });
  }

  if (rename_map.empty()) {
    MLLM_WARN("PDFusionPass: no matching subgraphs found for renaming (expected names like model.0.s32/model.0.s128)");
    return ir::PASS_RET_SUCCESS;
  }

  auto writer = ir::IRWriter(ir_ctx, module_op->getTopRegion());
  // 1) Update call sites.
  writer.walk<ir::graph::CallGraphOp>([&](ir::IRWriter& /*w*/, const ir::graph::CallGraphOp::ptr_t& call_op) {
    if (!call_op->getSymbolAttr()) return ir::IRWriter::WalkResult::WALK_CONTINUE;
    const std::string sym = call_op->getSymbolAttr()->str();
    auto it = rename_map.find(sym);
    if (it != rename_map.end()) { call_op->setSymbolAttr(ir_ctx->create<ir::SymbolAttr>(it->second)); }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // 2) Best-effort update: ops that carry "qnn_graph_name" attr.
  writer.walk<ir::Op>([&](ir::IRWriter& /*w*/, const ir::Op::ptr_t& any_op) {
    auto attr = any_op->getAttr("qnn_graph_name");
    if (!attr || !attr->isa_<ir::StrAttr>()) return ir::IRWriter::WalkResult::WALK_CONTINUE;
    const std::string gname = attr->cast_<ir::StrAttr>()->data();
    auto it = rename_map.find(gname);
    if (it != rename_map.end()) { any_op->setAttr("qnn_graph_name", ir_ctx->create<ir::StrAttr>(it->second)); }
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
  });

  // NOTE: Actual PD fusion rewrites (dual-slot attention/kv-cache) are TODO and will be implemented
  // once the custom ops and model-specific IR patterns are defined.

  return ir::PASS_RET_SUCCESS;
}

ir::Pass::ptr_t createPDFusionPass() { return std::make_shared<PDFusionPass>(); }

}  // namespace mllm::qnn::aot
