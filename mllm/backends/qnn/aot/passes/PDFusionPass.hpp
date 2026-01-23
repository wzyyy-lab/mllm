// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn::aot {

// PDFusionPass is a placeholder pass for Prefill-Decode Fusion (PD Fusion).
//
// The intended goal is to:
// - Enforce/align a fixed PD static shape (e.g. total_len=128 with prefill_len=127+decode=1).
// - Inject/propagate a control input (fusion_ctrl) to custom HTP kernels.
// - Rewrite attention/kv-cache related subgraphs to PD-fused custom ops.
//
// The actual fusion rewrite depends on target model IR patterns and custom-op availability.
class PDFusionPass final : public ir::Pass {
 public:
  PDFusionPass() = default;
  ~PDFusionPass() override = default;

  uint8_t run(const ir::node_ptr_t& op) override;
};

ir::Pass::ptr_t createPDFusionPass();

}  // namespace mllm::qnn::aot

