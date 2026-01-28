// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class QnnAOTPDKVCacheUpdatePattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTPDKVCacheUpdatePattern>> create() {
    return {OpTypes::kDynamicOp_Start, std::make_shared<QnnAOTPDKVCacheUpdatePattern>()};
  }
};

class QnnAOTFusedPDAttentionPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTFusedPDAttentionPattern>> create() {
    return {OpTypes::kDynamicOp_Start, std::make_shared<QnnAOTFusedPDAttentionPattern>()};
  }
};

class QnnAOTFusedPDAttentionK4Pattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;
  bool rewrite(ir::IRWriter& writer, const ir::op_ptr_t& op) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTFusedPDAttentionK4Pattern>> create() {
    return {OpTypes::kDynamicOp_Start, std::make_shared<QnnAOTFusedPDAttentionK4Pattern>()};
  }
};

}  // namespace mllm::qnn::aot
