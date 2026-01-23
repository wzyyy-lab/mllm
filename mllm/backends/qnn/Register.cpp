// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include <filesystem>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/mllm.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "CustomLayers.hpp"

namespace mllm {

// export initQnnBackend function to initialize QNN backend
void initQnnBackend(const std::string& context_path) {
  MLLM_RT_ASSERT(isQnnAvailable());
  auto& ctx = Context::instance();

  // 1. Register backend
  auto backend = std::make_shared<qnn::QNNBackend>();
  if (std::filesystem::exists(context_path)) {
    if (!backend->loadContext(context_path)) { MLLM_ERROR_EXIT(1, "Failed to load QNN context from {}", context_path); }
  } else {
    if (!backend->createContext()) { MLLM_ERROR_EXIT(1, "Failed to create QNN context"); }
  }
  ctx.registerBackend(backend);

  // 2. Initialize memory manager
  ctx.memoryManager()->registerAllocator(kQNN, backend->allocator(),
                                         {
                                             .really_large_tensor_threshold = 0,
                                             .using_buddy_mem_pool = false,
                                         });
  // 3. Initialize dispatcher manager
  ctx.dispatcherManager()->registerDispatcher(
      createQNNDispatcher(ctx.dispatcherManager()->getExecutor(), qnn::QNNDispatcherOptions()));

  // register QNN custom ops
  Context::instance().registerCustomizedOp(kQNN, "DequantizeAdd",
                                           std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::DequantizeAddFactory())));
  Context::instance().registerCustomizedOp(
      kQNN, "PDKVCacheUpdate",
      std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::PDKVCacheUpdateFactory())));
  Context::instance().registerCustomizedOp(
      kQNN, "FusedPDAttention",
      std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::FusedPDAttentionFactory())));
  // Also register under CPU so tracing/AOT compilation can create the op without requiring a QNN backend.
  Context::instance().registerCustomizedOp(
      kCPU, "PDKVCacheUpdate",
      std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::PDKVCacheUpdateFactory())));
  Context::instance().registerCustomizedOp(
      kCPU, "FusedPDAttention",
      std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::FusedPDAttentionFactory())));
}
}  // namespace mllm
