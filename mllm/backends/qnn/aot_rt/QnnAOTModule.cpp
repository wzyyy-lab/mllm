#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include <algorithm>

namespace mllm::qnn::aot {

QnnAOTModule::QnnAOTModule(const std::string& graph_name) : mllm::nn::Module(graph_name), graph_name_(graph_name) {}

std::vector<mllm::Tensor> QnnAOTModule::forward(const std::vector<mllm::Tensor>& inputs,
                                                const std::vector<mllm::AnyValue>& args) {
  (void)args;
  auto backend_base = Context::instance().getBackend(kQNN);
  if (!backend_base) {
    MLLM_ERROR("QnnAOTModule: QNN backend not initialized (graph='{}')", graph_name_);
    return {};
  }
  auto backend = std::dynamic_pointer_cast<mllm::qnn::QNNBackend>(backend_base);
  if (!backend) {
    MLLM_ERROR("QnnAOTModule: backend type is not QNNBackend (graph='{}')", graph_name_);
    return {};
  }

  if (output_tensors_.empty()) {
    MLLM_ERROR("QnnAOTModule: output_tensors_ not set for graph='{}' (call setOutputTensors first)", graph_name_);
    return {};
  }

  // Make mutable copies for backend execution API.
  auto in = inputs;
  auto out = output_tensors_;

  Qnn_ErrorHandle_t err = QNN_SUCCESS;
  if (!backend->graphExecuteChecked(graph_name_, in, out, &err)) {
    MLLM_ERROR("QnnAOTModule: graphExecuteChecked failed (graph='{}', err=0x{:x})", graph_name_, (uint32_t)err);
    return {};
  }

  output_tensors_ = out;
  return output_tensors_;
}

int64_t QnnAOTModule::sampleGreedy(mllm::Tensor& logits) {
  auto logits_data = logits.ptr<uint16_t>();
  int vocab_size = logits.shape().back();
  auto max_it = std::max_element(logits_data, logits_data + vocab_size);
  return std::distance(logits_data, max_it);
}

}  // namespace mllm::qnn::aot
