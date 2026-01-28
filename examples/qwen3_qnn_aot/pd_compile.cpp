// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <unordered_map>
#include <cstdlib>

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/engine/Context.hpp>
#include <mllm/backends/qnn/CustomLayers.hpp>

#include "modeling_qwen_qnn_aot.hpp"

using mllm::Argparse;

// This example compiles a PD-fusion QNN AOT graph:
// - fixed sequence length = total_len (default 128)
// - adds an extra input: fusion_ctrl [6] int32
// - adds two KV cache input sets: prefill + decode (naming matches PDFusionRunner)
// - attention is split into a prefill branch (rows [0..N-2]) and a decode branch (row [N-1]) with separate KV inputs.
MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& total_len = Argparse::add<int>("--total_len").help("PD total length (prefill+decode)").def(128);
  auto& context_len = Argparse::add<int>("--context_len").help("Context length").def(1024);
  auto& qnn_lib_dir =
      Argparse::add<std::string>("--qnn_lib_dir").help("Directory containing libQnnHtp.so (optional).").def("");
  auto& op_package_dir = Argparse::add<std::string>("--op_package_dir")
                             .help("Directory containing libQnnLLaMAPackage_{CPU,HTP}.so (optional).")
                             .def("");
  auto& out_context = Argparse::add<std::string>("--out_context").help("Output QNN context path").def("qwen3_pd_context.bin");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  const int N = total_len.get();
  const int CL = context_len.get();
  if (N <= 1 || CL <= N) { MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid lengths: total_len={}, context_len={}", N, CL); }

  if (!op_package_dir.get().empty()) {
#ifdef _WIN32
    _putenv_s("MLLM_QNN_OP_PACKAGE_DIR", op_package_dir.get().c_str());
#else
    setenv("MLLM_QNN_OP_PACKAGE_DIR", op_package_dir.get().c_str(), 1);
#endif
  }

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  // Register the PD KV-cache update custom op on CPU so it can be traced during AOT compilation.
  // The AOT lowering pass will emit it as a QNN custom op (LLaMAPackage::PDKVCacheUpdate).
  mllm::Context::instance().registerCustomizedOp(
      mllm::kCPU, "PDKVCacheUpdate",
      std::shared_ptr<mllm::BaseOpFactory>((mllm::BaseOpFactory*)(new mllm::qnn::PDKVCacheUpdateFactory())));
  // Optional (Kernel C): fused PD attention custom op (currently disabled by default in the model).
  mllm::Context::instance().registerCustomizedOp(
      mllm::kCPU, "FusedPDAttention",
      std::shared_ptr<mllm::BaseOpFactory>((mllm::BaseOpFactory*)(new mllm::qnn::FusedPDAttentionFactory())));

  auto model = mllm::models::qwen3::Qwen3ForCausalLM(model_cfg);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  model.load(params);

  // Graph input naming aligns with PDFusionRunner:
  // input_ids: [1, N], position_ids: [N], attention_mask: [1,1,N,CL], fusion_ctrl: [6]
  auto input_ids = mllm::Tensor::zeros({1, N}, mllm::kInt32).setName("input_ids");
  auto position_ids = mllm::Tensor::zeros({N}, mllm::kInt32).setName("position_ids");
  auto attention_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16).setName("attention_mask");
  // fusion_ctrl: [6]
  //   [0]=is_prefill_active, [1]=is_decode_active, [2]=prefill_n_update, [3]=reserved,
  //   [4]=prefill_n_past, [5]=decode_n_past
  auto fusion_ctrl = mllm::Tensor::zeros({6}, mllm::kInt32).setName("fusion_ctrl");

  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["input_ids"] = input_ids;
  trace_inputs["position_ids"] = position_ids;
  trace_inputs["attention_mask"] = attention_mask;
  trace_inputs["fusion_ctrl"] = fusion_ctrl;

  // KV cache shapes (KV heads):
  //   K cache uses token-major layout: [B, H_kv, CL-N, D] (D contiguous per token)
  //   V cache is already token-major:  [B, H_kv, CL-N, D]
  const int H_kv = model_cfg.num_key_value_heads;
  const int D = model_cfg.head_dim;
  const int past_len = CL - N;

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    const auto k_scale = params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale");
    const auto k_zp =
        params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point");
    const auto v_scale = params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale");
    const auto v_zp =
        params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point");

    const auto k0_name = "past_key_prefill_" + std::to_string(i);
    const auto v0_name = "past_value_prefill_" + std::to_string(i);
    const auto k1_name = "past_key_decode_" + std::to_string(i);
    const auto v1_name = "past_value_decode_" + std::to_string(i);

    auto k0 = mllm::Tensor::empty({1, H_kv, past_len, D}, mllm::kUInt8PerTensorSym).setName(k0_name);
    auto v0 = mllm::Tensor::empty({1, H_kv, past_len, D}, mllm::kUInt8PerTensorSym).setName(v0_name);
    auto k1 = mllm::Tensor::empty({1, H_kv, past_len, D}, mllm::kUInt8PerTensorSym).setName(k1_name);
    auto v1 = mllm::Tensor::empty({1, H_kv, past_len, D}, mllm::kUInt8PerTensorSym).setName(v1_name);

    k0.attach("scale", k_scale.impl(), true);
    k0.attach("zero_point", k_zp.impl(), true);
    v0.attach("scale", v_scale.impl(), true);
    v0.attach("zero_point", v_zp.impl(), true);

    k1.attach("scale", k_scale.impl(), true);
    k1.attach("zero_point", k_zp.impl(), true);
    v1.attach("scale", v_scale.impl(), true);
    v1.attach("zero_point", v_zp.impl(), true);

    trace_inputs[k0_name] = k0;
    trace_inputs[v0_name] = v0;
    trace_inputs[k1_name] = k1;
    trace_inputs[v1_name] = v1;
  }

  auto ir = model.trace(trace_inputs, {});

  const auto tm = mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get());
  auto qnn_aot_env =
      qnn_lib_dir.get().empty() ? mllm::qnn::aot::QnnAOTEnv(tm) : mllm::qnn::aot::QnnAOTEnv(qnn_lib_dir.get(), tm);

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
  pm.run();

  qnn_aot_env.saveContext("context.0", out_context.get());
  mllm::redirect("qwen3_qnn_aot_pd.mir", [&]() { mllm::print(ir["model"]); });
});
