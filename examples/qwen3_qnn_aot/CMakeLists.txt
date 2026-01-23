# AOT targets run on x86
if(MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE)
  add_executable(mllm-qwen3-aot-c compile.cpp)
  target_link_libraries(mllm-qwen3-aot-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-pd-c pd_compile.cpp)
  target_link_libraries(mllm-qwen3-aot-pd-c PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-pd-c PRIVATE ${MLLM_INCLUDE_DIR})

  add_executable(mllm-qwen3-aot-build-pd-context build_context_pd.cpp)
  target_link_libraries(mllm-qwen3-aot-build-pd-context PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
  target_include_directories(mllm-qwen3-aot-build-pd-context PRIVATE ${MLLM_INCLUDE_DIR})
endif()

add_executable(mllm-qwen3-aot-runner aot_run.cpp)
target_link_libraries(mllm-qwen3-aot-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-runner PRIVATE ${MLLM_INCLUDE_DIR})

add_executable(mllm-qwen3-aot-pd-runner pd_run.cpp)
target_link_libraries(mllm-qwen3-aot-pd-runner PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-aot-pd-runner PRIVATE ${MLLM_INCLUDE_DIR})
