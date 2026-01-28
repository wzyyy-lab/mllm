# Runtime of AOT Models

## PD Fusion (Prefill-Decode Fusion)

This folder now contains a runtime-side scaffolding for PD fusion:

- `PDFusionRunner` (`PDFusionRunner.hpp/.cpp`): packs one prefill chunk and one decode token into a single fixed-shape
  invocation and updates two KV caches from a shared KV output buffer.
  It also supports a simple `swapSlots()` to promote a prefilling request into the decode slot in a 2-slot pipeline.
- `PDFusionScheduler` (`PDFusionScheduler.hpp/.cpp`): a minimal ActionFlow-style packer with `prefill_queue` + `decode_queue`.
  Each tick packs at most one prefill request and one decode request into one PD graph execution; when no prefill is
  available it falls back to the `model.0.s1` graph to avoid wasting PD rows.
- `QnnAOTConfig` (`QnnAOTConfig.hpp`): includes PD-related fixed-shape parameters (`pd_total_len`, `pd_prefill_len`).

Notes:

- `PDFusionRunner` expects a PD graph named `model.0.pd.s{pd_total_len}` and tensor names matching `PDFusionRunner.cpp`.
- `PDFusionRunner` populates `attention_mask` using the same convention as `KVCacheManager`: `allowed=65535`, `masked=0`.
- `fusion_ctrl` is an int32 tensor of shape `[6]` used for runtime control (enable flags + lengths).
- When there is no prefill work to fuse, use `PDFusionRunner::decodeOnly()` (graph `model.0.s1`) to avoid wasting
  `pd_total_len-1` rows of compute in the PD graph.

P0 optimizations (PD graphs):
- Prefill is packed right-aligned into `[base..N-2]` and decode is fixed at `N-1` to make "useful rows" stable.
- PD graphs only output the last 2 logits rows (`[N-2, N-1]`) to reduce `lm_head` compute and logits bandwidth.

### Demo (AOT compile side)

The example `mllm-main/examples/qwen3_qnn_aot/pd_compile.cpp` generates a PD-fusion graph (including
`position_ids`, `fusion_ctrl`, and dual KV inputs) and saves a context containing `model.0.pd.s{N}`.

To build a single QNN context binary containing both `model.0.pd.s{N}` and `model.0.s1`, use
`mllm-main/examples/qwen3_qnn_aot/build_context_pd.cpp`.

Both tools accept `--qnn_lib_dir` to point to the directory containing `libQnnHtp.so` (optional). If not provided,
`libQnnHtp.so` must be discoverable via the dynamic loader (e.g. `LD_LIBRARY_PATH` on Linux).

To reduce padding waste for smaller prefill chunks without writing custom HTP kernels, you can compile multiple PD graphs
into the same context (e.g. `model.0.pd.s32`, `model.0.pd.s64`, `model.0.pd.s128`) using
`mllm-main/examples/qwen3_qnn_aot/build_context_pd.cpp --total_lens 32,64,128`, and then call
`PDFusionRunner::load({32,64,128})` (or construct `PDFusionScheduler` with the same list) so runtime selects the smallest
`N` that fits the current chunk.

Limitations of the "no custom HTP kernel" PD fusion:
- KV cache update is still performed by `KVCacheManager::updateCache()` (CPU-side memcopies between shared buffers).
- Prefill and decode branches are not guaranteed to overlap on HTP; true overlap requires fused kernels / vendor-level scheduling.

### On-device KV update (Kernel B)

If the traced model includes `PDKVCacheUpdate` custom ops and runtime sets `QnnAOTConfig::kv_update_on_device=true`,
KV cache updates can be performed inside the QNN graph by `LLaMAPackage::PDKVCacheUpdate`.

Notes / constraints:
- Runtime binds `updated_past_*` graph outputs to the same persistent KV buffers as the corresponding `past_*` graph inputs
  (true in-place update). Some QNN builds may reject input/output aliasing; if that happens, `PDKVCacheUpdate` falls back to
  copying the full cache from `in_*` to `out_*` before patching the updated span (correct but bandwidth-heavy).
- `MLLM_QNN_OP_PACKAGE_DIR` can be used to point both AOT and runtime to `libQnnLLaMAPackage_{CPU,HTP}.so`.
