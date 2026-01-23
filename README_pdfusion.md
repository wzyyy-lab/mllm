# PD Fusion (Prefill–Decode Fusion) on Qualcomm NPU (QNN/HTP) — mllm-based prototype

本文件是对 `mllm-main/` 中 **PD Fusion（Prefill–Decode Fusion）** 相关改动的项目级说明（补充 `mllm-main/README.md` 上游文档）。

目标：在 **Snapdragon NPU（QNN/HTP）静态图** 约束下，把“当前请求的 prefill chunk”与“上一轮/另一请求的 decode(1 token)”拼到同一次 NPU 执行里，减少 decode 气泡并提升吞吐。

仓库根目录还包含 `../ActionFlow-1D47/`，用于参考 ActionFlow 的调度/Profiling 思路。

快速入口（建议按这个顺序读）：

- Runtime packer：`mllm-main/mllm/backends/qnn/aot_rt/PDFusionRunner.hpp` / `mllm-main/mllm/backends/qnn/aot_rt/PDFusionScheduler.hpp`
- Kernel B（设备侧 KV 更新）：`mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/PDKVCacheUpdate.cpp`
- Kernel C（PD Attention，自定义 op）：`mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/FusedPDAttention.cpp`
- Qwen3 示例（编译/打包/运行）：`mllm-main/examples/qwen3_qnn_aot/`

---

## 现状概览（Status）

- ✅ **PD runtime packer**：`prefill_queue/decode_queue` + 选择最小 bucket + 无 prefill 时回退 `s1`。
- ✅ **Kernel B：PDKVCacheUpdate（on-device KV update）**：KV 更新留在 NPU/共享缓冲区侧，尽量避免 CPU memcpy。
- ✅ **Kernel C（FusedPDAttention）接线完成 + correctness-first 参考实现**：支持显式 `scale/zp` 输入做量化 de/quant；默认关闭（性能不优化，仅验证语义/接口）。
- ⏳ **真正有性能意义的 Kernel C**（HVX/HMX + VTCM/SRAM 分块 + quantized attention path）：待设备侧优化与 profiling 驱动迭代。

---

## 现在已经完成了什么（What’s implemented）

> 目标是先把 “PD 融合的工程路径 + 数据流 + 图/Runtime/OpPackage 打通”，再做性能内核。

### A) 图侧：PD bucket + decode-only 图，并打到同一个 context

- ✅ 支持编译 PD 图：`model.0.pd.s{N}`（如 `N=32/64/128`）与 decode-only 图：`model.0.s1`。
- ✅ 支持把多个 PD bucket 与 `s1` 打进同一个 QNN context binary，运行时按当前 `prefill_n_update` 选择最小的 `N`（降低 padding 浪费）。

相关入口：

- `mllm-main/examples/qwen3_qnn_aot/pd_compile.cpp`
- `mllm-main/examples/qwen3_qnn_aot/build_context_pd.cpp`
- `mllm-main/examples/qwen3_qnn_aot/qnn_aot_cfg_pd.json`

### B) Runtime：ActionFlow 风格 packer + 两队列调度

- ✅ `PDFusionScheduler`：维护 `prefill_queue` + `decode_queue`，每 tick 尝试拼一个 prefill + 一个 decode；无 prefill 时回退 `model.0.s1`。
- ✅ `PDFusionRunner`：负责 PD 图 I/O 绑定与数据打包（`input_ids/position_ids/attention_mask/fusion_ctrl`），以及双路 KV buffer 绑定（prefill slot / decode slot）。
- ✅ `fusion_ctrl` 从 `[4]` 扩展为 `[6]`，显式携带长度信息，避免 Kernel C 需要额外 mask 带宽：
  - `[0]=is_prefill_active, [1]=is_decode_active`
  - `[2]=prefill_n_update`
  - `[4]=prefill_n_past, [5]=decode_n_past`

相关文件：

- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionRunner.hpp`
- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionRunner.cpp`
- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionScheduler.hpp`
- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionScheduler.cpp`
- `mllm-main/mllm/backends/qnn/aot_rt/README.md`

### C) Kernel B：PDKVCacheUpdate（设备侧 KV 更新）

- ✅ 在 OpPackage 中新增 `LLaMAPackage::PDKVCacheUpdate`：把本 step 的 `present_k/v` patch 写入 `past_k/v` 对应的缓存区，尽量避免 CPU memcpy。
- ✅ Runtime 支持 “true alias”：把图的 `updated_past_*` 输出绑定到与 `past_*` 输入相同的持久 KV buffer（ION/rpcmem）。
- ✅ 处理 SSA / in-place 陷阱：如果 QNN 不允许 alias 导致 `out_ptr != in_ptr`，Kernel B 会先全量 `memcpy(in→out)` 再 patch span，保证正确性（但会更吃带宽）。

相关文件：

- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/PDKVCacheUpdate.cpp`
- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml`
- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/LLaMAPackageInterface.cpp`
- `mllm-main/mllm/backends/qnn/aot/visitor/Customized.cpp`

### D) Kernel C：FusedPDAttention（先打通接口与语义）

- ✅ 在 OpPackage 中新增 `LLaMAPackage::FusedPDAttention`，并完成 AOT lowering/注册，使得模型侧可以把 attention 子图整体替换成一个 custom op。
- ✅ 当前提供的是 correctness-first 的 reference kernel（不做性能优化），默认关闭，仅用于验证：
  - 通过 `fusion_ctrl[4]/[5]`（past_len）重建 decode 可见范围，避免传大 mask。
  - 显式输入 `q/k/v/out` 的 `scale/zp`（8 个标量）来避免 QNN 隐式插入 dequant/cast。
  - 输出类型支持 `UFIXED_POINT_16`（对应 `UInt16`）。
- ✅ 通过编译宏开关启用：`MLLM_ENABLE_FUSED_PD_ATTENTION=1`（默认 0）。

相关文件：

- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/FusedPDAttention.cpp`
- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml`
- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/LLaMAPackageInterface.cpp`
- `mllm-main/mllm/backends/qnn/CustomLayers.hpp`
- `mllm-main/mllm/backends/qnn/CustomLayers.cpp`
- `mllm-main/mllm/backends/qnn/aot/visitor/Customized.cpp`
- `mllm-main/examples/qwen3_qnn_aot/modeling_qwen_qnn_aot.hpp`

---

## 现在还差什么（What’s missing / performance-critical gaps）

### 1) 真正的性能内核（Kernel C 的 HTP 优化版）

当前的 `FusedPDAttention` 只是“接口/语义打通”的 reference，实现目标是：模型能跑通、数据流正确、AOT 能序列化、runtime 能绑定与调度。

要获得 PD Fusion 的性能红利，Kernel C 必须升级为 HTP 友好的实现：

- quantized attention path（尽量避免 float 全量计算）
- KV streaming / tiling（按块加载 K/V，做在线 softmax，减少 DDR 带宽）
- VTCM(8MB)/SRAM 管理（prefetch + double buffer）
- 两路 KV 读取：prefill slot 读自己的 KV（可能仅当前块），decode slot 读历史 KV（4k）
- 并发/流水：把 decode 的 KV 读取与 prefill 的 compute 尽可能重叠（这部分依赖具体 HTP/driver 能力）

### 2) “权重物理共享”是否真的发生（需要用工具验证）

把多张图打进同一个 context binary ≠ 权重一定只存一份。是否共享取决于 QNN builder 对常量 tensor 的复用策略。

需要在设备侧用 QNN 工具/统计（或查看 context 体积）确认：

- context binary 是否接近 1× model size（共享）还是接近 2×（未共享）

如果没共享，后续要考虑：在 graph build 阶段复用同一常量 tensor / shared buffer（可能需要更深的 QNN API 控制）。

### 3) “跨图一致性”：所有图的 KV 绑定必须一致

当 scheduler 在 `pd.sN` 与 `s1` 之间切换时，**所有图**（包括 `model.0.s1`）的 `past_k/v` 输入必须绑定到同一个全局持久 KV buffer（ION/rpcmem）。否则会出现 KV 断层。

### 4) 采样与状态机：PD 输出要同时处理两个 slot

PD 图输出 logits/hidden 是 `[1, N, ...]`：

- slot0（prefill）的“最后有效 token”需要采样，推进该请求进入 decode 队列
- slot1（decode）的 token 也需要采样，推进其下一步

这需要 sampler/token generator 能同时处理两个 index（而不是只取最后一个 logit）。

---

## 之后怎么做（Suggested next steps）

按“先验证正确性，再做性能”的节奏推荐：

1) 设备侧验证 Kernel B 的 alias 行为：尽量让 `out_ptr == in_ptr` 成立；如果长期走 memcpy 兜底，带宽会被吃掉。
2) 把 `model.0.s1` 的 KV 绑定路径和 PD bucket 完全统一（同一 ION/rpcmem handle），确保跨图切换不丢 KV。
3) 在 Android 设备上跑最小端到端 demo（`pd_run.cpp`），先验证：
   - prefill/decode 两队列状态机正确
   - KV 长度（`fusion_ctrl[4]/[5]`）正确推进到 4k
   - 同时采样两个 slot 的逻辑正确
4) 上性能：把 `FusedPDAttention` 从 reference 替换为 HTP 优化版（优先解决 4k KV streaming + VTCM 分块）。
5) 可选：再做 Kernel A（BranchingLinear）进一步消除 padding 空转；或继续扩展到多请求并发的更复杂 packing。

## 核心思路（Why PD Fusion）

传统端侧推理中 Prefill 与 Decode 串行切换，Decode 阶段常常带宽敏感、算力利用率低。PD Fusion 的思路是：

- 用固定形状 `N` 的静态图承载两条逻辑路径：
  - Slot0：`[0..N-2]` 做 prefill（chunk）。
  - Slot1：`[N-1]` 做 decode（1 token）。
- Runtime 每个 step 从 `prefill_queue` 和 `decode_queue` 各取一个任务拼成一次 `graphExecute`。
- 当没有 prefill 可拼时，不空跑大图，回退到 `model.0.s1`（1-token）图。
- KV cache 更新尽量在设备侧完成（避免 CPU↔NPU 数据乒乓）。

---

## 关键组件与文件说明（What’s where）

### 0) 仓库结构（Repository layout）

> 本文件在 `mllm-main/` 下，但 PD Fusion 相关改动主要集中在 QNN backend + examples + custom-op-package。

- `mllm-main/examples/qwen3_qnn_aot/`：PD 图编译/打包/运行示例（`pd_compile.cpp` / `build_context_pd.cpp` / `pd_run.cpp`）。
- `mllm-main/mllm/backends/qnn/aot_rt/`：PD runtime packer + bucket 选择 + KV buffer 绑定。
- `mllm-main/mllm/backends/qnn/aot/`：AOT passes 与 custom op lowering（`Customized.*` / `LLM2QnnLoweringPass.cpp` / `LLMQuantRecipePass.cpp`）。
- `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/`：HTP/CPU external op package（Kernel B/C 的实现与注册）。

### 1) Runtime：PD packer + 两套 KVCache

- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionRunner.hpp/.cpp`
  - 负责 PD 图 I/O 打包：`input_ids/position_ids/attention_mask/fusion_ctrl` 与双路 KV cache 绑定。
  - 支持多 bucket：`model.0.pd.s32/s64/s128...`，运行时选择最小满足 `prefill_n_update` 的 `N`。
  - 无 prefill 时回退执行 `model.0.s1`（decode-only）。
  - `fusion_ctrl`（shape `[6]`）由 runtime 填充（见下文）。
  - `kv_update_on_device=true` 时，会把 `updated_past_*` 输出绑定到持久 KV buffer（true alias）。

- `mllm-main/mllm/backends/qnn/aot_rt/PDFusionScheduler.hpp/.cpp`
  - 最小 ActionFlow 风格调度器：`prefill_queue` + `decode_queue`，每 tick 拼一次 PD 执行。

- `mllm-main/mllm/backends/qnn/aot_rt/KVCacheManager.hpp/.cpp`
  - 管理 KV cache 的 ION/rpcmem buffer 与 layout（`rearrangeCache()`）。

- Demo：
  - `mllm-main/examples/qwen3_qnn_aot/pd_run.cpp`

### 2) AOT 编译入口（Qwen3 示例）

- `mllm-main/examples/qwen3_qnn_aot/pd_compile.cpp`
  - 编译单个 `model.0.pd.s{N}` PD 图（Qwen3 示例）。

- `mllm-main/examples/qwen3_qnn_aot/build_context_pd.cpp`
  - 构建包含多个 PD bucket + `model.0.s1` 的单一 context binary。

- `mllm-main/examples/qwen3_qnn_aot/qnn_aot_cfg_pd.json`
  - PD fusion 的 AOT 配置（`pd_fusion.enable/total_len/prefill_len` 等）。

### 3) Custom Op（Kernel B / C）与 AOT lowering

#### Kernel B：`PDKVCacheUpdate`

目的：把 KV 更新留在设备侧，减少 CPU memcpy。

- HTP kernel：
  - `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/PDKVCacheUpdate.cpp`
  - 语义：把 `present_k/v` 的一段（由 `n_past/src_offset/n_update/enable` 控制）写入 `out_k_cache/out_v_cache`。
  - 重要兜底：如果 QNN 不允许 input/output alias（`out_ptr != in_ptr`），会先全量 `memcpy(in→out)` 再 patch span，保证正确性但带宽更大。

#### Kernel C：`FusedPDAttention`（correctness-first reference）

目的：把 PD attention 子图整体替换为一个自定义 op，为后续“真正 fused attention”打基础。

- HTP kernel（参考实现，慢）：
  - `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/FusedPDAttention.cpp`
  - 特点：
    - 不依赖全量 `attention_mask` 带宽：使用 `fusion_ctrl[4]/[5]` 的 `n_past` + 行号重建可见范围。
    - 显式接收 `q/k/v/out` 的 `scale/zp` 作为标量输入，避免 QNN 隐式插入 Dequant/Cast。
    - 输出类型支持 `UFIXED_POINT_16`（对应 `UInt16`）。
  - 默认不启用（见“启用方式”）。

- OpPackage 注册与 XML：
  - `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/config/LLaMAOpPackageHtp.xml`
  - `mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/src/LLaMAPackageInterface.cpp`

- AOT lowering（把 `CustomizedOp` 映射到 QNN custom op）：
  - `mllm-main/mllm/backends/qnn/aot/visitor/Customized.cpp`
  - `mllm-main/mllm/backends/qnn/aot/passes/LLM2QnnLoweringPass.cpp`
  - `mllm-main/mllm/backends/qnn/aot/passes/LLMQuantRecipePass.cpp`

---

## PD 图 I/O 约定

### Graph naming
- PD 图：`model.0.pd.s{N}`（如 `model.0.pd.s128`）
- Decode-only 图：`model.0.s1`

### `fusion_ctrl`（int32，shape `[6]`）

- `[0]` `is_prefill_active`（0/1）
- `[1]` `is_decode_active`（0/1）
- `[2]` `prefill_n_update`（本 step prefill 有效 token 数；Kernel B 使用）
- `[3]` reserved
- `[4]` `prefill_n_past`（slot0 的 past_len；Kernel C 使用）
- `[5]` `decode_n_past`（slot1 的 past_len；Kernel C 使用）

### `FusedPDAttention` 的额外量化输入（v0）

为了避免隐式 dequant/cast，Kernel C 显式接收 8 个标量输入：

- `q_scale (float32), q_zp (int32)`
- `k_scale (float32), k_zp (int32)`
- `v_scale (float32), v_zp (int32)`
- `out_scale (float32), out_zp (int32)`

模型侧在 `mllm-main/examples/qwen3_qnn_aot/modeling_qwen_qnn_aot.hpp` 中从参数文件 pull 对应的 `fake_quant.scale/zero_point` 并作为输入传入。

---

## 构建与运行（高层说明）

> 这个项目面向 Android Snapdragon 设备。完整构建链通常需要 QAIRT/QNN SDK + Hexagon SDK + Android NDK。
> 如果你在 Windows 上开发，通常是“写代码/跑单测在 Windows，上机编译/验证在 Linux/Android 工具链”。

### 1) 编译 OpPackage（LLaMAPackage）

目录：`mllm-main/mllm/backends/qnn/custom-op-package/LLaMAPackage/`

该目录的 `Makefile` 需要：
- `QAIRT_SDK_ROOT`（或 `QNN_INCLUDE/QNN_TARGET_LIB`）
- `HEXAGON_SDK_ROOT`
- `ANDROID_NDK_ROOT`（构建 aarch64 目标时）

### 2) AOT 编译 PD context

- `mllm-main/examples/qwen3_qnn_aot/pd_compile.cpp`：编译单一 PD 图。
- `mllm-main/examples/qwen3_qnn_aot/build_context_pd.cpp`：多 bucket + s1 一起打包。

两者都支持 `--op_package_dir`，用于设置/定位 `libQnnLLaMAPackage_{CPU,HTP}.so`。

也可以用环境变量统一配置（AOT + runtime 都能读）：

- `MLLM_QNN_OP_PACKAGE_DIR=<dir-containing-libQnnLLaMAPackage_*.so>`

### 3) 运行 demo

- `mllm-main/examples/qwen3_qnn_aot/pd_run.cpp`
  - 依赖已经生成的 QNN context（包含 `model.0.pd.s{N}` 与 `model.0.s1`）。

---

## 启用/关闭 Kernel C（FusedPDAttention）

默认关闭（因为 reference kernel 不做性能优化）。

开启方式：编译时添加宏：

- `-DMLLM_ENABLE_FUSED_PD_ATTENTION=1`

位置：
- `mllm-main/examples/qwen3_qnn_aot/modeling_qwen_qnn_aot.hpp`

---

## 已知陷阱（Pitfalls）

0) **HTP Watchdog / 超时崩溃（Kernel C reference）**  
`FusedPDAttention` 当前是 correctness-first reference（含 `expf` 的 softmax）。在 4k KV 上它可能非常慢，存在触发 DSP watchdog（如 `RPC_TIMEOUT` / `DSP_CRASH`）的风险。建议：默认保持 `MLLM_ENABLE_FUSED_PD_ATTENTION=0`；若要验证接线，仅用很小的 `past_len`（如 32/64/128）或在小模型/小 head_dim 上做 smoke test。代码侧也对 reference 版做了默认保护：`MLLM_FUSED_PD_ATTENTION_REF_MAX_PAST=256`（past 截断）与 `MLLM_FUSED_PD_ATTENTION_REF_MAX_SEQ=32`（seq_len 保护，可按需调大/关掉）。
另外：reference 版的 `seq_len` 保护也支持运行时覆盖（使用 `fusion_ctrl[3]`）：`0` 使用编译期默认值，`>0` 覆盖最大 seq_len，`<0` 禁用该保护（风险更高）。

1) **In-place / SSA 陷阱（Kernel B）**  
QNN 可能不允许 input/output alias。为避免“失忆”，Kernel B 在 non-alias 情况会先全量 copy 再 patch span（正确但带宽重）。

2) **多图 bucket 切换开销**  
切换 `pd.s32/s64/s128` 不是 0 开销，需要设备侧 profiling 找到最优 bucket 策略与 chunk size。

3) **量化类型与隐式 Dequant/Cast（Kernel C）**  
若不显式提供 scale/zp，QNN 可能插入隐式 Dequant/Cast，导致带宽与延迟不可控。本项目选择显式标量输入避免该问题。

---

## Roadmap（下一步）

- 把 Kernel C 从“correctness-first”升级到真正的 HTP 优化版：
  - quantized attention path（减少 float）
  - VTCM/SRAM 分块与 KV 预取
  - 进一步减少 mask 相关带宽与无效计算
- Kernel A（BranchingLinear）进一步减少 padding 空转（如果不采用多 bucket 或希望更细粒度）。
