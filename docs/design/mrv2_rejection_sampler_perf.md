# Model Runner V2 拒绝采样（Rejection Sample）性能优化设计文档

## 1. 引言

上游 vLLM 已将全部模型迁移至 Model Runner V2 架构（`C:\code\vllm\vllm\v1\worker\gpu\model_runner.py`），vllm-ascend 同步完成了 MRV2 迁移（`c:\code\vllm-ascend-lys1\vllm_ascend\worker\v2\model_runner.py`）。拒绝采样作为投机解码（Speculative Decoding）的核心环节，其 V1 → V2 迁移已完成基础功能对接，本文档旨在系统分析差异、梳理迁移过程，并制定 V2 环境下的性能优化方案。

**关键文件定位：**

- V1 拒绝采样：`vllm_ascend/sample/rejection_sampler.py`（`AscendRejectionSampler`）
- V2 拒绝采样：`vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`
- V2 Model Runner：`vllm_ascend/worker/v2/model_runner.py`
- 上游 V2 基线：`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
- 上游 V2 Triton 工具：`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`

---

## 2. Model Runner V1 与 V2 版本拒绝采样功能的核心区别

### 2.1 架构设计差异

**Model Runner V1：**

- 核心类 `AscendRejectionSampler`（继承上游 `RejectionSampler`），位于 `vllm_ascend/sample/rejection_sampler.py`
- 采用**包装模式**：在 vllm-ascend 中覆写上游类，注入 NPU 优化逻辑
- 数据流：`forward()` → `apply_logits_processors()`（penalties/bad_words/top-k/top-p）→ `apply_sampling_constraints()`（温度缩放 + top-k + all-gather + top-p）→ `rejection_sample()` → Triton kernels
- 显式处理 **TP 分布式 all-gather**（`greedy_sample()` 中 `tp_group.all_gather()`）
- 独立 Triton kernels：`rejection_random_sample_kernel`、`rejection_greedy_sample_with_triton`、`sample_recovered_tokens_kernel`，位于 `vllm_ascend/ops/triton/reject_sample.py`
- 支持 `enable_reduce_sample` 配置，走 top-k → all-gather → top-p 路径

**Model Runner V2：**

- 核心函数 `rejection_sample()` 为**纯函数式接口**（无类实例），位于 `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py`
- 采用**直接 Triton kernel 调用**：4 个核心 kernel（`_compute_block_stats_kernel` → `_probabilistic_rejection_kernel` → `_resample_kernel` → `_insert_resampled_kernel`）
- 数据流：`model_runner` 直接调用 `rejection_sample()`，所有采样约束处理（温度/top-k/top-p/penalties/bad_words）已在上游 `Sampler.apply_sampling_params()` 中统一完成
- **不再显式处理 TP all-gather**：上游 `Sampler` 内部已处理分布式逻辑
- 采样约束（top-k/top-p/温度/penalties）与拒绝采样解耦

### 2.2 接口与参数变化

| 维度 | V1 (`AscendRejectionSampler.forward()`) | V2 (`rejection_sample()`) |
|------|----------------------------------------|--------------------------|
| **输入** | `metadata: SpecDecodeMetadata`, `draft_probs`, `logits`, `sampling_metadata` | `target_logits`, `draft_logits`, `draft_sampled`, `cu_num_logits`, `pos`, `idx_mapping`, `expanded_idx_mapping`, `expanded_local_pos`, `temperature`, `seed`, `num_speculative_steps` |
| **draft 表示** | `draft_probs`（概率分布 `[num_tokens, vocab]`）| `draft_logits`（原始 logits `[max_num_reqs, num_spec_steps, V]`）|
| **元数据来源** | `SpecDecodeMetadata`（由 scheduler 构建）| `InputBatch` 字段（由 `model_runner` 准备）|
| **温度/采样** | `sampling_metadata`（CPU 元数据）| `temperature`, `seed`（GPU 张量）|
| **输出** | `SamplerOutput`（包含 logprobs）| `(sampled, num_sampled)` 元组 |
| **chunked verify** | 不支持 | 上游支持 `_verify_in_chunks()`（内存保护）|

### 2.3 核心算法实现差异

**V1 算法路径：**

1. 提取 `bonus_logits` 和 `target_logits`（索引切片）
2. `apply_logits_processors()`：penalties → bad_words → min_tokens
3. `apply_sampling_constraints()`：温度缩放 → top-k → all-gather → top-p
4. `rejection_sample()` → Triton kernels 逐 token 拒绝/接受

**V2 算法路径（4-kernel 流水线）：**

1. **`_compute_block_stats_kernel`**：预计算每个 vocab block 的 local argmax/max/sumexp（greedy 用 argmax，非 greedy 用 max + sumexp），支持 draft logits 的 stats 计算
2. **`_probabilistic_rejection_kernel`**：主拒绝逻辑，串行遍历每个请求的 spec tokens，逐 token 判断接受/拒绝（greedy: argmax 比对；非 greedy: log(p) > log(u) + log(q)）
3. **`_resample_kernel`**：对拒绝/bonus token 进行重采样（NPU Gumbel-max 实现）
4. **`_insert_resampled_kernel`**：将重采样结果写入输出

**关键算法差异：**

- V2 引入 **block-level 统计量预计算**（`_compute_block_stats_kernel`），将 vocab 维度分块（`VOCAB_BLOCK_SIZE=8192`），避免每个请求重复加载完整 vocab logits
- V2 的 Gumbel-max 重采样使用**块级并行**（`_npu_gumbel_block_argmax`），而非 V1 的串行逐 token
- V1 在 `apply_sampling_constraints` 中做 all-gather，V2 完全跳过这一步（由上游 Sampler 处理）
- V2 原生支持 **synthetic mode**（合成验收率模式）和 **block verification**（`use_block_verification`，但 NPU 端尚未实现）

### 2.4 性能表现初步对比

基于代码分析识别的性能特征：

| 指标 | V1 | V2 |
|------|----|----|
| **kernel 调用次数** | 2-3 次（greedy/random + recover） | 4 次（stats + reject + resample + insert）|
| **温度处理** | 逐 token `div_()` | 融入 kernel 内部（`APPLY_TEMPERATURE`）|
| **TP all-gather** | 显式调用 | 无（上游处理）|
| **内存分配** | 每次 forward 动态分配 | 复用 `target_logits.new_empty()` |
| **chunked verify** | 不支持 | 上游支持（1GB 上限）|
| **block verify** | 支持（`enable_block_verify`） | 未实现（NPU TODO）|

---

## 3. 拒绝采样功能从 Model Runner V1 到 V2 的迁移策略与实施过程

### 3.1 迁移准备工作

- 环境：vllm-ascend 已切换到 vLLM 上游 MRV2（`vllm.v1.worker.gpu.model_runner`），Ascend MRV2 实现继承自 `GPUModelRunner`
- 代码库：V2 拒绝采样相关文件统一在 `vllm_ascend/worker/v2/spec_decode/` 目录下
- 测试基准：确保 V1 功能正确性作为迁移参照

### 3.2 迁移实施步骤

1. **移植 Triton kernels**：将上游 `rejection_sampler_utils.py` 中的 4 个 kernel 适配到 NPU Triton（`vllm.triton_utils`）
2. **NPU 特定适配**：
   - `tl.randint(seed, pos)`：pos 需 `.to(tl.int32)`（NPU `umulhi` 不支持 uint64）
   - `tl.rand()` 替代 `tl_rand64/tl_rand32`（NPU 不支持 float64 标量 rand）
   - Gumbel 噪声用 float32（`-tl.log(-tl.log(r + 1e-20) + 1e-20)`）
   - `float32` 替代 `float64`（`resampled_local_max`）
3. **集成到 Model Runner**：在 `model_runner.py` 的采样分支处直接调用 `rejection_sample()`
4. **移除 V1 遗留**：`AscendRejectionSampler` 不再被 V2 model runner 使用（仍保留给 V1 runner）

### 3.3 兼容性处理方案

- **NPU Triton 兼容性**：
  - `has_auto_blockify_blacklist_op=True`：绕过 Triton Ascend AutoBlockify 对 max-with-index 归约的 bug
  - `use_fp64=False`：NPU 不支持 FP64 Gumbel 噪声，硬编码禁用
  - `use_block_verification=False`：NPU 未实现 block verify，接口保留但抛出 `NotImplementedError`
- **V1/V2 共存**：V1 runner（`model_runner_v1.py`）仍使用 `AscendRejectionSampler`；V2 runner 使用独立 `rejection_sample()` 函数

### 3.4 迁移验证与测试

- 功能验证：对比 V1/V2 输出 token 序列一致性（greedy 场景）
- 概率场景：验证 `_probabilistic_rejection_kernel` 接受率与 V1 对齐
- 边界场景：全拒绝（bonus token）、全接受、`num_draft_tokens=0`（非 spec decode 路径）

---

## 4. Model Runner V2 拒绝采样功能性能优化目标

### 4.1 性能指标定义

- **吞吐量**：拒绝采样阶段（`_verify`）的端到端延迟应占总采样时间 < 5%
- **延迟**：`_probabilistic_rejection_kernel` 单步延迟 ≤ V1 的 80%
- **资源利用率**：NPU 计算单元利用率 ≥ 85%（通过 `msprof` 测量）
- **内存**：FP32 logits 缓冲区峰值 ≤ 1GB（对齐上游 `MAX_CHUNK_BYTES`）

### 4.2 优化范围界定

**高优先级：**

1. `_compute_block_stats_kernel` 的 NPU 优化（最大计算密度 kernel）
2. `_probabilistic_rejection_kernel` 的串行循环优化（V2 关键瓶颈）
3. `_resample_kernel` 的 Gumbel-max 块并行效率

**低优先级：**

4. 与 ACLGraph 的集成（当前 V2 拒绝采样不在 graph 中）
5. FP16 替代 FP32（精度风险较高）

---

## 5. 性能优化方案设计

### 5.1 算法层面优化

1. **块级统计量与拒绝逻辑融合**：当前 `_compute_block_stats_kernel` 和 `_probabilistic_rejection_kernel` 分别遍历 vocab 和 spec tokens。可将 block stats 计算融入 rejection kernel，减少一次完整 vocab 遍历
2. **提前退出（Early Exit）**：`_probabilistic_rejection_kernel` 中一旦 `accepted=False`，后续 tokens 可直接跳过（当前仍遍历 `range(num_tokens - 1)`）
3. **温度预处理**：对 `temp == 0.0`（greedy）的请求完全跳过非 greedy 分支的 LSE 计算

### 5.2 工程实现优化

1. **异步 kernel 发射**：当前 4 个 kernel 串行发射，可通过 NPU stream 并行化无依赖的阶段（如 stats 计算与 bonus token 准备）
2. **内存复用**：`target_local_argmax/max/sumexp` 和 `draft_local_max/sumexp` 共 5 个中间张量，可通过内存池或 in-place 复用减少分配开销
3. **BLOCK_SIZE 调优**：`VOCAB_BLOCK_SIZE=8192` 和 `RESAMPLE_BLOCK_SIZE=1024` 需针对 Ascend 910B/C 的向量核心宽度重新调优

### 5.3 Ascend 平台适配优化

1. **Triton AutoBlockify 修复**：当前 `has_auto_blockify_blacklist_op=True` 是 workaround，需推动 Triton-Ascend 修复 max-with-index 归约的 AutoBlockify bug，移除黑名单后可提升 kernel 编译效率
2. **标量 rand 支持**：NPU Triton 缺少标量 `tl.rand`，当前用 `tl.max(tl.rand(seed, tl.arange(0, 1)))` 模拟，引入额外归约开销。推动 Triton-Ascend 支持标量 rand 可消除此开销
3. **int32 vs int64 索引**：`expanded_idx_mapping` 和 `pos` 当前为 int32/int64 混合，统一为 int32 可减少类型转换

### 5.4 配置参数调优

| 参数 | 当前值 | 建议调优范围 | 说明 |
|------|--------|-------------|------|
| `VOCAB_BLOCK_SIZE` | 8192 | [4096, 8192, 16384] | 影响 stats kernel 的并行度与寄存器压力 |
| `RESAMPLE_BLOCK_SIZE` | 1024 | [512, 1024, 2048] | 影响 resample kernel 的块内归约效率 |
| `num_warps` | 1（rejection kernel） | [1, 2, 4] | 串行循环 kernel，多 warp 可能无收益 |

---

## 6. 优化实施计划与里程碑

### 6.1 分阶段实施计划

**阶段 1（基础 profiling）：**

- 使用 `msprof` 对当前 V2 拒绝采样 4 个 kernel 做性能分析
- 确定瓶颈 kernel（预计 `_compute_block_stats_kernel` 或 `_probabilistic_rejection_kernel`）
- 输出：kernel 级耗时分布报告

**阶段 2（kernel 优化）：**

- 实施 5.1-5.3 中确定的优化项
- 单元测试验证正确性
- 输出：优化后 kernel 代码 + 性能对比数据

**阶段 3（集成验证）：**

- 端到端 spec decode 场景测试
- 与 V1 性能对比
- 输出：性能优化报告

### 6.2 关键里程碑定义

| 里程碑 | 交付物 | 验收标准 |
|--------|--------|----------|
| M1: Profiling 完成 | kernel 耗时报告 | 识别出 Top-1 瓶颈 kernel |
| M2: Kernel 优化完成 | 优化代码 + UT | 正确性通过，性能提升 ≥ 15% |
| M3: 端到端验证 | 性能对比报告 | 总延迟降低 ≥ 10%，接受率不变 |

---

## 7. 测试与验证策略

### 7.1 性能测试方案

- **工具**：`msprof`（NPU kernel profiling）、`torch.profiler`（端到端）
- **基准数据集**：ShareGPT（真实分布）、合成均匀分布 draft tokens
- **测量方法**：固定 batch_size ∈ {1, 8, 32, 64}，num_speculative_tokens ∈ {1, 3, 5}，vocab_size = 151936（Qwen2.5）
- **对比基线**：V1 rejection sampling 延迟 + V2 优化前延迟

### 7.2 功能验证方案

- **正确性回归**：对比优化前后 `sampled` 和 `num_sampled` 输出完全一致
- **接受率一致性**：统计 acceptance rate，偏差 < 0.1%
- **Greedy vs 非 Greedy**：两种路径分别验证

### 7.3 稳定性与可靠性测试

- **长稳测试**：连续运行 10000 步 spec decode，无内存泄漏、无 NaN
- **压力测试**：max batch + max spec tokens + 最大 vocab size
- **异常场景**：`draft_logits=None`（ngram 路径）、`temp=0`（纯 greedy）、`all_rejected`（bonus only）

---

## 8. 风险分析与应对措施

### 8.1 潜在风险识别

| 风险类型 | 描述 | 影响 |
|----------|------|------|
| **技术风险** | Triton-Ascend 编译器 bug（AutoBlockify）导致 kernel 优化受限 | 性能提升幅度不及预期 |
| **技术风险** | FP32 → FP16 精度损失导致接受率下降 | 输出质量下降 |
| **进度风险** | NPU Triton 标量 rand 支持需上游排期 | 优化项延期 |
| **质量风险** | kernel 融合引入边界条件 bug | 功能回归 |

### 8.2 风险应对策略

- **技术风险**：保留 `has_auto_blockify_blacklist_op` workaround 作为回退方案；优先优化不依赖编译器修复的路径
- **精度风险**：FP16 优化设为可选配置（环境变量控制），默认保持 FP32
- **进度风险**：标量 rand 优化降级为"减少 1-element block 归约次数"（如合并多个 u 的生成）
- **质量风险**：每个优化项独立提交，配完整的单元测试和回归测试

---

## 9. 结论与展望

V2 拒绝采样迁移已完成核心功能对接，4-kernel 流水线架构相比 V1 的包装模式更加清晰，采样约束处理与拒绝逻辑解耦为独立优化提供了可能。当前主要优化空间在于：

1. **块级统计量预计算**是 V2 的核心创新，但其 NPU 性能尚未充分调优
2. **串行拒绝循环**是 V2 的主要瓶颈（`for i in range(num_tokens - 1)`），需通过提前退出和循环展开优化
3. **NPU Triton 限制**（无标量 rand、无 FP64、AutoBlockify bug）是当前的主要约束

**未来优化方向：**

- 将拒绝采样纳入 ACLGraph（当前不在 graph 中，kernel 发射开销可进一步降低）
- Block verification 的 NPU 实现（上游已支持，NPU 端 TODO）
- 基于接受率预测的动态 spec tokens 调整（adaptive verification）

---

## 附录

### 关键代码路径索引

| 模块 | 文件 | 核心函数/类 |
|------|------|------------|
| V2 拒绝采样入口 | `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` | `rejection_sample()` |
| V2 块统计 kernel | 同上 | `_compute_block_stats_kernel` |
| V2 概率拒绝 kernel | 同上 | `_probabilistic_rejection_kernel` |
| V2 重采样 kernel | 同上 | `_resample_kernel` |
| V2 NPU Gumbel argmax | 同上 | `_npu_gumbel_block_argmax` |
| V2 Model Runner 集成 | `vllm_ascend/worker/v2/model_runner.py` | `rejection_sampler()` 调用点 |
| V1 拒绝采样 | `vllm_ascend/sample/rejection_sampler.py` | `AscendRejectionSampler` |
| V1 Triton kernels | `vllm_ascend/ops/triton/reject_sample.py` | `rejection_random_sample_kernel` |
| 上游 V2 参考 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | `RejectionSampler` |
| 上游 V2 工具 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | `rejection_sample()` |
| V2 spec decode 初始化 | `vllm_ascend/worker/v2/spec_decode/__init__.py` | `init_speculator()` |
