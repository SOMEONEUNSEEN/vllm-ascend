# Block Verify 与 Entropy Verify 功能设计文档

## 1. 功能概述

Block Verify 和 Entropy Verify 是 vLLM Ascend 推测解码（Speculative Decoding）拒绝采样流程中的两种优化机制，旨在提升草稿 token 的接受率，从而提高推测解码的整体吞吐量。

### 1.1 Block Verify（块验证）

Block Verify 将同一请求的所有草稿 token 作为一个整体进行验证，使用累积概率乘积（cumulative probability product）来判断接受区间，而非逐 token 独立判断。该机制源自 MagicMTP 思想，在 `max_spec_len >= 3` 时效果显著。

**核心思想**：传统拒绝采样中，每个 token 独立生成均匀随机数 `u_i` 并判断 `p_target(x_i)/p_draft(x_i) >= u_i`，一旦某个 token 被拒绝则后续 token 全部丢弃。Block Verify 改为计算累积接受概率 `π = ∏ min(p_target(x_i)/p_draft(x_i), 1)` 和累积均匀概率 `U = ∏ u_i`，然后找到最大的 `k` 使得 `π_k >= U_k`，接受前 `k` 个 token。

### 1.2 Entropy Verify（熵验证）

Entropy Verify 根据目标模型分布的熵值动态调整接受阈值。高熵（不确定分布）时降低阈值使更多 token 被接受，低熵（确定分布）时保持较严格的阈值。

**核心思想**：计算目标分布的熵 `H = -∑ p(x) log p(x)`，然后计算调整后的阈值 `threshold = min(exp(-H × α), posterior_threshold)`，用 `threshold × U` 替代原始均匀随机数 `U` 作为接受判据。

### 1.3 组合使用

两者可独立启用，也可组合使用。组合时，Block Verify 的累积接受概率与 Entropy Verify 的熵调整阈值同时生效，接受条件变为 `π_k >= threshold_k × U_k`。

---

## 2. 核心算法原理

### 2.1 传统拒绝采样算法

对于请求中的第 `i` 个草稿 token `x_i`：

```
接受条件: p_target(x_i) / p_draft(x_i) >= u_i
其中 u_i ~ Uniform(0, 1)
```

若被拒绝，从修正分布 `max(p_target - p_draft, 0) / q` 中采样恢复 token，其中 `q` 为指数分布归一化因子。

### 2.2 Block Verify 算法

对于请求中所有草稿 token `[x_0, x_1, ..., x_{n-1}]`：

```
1. 初始化: π = 1.0, U = 1.0, last_accepted_pos = -1
2. 对每个 token 位置 pos = 0, 1, ..., n-1:
   a. 计算 ratio = min(p_target(x_pos) / p_draft(x_pos), 1.0)
   b. π = π × ratio
   c. U = U × u_pos   (u_pos 为该位置的均匀随机数)
   d. 若 draft_prob > 0 且 π >= U:
      last_accepted_pos = pos
3. 接受 [0, last_accepted_pos] 范围内的所有草稿 token
4. 若 last_accepted_pos + 1 < n:
   在 last_accepted_pos + 1 处放置恢复 token
5. 否则:
   在末尾放置 bonus token
```

**关键差异**：传统方法在第一个拒绝处立即停止；Block Verify 遍历所有位置计算累积概率，即使中间位置的独立比率较低，只要累积乘积满足条件，后续 token 仍可能被接受。

### 2.3 Entropy Verify 算法

```
1. 计算目标分布的熵:
   H = -∑_{v ∈ V} p(v) × log(p(v) + ε)

2. 计算熵调整阈值:
   threshold = min(exp(-H × α), posterior_threshold)

3. 修改接受条件:
   传统: p_target(x_i) / p_draft(x_i) >= u_i
   熵验证: p_target(x_i) / p_draft(x_i) >= threshold × u_i
```

**参数说明**：
- `posterior_threshold`（默认 0.95）：阈值上限，即使熵极低，有效阈值也不超过此值
- `posterior_alpha`（默认 0.4）：熵缩放因子，控制熵对阈值的影响强度
- 当 `α = 0` 时，`threshold = min(1, posterior_threshold) = posterior_threshold`，熵无影响
- 当 `α` 较大时，高熵 token 的阈值显著降低，接受率大幅提升

### 2.4 Block Verify + Entropy Verify 组合算法

```
1. 初始化: π = 1.0, U = 1.0, last_accepted_pos = -1
2. 对每个 token 位置 pos = 0, 1, ..., n-1:
   a. 计算 ratio = min(p_target(x_pos) / p_draft(x_pos), 1.0)
   b. π = π × ratio
   c. U = U × u_pos
   d. 计算该位置的熵 H_pos 及阈值 threshold_pos
   e. 若 draft_prob > 0 且 π >= threshold_pos × U:
      last_accepted_pos = pos
3. 后续处理同 Block Verify
```

### 2.5 恢复 Token 采样的差异

Block Verify 模式下，恢复 token 的采样分布与传统模式不同：

- **传统模式**：`residual = max(p_target - p_draft, 0)`
- **Block Verify 模式**：`residual = max(π_i × p_target - p_draft, 0)`

其中 `π_i` 是到第 `i` 个位置的累积接受概率。这确保了恢复 token 的分布正确反映了条件概率。

---

## 3. 数据流程

### 3.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AscendRejectionSampler.forward()             │
│                                                                     │
│  输入:                                                              │
│    metadata (SpecDecodeMetadata)                                    │
│    draft_probs [num_tokens, vocab_size] | None                      │
│    logits [num_tokens + batch_size, vocab_size]                     │
│    sampling_metadata (SamplingMetadata)                             │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Bonus Token 采样                                           │
│    bonus_logits = logits[bonus_logits_indices]                      │
│    bonus_token_ids = sampler(bonus_logits)                          │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Target Logits 处理                                         │
│    raw_target_logits = logits[target_logits_indices]                │
│    target_logits = apply_logits_processors(...)                     │
│    target_logits = apply_sampling_constraints(...)                  │
│      → 返回 (target_logits, target_indices) 或 target_logits       │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Entropy Verify 预处理                                      │
│    if using_entropy_verify and ori_target_logits is not None:       │
│        ori_target_probs = ori_target_logits.softmax(dim=-1)         │
│    else:                                                            │
│        ori_target_probs = None                                      │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: rejection_sample() 核心采样                                 │
│    ├─ Greedy 路径: rejection_greedy_sample_with_triton/pytorch      │
│    └─ Random 路径:                                                  │
│         ├─ target_probs = target_logits.softmax(dim=-1)             │
│         ├─ uniform_probs = generate_uniform_probs(...)              │
│         ├─ recovered_token_ids = sample_recovered_tokens(...)       │
│         ├─ if using_block_verify:                                   │
│         │    rejection_random_sample_block_verify_kernel/pytorch    │
│         └─ else:                                                    │
│              rejection_random_sample_kernel/pytorch                 │
└──────────┬──────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  输出: output_token_ids [batch_size, max_spec_len + 1]              │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 关键数据结构

| 数据 | 形状 | 说明 |
|------|------|------|
| `draft_token_ids` | `[num_tokens]` | 草稿模型生成的 token ID |
| `draft_probs` | `[num_tokens, vocab_size]` | 草稿模型概率分布（NGRAM 模式下为 None） |
| `target_probs` | `[num_tokens, vocab_size]` 或 `[num_tokens, selected_vocab_size]` | 目标模型概率分布 |
| `target_indices` | `[num_tokens, selected_vocab_size]` | Reduce Sampling 下的全局词表索引 |
| `bonus_token_ids` | `[batch_size, 1]` | Bonus token ID |
| `recovered_token_ids` | `[num_tokens]` | 恢复 token ID |
| `uniform_probs` | `[num_tokens]` | 均匀随机数 |
| `ori_target_probs` | `[num_tokens, vocab_size]` | 原始目标概率（用于熵计算） |
| `cu_num_draft_tokens` | `[batch_size]` | 累积草稿 token 数 |
| `is_greedy` | `[batch_size]` | 是否贪心采样 |
| `output_token_ids` | `[batch_size, max_spec_len + 1]` | 输出 token ID |

---

## 4. 接口定义

### 4.1 配置接口 — `RejectionSamplerConfig`

```python
class RejectionSamplerConfig:
    def __init__(self, config: dict | None = None):
        self.enable_block_verify: bool = config.get("enable_block_verify", False)
        self.enable_entropy_verify: bool = config.get("enable_entropy_verify", False)
        self.posterior_threshold: float = config.get("posterior_threshold", 0.95)
        self.posterior_alpha: float = config.get("posterior_alpha", 0.4)
        self._validate()
```

**配置参数**：

| 参数 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `enable_block_verify` | bool | False | — | 是否启用块验证 |
| `enable_entropy_verify` | bool | False | — | 是否启用熵验证 |
| `posterior_threshold` | float | 0.95 | (0, 1] | 熵调整阈值上限 |
| `posterior_alpha` | float | 0.4 | >= 0 | 熵缩放因子 |

**使用方式**：

```python
# 在线推理
vllm serve <model> --additional-config \
    '{"rejection_sampler_config": {"enable_block_verify": true, \
    "enable_entropy_verify": true, "posterior_threshold": 0.95, \
    "posterior_alpha": 0.4}}'

# 离线推理
llm = LLM(model, additional_config={
    "rejection_sampler_config": {
        "enable_block_verify": True,
        "enable_entropy_verify": True,
        "posterior_threshold": 0.95,
        "posterior_alpha": 0.4,
    }
})
```

### 4.2 核心采样接口 — `rejection_sample()`

```python
def rejection_sample(
    draft_token_ids: torch.Tensor,           # [num_tokens]
    num_draft_tokens: list[int],             # [batch_size]
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,       # [batch_size]
    draft_probs: torch.Tensor | None,        # [num_tokens, vocab_size]
    target_logits_or_tuple: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    bonus_token_ids: torch.Tensor,           # [batch_size, 1]
    sampling_metadata: SamplingMetadata,
    ori_target_logits: torch.Tensor | None = None,  # [num_tokens, vocab_size]
) -> torch.Tensor:                           # [batch_size, max_spec_len + 1]
```

### 4.3 Triton Kernel 接口

#### `rejection_random_sample_kernel`

标准拒绝采样 Triton kernel，支持 Entropy Verify：

```python
@triton.jit
def rejection_random_sample_kernel(
    output_token_ids_ptr, cu_num_draft_tokens_ptr,
    draft_token_ids_ptr, draft_probs_ptr,
    target_probs_ptr, target_indices_ptr,
    bonus_token_ids_ptr, recovered_token_ids_ptr,
    uniform_probs_ptr, is_greedy_ptr,
    max_spec_len, vocab_size, global_vocab_size, vec_len,
    ori_target_probs_ptr,
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
    POSTERIOR_THRESHOLD: tl.constexpr = 0.95,
    POSTERIOR_ALPHA: tl.constexpr = 0.4,
    SUB_BLOCK: tl.constexpr = 4096,
    EPSILON: tl.constexpr = 1e-10,
)
```

#### `rejection_random_sample_block_verify_kernel`

Block Verify 拒绝采样 Triton kernel，同时支持 Entropy Verify：

```python
@triton.jit
def rejection_random_sample_block_verify_kernel(
    output_token_ids_ptr, cu_num_draft_tokens_ptr,
    draft_token_ids_ptr, draft_probs_ptr,
    target_probs_ptr, target_indices_ptr,
    bonus_token_ids_ptr, recovered_token_ids_ptr,
    uniform_probs_ptr, is_greedy_ptr,
    max_spec_len, vocab_size, global_vocab_size, vec_len,
    ori_target_probs_ptr,
    NO_ORI_TARGET_PROBS: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    ENABLE_REDUCE_SAMPLING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENTROPY_VERIFY: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
    POSTERIOR_THRESHOLD: tl.constexpr = 0.95,
    POSTERIOR_ALPHA: tl.constexpr = 0.4,
    SUB_BLOCK: tl.constexpr = 4096,
    EPSILON: tl.constexpr = 1e-10,
)
```

### 4.4 PyTorch 回退接口

| 函数 | 说明 |
|------|------|
| `rejection_random_sample_pytorch()` | 标准拒绝采样 PyTorch 实现，支持 Entropy Verify |
| `rejection_random_sample_block_verify_pytorch()` | Block Verify 拒绝采样 PyTorch 实现，支持 Entropy Verify |
| `sample_recovered_tokens_pytorch()` | 标准恢复 token 采样 |
| `sample_recovered_tokens_blockwise_pytorch()` | Block Verify 模式下的恢复 token 采样 |

---

## 5. 类结构设计

### 5.1 类图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RejectionSampler (vLLM 上游)                     │
├─────────────────────────────────────────────────────────────────────────┤
│ + sampler: Sampler                                                      │
│ + is_processed_logprobs_mode: bool                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ + forward(metadata, draft_probs, logits, sampling_metadata)             │
│ + apply_logits_processors(...)                                          │
│ + _get_logprobs_tensors(...)                                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ 继承
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AscendRejectionSampler (vLLM Ascend)                 │
├─────────────────────────────────────────────────────────────────────────┤
│ - _ascend_optimizations_enabled: bool                                   │
│ - top_k: int | None                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ + __init__(sampler)                                                     │
│ + forward(metadata, draft_probs, logits, sampling_metadata)             │
│ + apply_penalties(logits, sampling_metadata, metadata, ...)             │
│ + prepare_sampling(top_k)                                               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ 调用
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RejectionSamplerConfig (vLLM Ascend)                 │
├─────────────────────────────────────────────────────────────────────────┤
│ + enable_block_verify: bool                                             │
│ + enable_entropy_verify: bool                                           │
│ + posterior_threshold: float                                            │
│ + posterior_alpha: float                                                │
├─────────────────────────────────────────────────────────────────────────┤
│ + __init__(config: dict | None)                                         │
│ - _validate()                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     模块级函数 (rejection_sampler.py)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ + rejection_sample(...)                                                 │
│ + greedy_sample(logits)                                                 │
│ + apply_sampling_constraints(logits, cu_num_draft_tokens, ...)          │
│ + expand_batch_to_tokens(x, cu_num_tokens, num_tokens, ...)             │
│ + sample_recovered_tokens(...)                                          │
│ - rejection_greedy_sample_spec_len_1_pytorch(...)                       │
│ - rejection_greedy_sample_pytorch(...)                                  │
│ - rejection_random_sample_pytorch(...)                                  │
│ - rejection_random_sample_block_verify_pytorch(...)                     │
│ - sample_recovered_tokens_pytorch(...)                                  │
│ - sample_recovered_tokens_blockwise_pytorch(...)                        │
│ - expand_pytorch(...)                                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     Triton Kernels (reject_sample.py)                    │
├─────────────────────────────────────────────────────────────────────────┤
│ + rejection_greedy_sample_spec_len_1_triton(...)                        │
│ + rejection_greedy_sample_triton(...)                                   │
│ + rejection_random_sample_kernel(...)                                   │
│ + rejection_random_sample_block_verify_kernel(...)                      │
│ + sample_recovered_tokens_kernel(...)                                   │
│ + expand_kernel(...)                                                    │
│ + cal_grid_and_block_size(batch_size)                                   │
│ + rejection_greedy_sample_with_triton(...)                              │
│ + expand_triton(...)                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 类关系说明

- `AscendRejectionSampler` 继承自 vLLM 上游的 `RejectionSampler`，重写了 `forward()`、`apply_penalties()` 等方法以适配 Ascend NPU
- `RejectionSamplerConfig` 作为独立配置类，通过 `get_ascend_config().rejection_sampler_config` 获取
- 模块级函数 `rejection_sample()` 是核心调度入口，根据配置选择不同的采样路径
- Triton Kernel 提供高性能 NPU 实现，PyTorch 函数作为回退

---

## 6. 关键流程实现

### 6.1 拒绝采样路径选择时序图

```
AscendRejectionSampler     rejection_sample()     Triton Kernel     PyTorch Fallback
        │                         │                     │                  │
        │  forward()              │                     │                  │
        ├────────────────────────►│                     │                  │
        │                         │                     │                  │
        │                         │ 读取配置:            │                  │
        │                         │ using_block_verify   │                  │
        │                         │ using_entropy_verify │                  │
        │                         │                     │                  │
        │                         │──── Greedy 路径 ────│                  │
        │                         │────────────────────►│                  │
        │                         │  greedy_sample()    │                  │
        │                         │  rejection_greedy_  │                  │
        │                         │  sample_with_triton │                  │
        │                         │                     │                  │
        │                         │──── Random 路径 ────│                  │
        │                         │                     │                  │
        │                         │ if all_greedy:      │                  │
        │                         │   return             │                  │
        │                         │                     │                  │
        │                         │ 计算 target_probs   │                  │
        │                         │ 生成 uniform_probs  │                  │
        │                         │ 采样 recovered_ids  │                  │
        │                         │                     │                  │
        │                         │── using_block_verify? ──               │
        │                         │                     │                  │
        │                         │ [No] 标准拒绝采样:   │                  │
        │                         ├────────────────────►│                  │
        │                         │ rejection_random_   │                  │
        │                         │ sample_kernel       │                  │
        │                         │  (或 pytorch 回退)   ├─────────────────►│
        │                         │                     │  rejection_      │
        │                         │                     │  random_sample_  │
        │                         │                     │  pytorch         │
        │                         │                     │                  │
        │                         │ [Yes] Block Verify:  │                  │
        │                         ├────────────────────►│                  │
        │                         │ rejection_random_   │                  │
        │                         │ sample_block_       │                  │
        │                         │ verify_kernel       │                  │
        │                         │  (或 pytorch 回退)   ├─────────────────►│
        │                         │                     │  rejection_      │
        │                         │                     │  random_sample_  │
        │                         │                     │  block_verify_   │
        │                         │                     │  pytorch         │
        │                         │                     │                  │
        │  return output_token_ids│                     │                  │
        │◄────────────────────────┤                     │                  │
```

### 6.2 Block Verify 核心流程时序图

```
rejection_sample()    sample_recovered_tokens()    block_verify_kernel/pytorch
      │                        │                            │
      │ 1. 计算 target_probs   │                            │
      │ 2. 生成 uniform_probs  │                            │
      │                        │                            │
      │ 3. 采样恢复 token      │                            │
      ├───────────────────────►│                            │
      │  (blockwise 模式)      │                            │
      │   residual = max(π×p_t - p_d, 0)                   │
      │   recovered_ids = argmax(residual/q)               │
      │                        │                            │
      │ 4. 执行 Block Verify   │                            │
      ├────────────────────────────────────────────────────►│
      │                        │                            │
      │                        │            5. 对每个请求:   │
      │                        │               π = 1.0      │
      │                        │               U = 1.0      │
      │                        │               last_pos = -1│
      │                        │                            │
      │                        │            6. 遍历 token:  │
      │                        │               π *= ratio   │
      │                        │               U *= u_i     │
      │                        │               if ENTROPY:  │
      │                        │                 计算熵 H   │
      │                        │                 threshold  │
      │                        │                 = min(     │
      │                        │                   e^(-Hα), │
      │                        │                   θ)       │
      │                        │                 _U = θ×U   │
      │                        │               else:        │
      │                        │                 _U = U     │
      │                        │               if π >= _U:  │
      │                        │                 last=pos   │
      │                        │                            │
      │                        │            7. 写入结果:    │
      │                        │               接受 [0, k]  │
      │                        │               恢复/补充    │
      │                        │                            │
      │  output_token_ids      │                            │
      │◄───────────────────────────────────────────────────┤
```

### 6.3 Entropy Verify 阈值计算流程

```
┌──────────────────────────────────────────────────────────────┐
│  输入: ori_target_probs [num_tokens, vocab_size]             │
│        (若为 None 则使用 target_probs)                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 1: 计算熵                                              │
│    H = -∑_{v} p(v) × log(p(v) + ε)                         │
│    (分 SUB_BLOCK 块累加，避免单次计算过大)                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 2: 计算熵调整阈值                                      │
│    exp_neg_entropy = exp(-H × α)                            │
│    threshold = min(exp_neg_entropy, posterior_threshold)     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Step 3: 修改接受条件                                        │
│    传统: p_t/p_d >= u                                        │
│    熵验证: p_t/p_d >= threshold × u                          │
│    Block+熵: π >= threshold × U                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 性能优化策略

### 7.1 Triton Kernel 加速

- **自定义 Kernel**：所有核心计算（拒绝采样、恢复 token 采样、扩展操作）均提供 Triton 实现，直接在 NPU Vector Core 上执行
- **Grid/Block 自适应**：`cal_grid_and_block_size()` 根据 batch_size 和 vectorcore_num 自动调整并行度
- **constexpr 特化**：`NO_DRAFT_PROBS`、`ENABLE_REDUCE_SAMPLING`、`ENTROPY_VERIFY` 等编译时常量避免运行时分支

### 7.2 Reduce Sampling 优化

- **Top-K/Top-P 预筛选**：通过 `apply_top_k_top_p()` 预先筛选目标概率分布，仅保留 top-k 候选
- **压缩词表**：`target_indices` 将压缩后的局部索引映射回全局词表，减少计算量和通信量
- **VOCAB_BLOCK_SIZE 分块**：在 Reduce Sampling 模式下，以 512 为块大小遍历候选词表

### 7.3 熵计算优化

- **SUB_BLOCK 分块**：熵计算以 4096 为子块大小分块累加，避免单次分配过大的共享内存
- **ori_target_probs 复用**：优先使用原始 logits 的 softmax 结果计算熵，避免经过 top-k/top-p 截断后熵值失真
- **Fallback 机制**：当 `ori_target_probs` 不可用时，回退使用 `target_probs` 计算熵

### 7.4 内存优化

- **原地操作**：`logits.div_(temperature)` 等原地运算减少内存分配
- **pin_memory**：CPU 端常量张量使用 `pin_memory=True` 加速 CPU→NPU 传输
- **non_blocking 传输**：`to(device, non_blocking=True)` 实现异步数据搬运

### 7.5 分布式优化

- **TP AllGather**：贪心采样通过 `get_tp_group().all_gather()` 跨 rank 聚合最大 logits 和索引
- **局部词表分片**：每个 rank 仅持有部分词表的 logits，通过 `enable_reduce_sample` 控制是否进行 Top-K AllGather

---

## 8. 错误处理机制

### 8.1 配置验证

`RejectionSamplerConfig._validate()` 在初始化时进行严格校验：

| 校验项 | 条件 | 异常信息 |
|--------|------|----------|
| `enable_block_verify` 类型 | `isinstance(bool)` | `must be a bool, got {type}` |
| `enable_entropy_verify` 类型 | `isinstance(bool)` | `must be a bool, got {type}` |
| `posterior_threshold` 类型 | `isinstance(int/float)` | `must be a float, got {type}` |
| `posterior_alpha` 类型 | `isinstance(int/float)` | `must be a float, got {type}` |
| `posterior_threshold` 范围 | `0 < val <= 1` | `must be in (0, 1], got {val}` |
| `posterior_alpha` 范围 | `val >= 0` | `must be >= 0, got {val}` |

### 8.2 运行时断言

```python
assert metadata.max_spec_len <= MAX_SPEC_LEN          # 最大推测长度限制
assert draft_token_ids.ndim == 1                       # 维度校验
assert draft_probs is None or draft_probs.ndim == 2    # 维度校验
assert cu_num_draft_tokens.ndim == 1                   # 维度校验
assert target_logits.ndim == 2                         # 维度校验
assert target_logits.shape[0] == num_tokens            # token 数一致性
assert draft_token_ids.is_contiguous()                 # 内存连续性
assert target_probs.is_contiguous()                    # 内存连续性
```

### 8.3 数值安全

- **EPSILON = 1e-10**：`log(probs + EPSILON)` 防止 `log(0)` 产生 `-inf`
- **draft_prob > 0 检查**：避免 `target_prob / 0` 产生 NaN，draft_prob 为 0 时直接拒绝
- **q 值安全处理**：`q_values_safe = where(q == 0 | isinf(q), epsilon, q)` 防止除零
- **ratio 上限**：`min(ratio, 1.0)` 确保累积概率不超过 1

### 8.4 回退机制

- **Triton 不可用**：当 `HAS_TRITON = False` 时，自动回退到 PyTorch 实现
- **Reduce Sampling 不可用**：当 `target_indices` 为 None 时，回退到全词表模式
- **ori_target_probs 不可用**：当 `ori_target_probs` 为 None 时，使用 `target_probs` 计算熵
- **Fallback 路径警告**：非 Reduce Sampling 路径会输出 `logger.warning_once`

### 8.5 Block Verify 前置条件

```python
using_block_verify = max_spec_len >= 3 and bool(
    get_ascend_config().rejection_sampler_config.enable_block_verify
)
```

即使配置启用了 Block Verify，当 `max_spec_len < 3` 时仍自动降级为标准模式。

---

## 9. 测试用例设计

### 9.1 Block Verify 测试

| 测试场景 | 验证要点 |
|----------|----------|
| 部分接受 | 前 k 个 token 被接受，第 k+1 个被替换为恢复 token |
| 全部接受 + Bonus | 所有草稿 token 被接受，末尾附加 bonus token |
| 全部拒绝 | 第一个 token 即被拒绝，放置恢复 token |
| 多请求批量 | 不同请求有不同的接受位置 |
| NGRAM 模式 | `draft_probs=None`，`draft_prob=1.0` |
| Reduce Sampling | 使用 `target_indices` 进行压缩词表查找 |

### 9.2 Entropy Verify 测试

| 测试场景 | 验证要点 |
|----------|----------|
| 高熵分布 → 更易接受 | 均匀分布下阈值降低，接受率提升 |
| 低熵分布 → 更严格 | 尖峰分布下阈值接近 `posterior_threshold` |
| 无 ori_target_probs 回退 | `ori_target_probs=None` 时使用 `target_probs` |
| NGRAM + Entropy Verify | `draft_probs=None` 与熵验证组合 |
| Block Verify + Entropy Verify | 累积概率与熵调整阈值同时生效 |
| NGRAM + Block + Entropy | 三种模式组合 |

### 9.3 边界与异常测试

| 测试场景 | 验证要点 |
|----------|----------|
| `max_spec_len < 3` | Block Verify 自动降级为标准模式 |
| `draft_prob = 0` | 不产生 NaN，直接拒绝 |
| `draft_token_id = -1` | 占位符 token 正确处理 |
| 空 batch | `num_tokens = 0` 不崩溃 |
| `posterior_alpha = 0` | 熵无影响，阈值等于 `posterior_threshold` |
| `posterior_threshold = 1.0` | 阈值上限为 1.0 |
| 配置类型错误 | `_validate()` 抛出 `ValueError` |
| 配置范围错误 | `posterior_threshold > 1` 或 `< 0` 抛出异常 |

### 9.4 精度对比测试

| 测试场景 | 验证要点 |
|----------|----------|
| Triton vs PyTorch 一致性 | 两种实现的输出一致 |
| Block Verify vs 标准模式 | Block Verify 接受率 >= 标准模式 |
| Entropy Verify 接受率变化 | 高熵 token 接受率提升 |
| Reduce Sampling 一致性 | 压缩词表与全词表结果一致 |

---

## 10. 未来扩展方向

### 10.1 自适应阈值调整

当前 `posterior_threshold` 和 `posterior_alpha` 为静态配置。未来可引入运行时自适应机制：

- 根据历史接受率动态调整 `posterior_alpha`
- 基于当前 batch 的平均熵自动选择阈值参数
- 引入 PID 控制器式的反馈调节

### 10.2 多级 Block Verify

当前 Block Verify 为全量块验证。可探索分级策略：

- **滑动窗口 Block Verify**：以固定窗口大小进行局部块验证
- **分层 Block Verify**：先以粗粒度块验证快速筛选，再以细粒度验证精确判断
- **自适应块大小**：根据草稿 token 的置信度动态调整块大小

### 10.3 熵感知草稿模型调度

将 Entropy Verify 的熵信息反馈给草稿模型：

- 高熵位置减少草稿 token 数量（低价值推测）
- 低熵位置增加草稿 token 数量（高价值推测）
- 实现草稿深度的动态调整

### 10.4 硬件感知优化

- **多缓冲并行**：`sample_recovered_tokens_kernel` 已预留 `multibuffer` 参数，待精度问题解决后启用
- **ACL Graph 集成**：将拒绝采样流程捕获为 ACL Graph，减少 CPU→NPU 调度开销
- **混合精度**：探索 FP16/BF16 熵计算以减少内存带宽压力

### 10.5 上游贡献

- 将 Block Verify 和 Entropy Verify 机制贡献至 vLLM 上游
- 统一 Ascend 与 CUDA 的 Triton Kernel 接口
- 推动标准化 `RejectionSamplerConfig` 作为 vLLM 核心配置

### 10.6 更多验证策略

- **Top-K Verify**：仅验证目标模型 top-k 候选中的草稿 token
- **Confidence-Based Verify**：基于草稿模型置信度调整验证强度
- **Distillation-Guided Verify**：利用蒸馏信号指导接受/拒绝决策
