# MRV2 Block Verification 测试方案

本方案针对分支 `codex/mrv2-block-verification`，验证以下两个提交：

- `b21c79888`：接入 MRV2 block verification。
- `76029b23e`：所有模式统一按照上游语义处理 placeholder。

rebase 后主干基线为 `23d1d9055f2161b3c15613c7fc309f7d5d2ef629`。该主干验证使用的上游 vLLM 基线为 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`。

当前仅通过静态检查；以下 NPU 用例尚未在本机执行通过。启动成功、用例被 skip、greedy 输出一致，都不能单独证明随机 block verification 已生效。

## 1. 环境与源码确认

本地源码定位：

| 用途 | 文件 |
| --- | --- |
| 上游 MRV2 | `C:\code\vllm\vllm\v1\worker\gpu\model_runner.py` |
| 上游 verification 包装层 | `C:\code\vllm\vllm\v1\worker\gpu\spec_decode\rejection_sampler.py` |
| Ascend MRV2 | `C:\code\vllm-ascend-lys1\vllm_ascend\worker\v2\model_runner.py` |
| Ascend verification | `C:\code\vllm-ascend-lys1\vllm_ascend\worker\v2\spec_decode\rejection_sampler_utils.py` |
| Ascend recovery | `C:\code\vllm-ascend-lys1\vllm_ascend\ops\triton\v2\spec_decode\resample.py` |

以下命令在 Linux Ascend NPU 环境的 Bash 中执行。服务器路径以实际部署为准；`ASCEND_REPO` 对应上述 Ascend 仓库，不能指向另一份旧安装。使用已安装匹配的 PyTorch、torch-npu、Triton-Ascend、vLLM 和测试依赖的解释器。

```bash
# 替换成服务器上的真实路径。
export ASCEND_REPO=/path/to/vllm-ascend
export TEST_PYTHON=/path/to/npu-env/bin/python
export RESULT_DIR=/tmp/mrv2-block-verification-results
mkdir -p "$RESULT_DIR"
cd "$ASCEND_REPO"
set -o pipefail

git log -2 --oneline
git merge-base --is-ancestor 76029b23e HEAD
npu-smi info | tee "$RESULT_DIR/npu-info.txt"

"$TEST_PYTHON" - <<'PY' | tee "$RESULT_DIR/environment.txt"
import inspect
import torch
import torch_npu
import triton
import vllm
import vllm_ascend

assert torch.npu.is_available(), "NPU is required"
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__version__)
print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)

# 独立进程检查 patch 后的函数绑定，不代表真实服务进程已执行此路径。
import vllm_ascend.patch.worker.patch_v2.patch_triton
from vllm.v1.worker.gpu.spec_decode import rejection_sampler
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

assert rejection_sampler.rejection_sample is rejection_sample
print("rejection_sample:", inspect.getsourcefile(rejection_sampler.rejection_sample))
PY
```

通过标准：NPU 可用；导入路径和待测 checkout 一致；函数绑定指向 Ascend 实现。记录实际 vLLM commit、CANN/驱动版本、卡型、卡数及完整启动参数。若更换上游 vLLM 版本，需单独检查复用函数的签名兼容性。

## 2. 必测：算子功能与正确性

已有测试文件：

- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py`
- `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py`

先运行这两个文件的完整测试，包括原有回归用例：

```bash
"$TEST_PYTHON" -m pytest -sv \
  "$ASCEND_REPO/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py" \
  "$ASCEND_REPO/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py" \
  --junitxml="$RESULT_DIR/operator-tests.xml" \
  2>&1 | tee "$RESULT_DIR/operator-tests.log"
```

pytest 收集失败、NPU 不可用导致 skip、编译失败或运行时报错，均不计为通过。首次执行包含 Triton 编译时间，不用于性能比较。

| 验证目的 | 已有测试函数 | 通过标准 |
| --- | --- | --- |
| block 参数真正改变算法 | `test_block_verification_extends_accepted_prefix` | 固定三 token 分布下，block 平均接受 draft 数比 standard 大于 0.02；不能只是接受参数后仍运行 standard |
| 输出分布保持正确 | `test_block_verification_preserves_target_distribution` | K=1/3、温度=0.6/1.0、完整/one-hot draft，所有有效输出位置符合 target 分布；误差界由各位置样本数计算 |
| prefix residual 缩放正确 | `test_block_residual_scales_target_by_accepted_prefix` | 完整 draft 恢复采用 `max(p_tau * p - q, 0)`，包括跨词表块的质量计算和最终选 token |
| 接受与恢复随机流独立 | `test_rejection_sample_residual_is_independent_of_acceptance_draw` | standard/block × 完整/one-hot draft × 第 0/1 步拒绝，共 8 组；恢复分布不受接受阈值条件化影响 |
| placeholder 统一恢复 | `test_placeholder_recovers_from_target` | standard/synthetic/block × 完整/one-hot draft × greedy/random × placeholder 位于第 0/1 步，共 24 组；不得继续接受 placeholder 之后的 draft |
| 变长请求和映射 | `test_block_verification_ragged_requests_and_mapping` | 请求输出长度为 `[4, 1, 3, 2]`，正确使用非连续 request-state 映射；覆盖 bonus-only、混合温度、8193 词表和最后一个短请求 |
| greedy 回归 | `test_block_verification_keeps_greedy_results` | block 与 standard 的有效 token 和数量完全一致；不比较未写入的输出尾部 |
| 不兼容配置 | `test_block_verification_rejects_synthetic_rates` | block 与 synthetic rates 同时传入时明确报错 |
| 词表 padding | `test_draft_vocab_narrower_than_target_vocab` | standard/block 均与未 padding 的输入结果一致，不采样 padding 列、不越界 |

两个最重要的数值判据：

1. 接受长度用例的理论平均接受 draft 数为 standard≈2.439、block≈2.509。这里不含额外恢复/bonus token；若看包含该 token 的长度，对应约 3.439 和 3.509。测试比较的是两种模式的差值。
2. placeholder 用例的 target 为 `[0.1, 0.2, 0.3, 0.4]`：随机模式下恢复分布应匹配该向量，绝对误差不超过 0.03；greedy 恢复 token 应为 3。完整 draft 设置为与 target 相同，可暴露错误使用零 residual 后回退 token 0 的问题。

只想先确认“修改是否生效”，可先跑以下三个节点，再跑全文件回归：

```bash
"$TEST_PYTHON" -m pytest -sv \
  "$ASCEND_REPO/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py::test_block_verification_extends_accepted_prefix" \
  "$ASCEND_REPO/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py::test_placeholder_recovers_from_target" \
  "$ASCEND_REPO/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py::test_block_residual_scales_target_by_accepted_prefix"
```

统计用例失败时保留原始输出和环境，分析样本数、RNG、分布与索引问题；不要以“重跑到通过”为验收方式。固定 seed 的重复运行主要验证可重复性，不是增加独立统计样本。

## 3. 必测：真实模型配置与端到端路径

使用已有跑通的 MRV2 target/draft 模型组合。仓库已有 `Qwen/Qwen3-8B` + `deepseek-ai/dspark_qwen3_8b_block7` 测试，可作为参考；需要对应权重和足够 NPU 内存。其他模型组合按其实际 spec method 配置，不直接套用 DSpark 的 K=7。

配置重点：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

```python
speculative_config = {
    "model": "/actual/path/to/dspark_qwen3_8b_block7",
    "method": "dspark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "rejection_sample_method": "block",
    "enable_adaptive_verification": False,
}
```

`draft_sample_method="probabilistic"` 用于覆盖完整 draft logits 路径；`"greedy"` 用于覆盖 one-hot 路径。`rejection_sample_method` 才决定 standard/block，不能用 MRV1 的 `rejection_sampler_config.enable_block_verify` 替代。target `temperature=0` 主要用于回归，必须另外测试 `temperature>0`。

下面的 smoke 脚本可复制到 `$RESULT_DIR/run_block_e2e.py`。它是执行模板，尚未在 NPU 上验证；每组参数启动独立进程，避免同时驻留两个模型实例。

```python
import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--method", choices=["standard", "block"], required=True)
    parser.add_argument("--draft-sampling", choices=["greedy", "probabilistic"], default="probabilistic")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=1024,
        max_num_seqs=8,
        enforce_eager=args.eager,
        disable_log_stats=False,
        enable_prefix_caching=False,
        async_scheduling=True,
        speculative_config={
            "model": args.draft,
            "method": "dspark",
            "num_speculative_tokens": 7,
            "draft_sample_method": args.draft_sampling,
            "rejection_sample_method": args.method,
            "enable_adaptive_verification": False,
        },
    )
    prompts = [
        "Explain why the sky appears blue in simple terms.",
        "Write a Python function that reverses a list.",
        "请用三句话介绍二分查找。",
        "Continue this story: A small robot found a garden on the moon.",
    ] * 8
    params = SamplingParams(
        temperature=args.temperature, top_p=1.0, top_k=-1,
        max_tokens=128, ignore_eos=True, seed=42,
    )

    def counters():
        names = (
            "vllm:spec_decode_num_drafts",
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        )
        values = {name: 0 for name in names}
        for metric in llm.get_metrics():
            if metric.name in values:
                values[metric.name] += metric.value
        return values

    llm.generate(prompts[:4], params, use_tqdm=False)
    before = counters()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    after = counters()
    delta = {name: after[name] - before[name] for name in before}
    drafts = delta["vllm:spec_decode_num_drafts"]
    assert drafts > 0, "No speculative verification rounds were observed"
    ids = [out.outputs[0].token_ids for out in outputs]
    assert len(ids) == len(prompts)
    assert all(len(tokens) == 128 for tokens in ids)
    assert all(token >= 0 for tokens in ids for token in tokens)
    report = {
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(map(len, ids)) / elapsed,
        "metrics_delta": delta,
        "mean_accepted_drafts": delta["vllm:spec_decode_num_accepted_tokens"] / drafts,
        "token_ids": ids,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "token_ids"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

执行示例，替换模型路径；先 eager，再去掉 `--eager` 测实际默认图模式：

```bash
for method in standard block; do
  "$TEST_PYTHON" "$RESULT_DIR/run_block_e2e.py" \
    --model /actual/path/to/Qwen3-8B \
    --draft /actual/path/to/dspark_qwen3_8b_block7 \
    --method "$method" --draft-sampling probabilistic \
    --temperature 0.7 --eager \
    --output "$RESULT_DIR/eager-${method}.json" \
    2>&1 | tee "$RESULT_DIR/eager-${method}.log"
done
```

端到端矩阵：

| 维度 | 必测值 |
| --- | --- |
| 验证模式 | standard、block |
| draft 采样 | greedy、probabilistic |
| target 温度 | 0、0.7 |
| 执行模式 | eager、部署使用的图模式 |

共 16 种组合。相同组合固定输入、seed、模型和版本。首次测试可用上述 smoke 负载，正式验收替换为固定真实业务 prompts，包含不同输入长度及长输出。

通过标准：

- 无编译错误、非法访存、NaN 异常或进程挂死；每个请求完整返回，speculative 计数非零。
- greedy 下，同执行模式、同 draft 采样配置的 standard/block 有效 token ID 一致；原有图模式对齐问题需单独排查。
- 随机模式不要求 standard/block 文本逐字一致，也不要求与 GPU 使用相同 seed 时逐 token 一致；两种验证模式的接受策略不同。分布正确性由第 2 节受控测试判定，真实模型质量需用既有评测集补充验证。
- kernel trace 中，block 应出现 `_compute_cumulative_log_p_kernel`；完整 draft logits 且有有效后续 draft 时还应出现 `_compute_local_residual_mass_kernel`。standard 不应新增这些预计算。使用部署环境已有 profiler，预热后采集实际 decode 阶段；kernel 名可能带编译后缀。
- 开启 block 不保证每种业务负载都提高吞吐或每个 batch 都增加接受长度。真实模型收益作为单独结果记录，不能代替正确性验收。

已有 `test_qwen3_8b_dspark.py` 可做旧模式回归，但它未设置 `rejection_sample_method="block"`，单独跑过该文件不能证明本次功能生效。其 standard acceptance golden 也不能直接用于 block 验收。

## 4. 建议补充：兼容性和性能

以下内容不在本次新增算子测试的完整覆盖范围内，按实际部署需要补做：

| 场景 | 验证重点 |
| --- | --- |
| Async scheduling、混合 prefill/decode、请求加入退出 | request-state 复用、变长 mapping、padding、不读旧 draft 数据 |
| DSpark adaptive verification | 先通过非 adaptive 基线，再单独打开；确认压缩后的边界及短请求无越界 |
| 大 batch、大词表 | 实际触发上游请求分块，记录峰值显存和 logits 临时 buffer，确认 chunk 边界不影响结果 |
| 多 TP/DP/PP | 使用当前已经支持的并行组合，验证各 rank 计数和状态更新，不能由单卡结果推断支持 |
| 业务约束 | top-k/top-p、penalties、logprobs、structured output 按已有功能矩阵回归 |
| 真实模型质量 | 固定评测集和采样设置，分别记录 standard/block 的任务指标；先按项目现有误差阈值判定 |

性能比较使用相同模型、硬件、驱动、vLLM、输入长度、输出长度和并发。记录两组比较：

1. 修改前 standard 对比修改后 standard：检查统一 placeholder 处理和新增参数是否带来旧模式性能回退。
2. 修改后 standard 对比修改后 block：量化接受长度变化及新增 kernel 成本。

先完成编译和图捕获预热，再至少测 5 个独立计时轮次，记录中位数和波动范围。记录总吞吐、decode 吞吐或 TPOT、峰值 NPU 内存、验证阶段耗时、平均接受 draft 数。离线 smoke 的 tokens/s 包含 prefill 和调度，不能直接当作 decode TPOT；TPOT 使用现有在线压测流程测量。

可将旧模式中位吞吐下降超过 5% 设为初步排查线，但这只是本方案建议阈值，不是仓库既定门槛；最终以业务 SLA 和重复测量噪声为准。block 的性能门槛单独制定，不将“接受长度增加”视为必然加速。

计数口径：平均接受 draft 数 = accepted draft tokens / verification rounds；平均输出长度（包含恢复/bonus）在相应完整轮次口径下为前者 +1。不要混用 draft token 接受率、平均接受数和包含 bonus 的长度。

## 5. 结果记录与验收

保存以下内容到 `RESULT_DIR`：

- Ascend/vLLM commit、运行环境、模型版本、启动配置。
- pytest 原始日志和 JUnit XML；失败时的完整 traceback。
- 每组端到端配置、token IDs、metrics 增量、耗时及异常记录。
- 性能比较的每轮原始数据和 profiler trace。

| 验收项 | 结果 |
| --- | --- |
| 加载了待测源码及 NPU rejection_sample | 待执行 |
| block 接受长度测试通过 | 待执行 |
| 输出分布与 prefix residual 测试通过 | 待执行 |
| placeholder 24 组测试通过 | 待执行 |
| 两个算子测试文件完整回归通过，无非预期 skip | 待执行 |
| 端到端配置生效，实际执行新增 block kernel | 待执行 |
| greedy 一致性、随机模式稳定性通过 | 待执行 |
| 业务质量与性能符合约定门槛 | 待执行 |

第 2 节通过可证明算子层功能及主要分布语义生效；第 3 节通过才可确认真实 MRV2 调用链启用并运行；生产验收还需第 4 节中与部署相关的场景。
