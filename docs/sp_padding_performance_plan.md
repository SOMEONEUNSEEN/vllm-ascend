# Sequence parallel：pad 与 cat 性能验证及工作记录

## 目标与改动范围

- 日期：2026-10-12。
- 起点：本地 main `e00cb64a4469dfcd1ea8ba2d5d5e4eb2ab7bcaf1`，已包含合入提交 `51e47bbb77a2a8561ab2a4a1e236b01befda1cab`。
- 分支：`codex/sp-restore-cat`。
- 背景：[PR #16899](https://github.com/vllm-project/vllm-ascend/pull/16899)。此次核对以本地合入提交的实际 diff 为依据。
- `_ascend_sp_shard_impl`：将 `F.pad` 恢复为 `new_zeros + torch.cat(dim=0)`，补齐形状保留所有非 token 维度，包括 HC 维度。
- `_ascend_sp_padding_mask_impl`：将 `F.pad(value=True)` 恢复为 `new_ones + torch.cat(dim=0)`。
- `_ascend_sp_reduce_scatter_impl` 已使用 cat，无需修改。
- 保留 padding guard、dim 0 分片、custom op/fake 注册、无 padding 时的 clone，以及 collective 选择逻辑。没有新增环境变量。

当前恢复 cat 是用户要求的实现选择；**尚无 NPU 测量结果证明 cat 更快**。

## 公平比较的对象

比较完整表达式 `F.pad` 和 `new_zeros/new_ones + cat`。cat 的尾部张量分配、初始化必须在计时范围内；不能提前分配尾部而仍称其为本次实现。

分三个层次验证：

1. **单卡表达式**：仅补齐，以及补齐加分片两种 scope。独立脚本不加载模型或发起通信，快速识别 shape、dtype、stride 的影响。
2. **实际 custom op 与图执行**：使用同一套 vLLM/CANN 环境，分别运行本分支起点与本次生产代码改动，经过真实 custom op dispatcher 和模型编译/ACL graph 路径。确认图捕获、动态 token bucket、非别名约束均正常。
3. **完整模型**：对相同请求回放，比较 MRV2 下 DeepSeek V4.1 的吞吐、TTFT、TPOT、端到端延迟和显存峰值。分别覆盖 eager 与实际部署图模式，并记录实际进入两个 padding 分支的次数和 shape 分布。

仅微基准更快不足以说明模型整体更快。对齐长度可能完全绕过 padding，因此需要记录真实命中率。

## 正确性先行

新增 UT 比较 cat 实现与原 pad 参考，覆盖：

- 2D、3D hidden states；float32、float16、bfloat16；1D bool mask。
- 连续和非连续输入；全部 TP rank。
- 已对齐、未对齐、token 数小于 TP、TP=1、空输入。
- 精确输出、dtype/device、输入不变；修改返回值验证不与输入别名。
- 保留已有 reduce-scatter custom collective/fallback 回归测试。

在具备项目依赖的 Linux 环境运行：

```bash
pytest -sv tests/ut/models/test_sequence_parallel.py
```

NPU 上再运行相同边界用例，经公开 custom op 包装器检查输出和图捕获；CPU UT 直接调用实现，不能替代 dispatcher、编译及 NPU 验证。图回放需在捕获后更换静态输入缓冲区内容并核对输出，避免只检验首次捕获结果。

## 微基准使用

脚本：`tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_sp_padding.py`。这是手动基准，不作为普通 pytest 测试自动运行。需要真实 NPU、torch、torch_npu，不要求模型权重或 TP 通信组。

```bash
BENCH=tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_sp_padding.py
python "$BENCH" --shape 129 7168 --dtype bfloat16 --tp-size 8 --scope padding --output hidden-padding.json
python "$BENCH" --shape 129 4 7168 --dtype bfloat16 --tp-size 8 --rank 7 --scope shard --output hidden-hc-shard.json
python "$BENCH" --shape 129 --dtype bool --tp-size 8 --rank 7 --scope shard --output mask-shard.json
python "$BENCH" --shape 128 7168 --dtype bfloat16 --tp-size 8 --scope shard --output aligned-control.json
```

以上 hidden/HC 维度是初始探针，最终必须用实际模型配置及 profiler 观察到的形状替换。输出路径的父目录需存在；脚本拒绝覆盖已有 JSON。

建议矩阵：

| 维度 | 测试取值 |
| --- | --- |
| 设备 | 目标 Ascend 910B/910C，分别报告，不混合统计 |
| token 数 N | 0、1、TP−1、TP、TP+1；128、512、2048、8192 附近的对齐值及 ±1 |
| padding 余数 | 小 shape 对每个 `N % TP` 枚举，覆盖补 1 至 TP−1 行 |
| TP | 1、2、4、8，以及实际部署值 |
| hidden shape | `[N,H]`、`[N,HC,H]`；实际 H/HC 优先 |
| dtype | FP16、BF16；FP32 对照；mask 使用 bool |
| 布局 | 连续、`--noncontiguous`，以及实际 profiler 发现的 stride |
| scope/rank | padding、shard；rank 0 与末尾 rank，正确性验证全部 rank |

先跑小矩阵确认可用及显存容量，再扩展；空张量和 TP=1 属于正确性/分支控制组。`scope=padding` 且 padding=0 时两者均直接返回输入，不能用这一组推断算子快慢。

## 计时与统计协议

- 固定设备、功耗/频率策略、软件版本及线程配置；避免同卡其他任务。保存 `npu-smi info`、驱动/CANN、torch/torch_npu、vLLM/Ascend 提交、模型版本和完整命令。
- 输入生成、精确正确性检查均在计时之外。默认每个 variant 预热 100 次，每批 500 次，20 个配对轮次，固定 seed 随机决定每轮先后顺序。
- 脚本每批前后 `torch.npu.synchronize()`，用单调墙钟除以迭代数。指标包含 host dispatch、分配、实际设备工作和批末等待，**不是纯 kernel 时间**。不在每次调用后同步，也不在计时区打印或 `.item()`。
- JSON 保存每轮原始微秒/次数据、顺序、版本、shape/stride、padding，以及中位数。`pad/cat > 1` 表示此项测量中 cat 更快。
- 每组至少启动 5 个独立进程重复，变更 seed；保留各轮差值与比值，报告中位数、IQR 及以独立进程为重采样单位的配对 bootstrap 95% 置信区间。批均值的 p95 不能冒充单请求 p95。
- 若波动与差异相当，或置信区间跨过 1，结论应为“未分辨出稳定差异”；微秒级差异需同时给出绝对值。不要把一个 shape 的结论推广到全部情况。
- 单独做 profiler 采样，比较 pad 与初始化+cat 的 kernel 数量、设备总时长、拷贝和分配；profiling 与主计时分开。NPU Event 的区间也可能含 host 提交空隙，不能直接当成 kernel 时长之和。

## 图执行与模型 A/B

微基准脚本目前只覆盖 eager 表达式。以下是后续 NPU 验证步骤，不是已实现或已通过的测试：

1. 建立两个干净运行目录：A 固定为上述起点，B 为仅加入本次两处生产改动的版本；其余配置、依赖、权重、并行度和随机种子完全一致。
2. 先确认两侧请求输出一致，并覆盖连续跨越 TP 对齐边界的 token bucket、prefill/decode、推测解码开启/关闭；记录可接受的数值比较标准及采样设置。
3. 在真实图路径中分别捕获 A/B，保留各自静态输入/输出及图内分配的生命周期。每次正式测量前完成编译、捕获和预热；捕获成本单独报告。多次 replay 后检查输出，特别检查 mask True 补齐和 HC 维的 token 轴。
4. 回放相同固定请求集，分小 batch decode、大 batch decode、长 prefill、混合负载；保持相同负载速率及并发数。交替 A/B 至少 5 轮，记录吞吐、TTFT/TPOT/端到端延迟 p50/p95/p99、显存峰值及失败率。
5. 通过独立 profiling 确认本次函数实际执行，按层数、分支命中次数和 shape 分布估算累计收益，再与端到端变化核对。若图内始终对齐，应明确报告该负载没有测到 pad/cat 分支。

接受条件：正确性、custom op 非别名约束及图回放均通过；目标负载上的差异可重复且大于测量噪声；端到端无明显退化。若不同形状各有胜负，先保留分场景结论，不据此立即引入新开关或运行时分派。

## 工作记录

1. 检查本地工作区：main 上没有 tracked 文件改动，但有多份既有 untracked 文件；本次保留这些文件。
2. 检查原合入 diff 与当前文件：只发现两处 F.pad，reduce-scatter 已经是 cat。
3. 从本地 main 创建新分支，完成两处恢复，移除未使用的 functional import。
4. 补充功能回归测试、独立配对微基准和本验证方案。
5. 初始环境检查：系统 Python 3.12 无 torch，尚不能运行张量 UT 或真实 NPU 测量。最终静态检查及运行结果见下方补记。

### 本次验证补记

- 三个 Python 文件的 `ruff check`、`ruff format --check` 均通过；`git diff --check` 通过。
- `python -m pytest -sv tests/ut/models/test_sequence_parallel.py` 已尝试，加载仓库 conftest 时因 `ModuleNotFoundError: No module named 'torch'` 失败，未执行测试用例。
- `bash format.sh ci` 已尝试，仓库当前 Windows checkout 中的脚本含 CRLF，Bash 在解析函数定义时失败，尚未进入 pre-commit。因此不能宣称完整格式检查通过；本次未为此修改全仓库行尾。
- NPU 微基准、custom op 图回放和模型 A/B 均未运行，JSON 性能结果尚不存在。脚本仅通过静态检查，首次 NPU 运行仍需完成上文正确性检查。
- 按用户后续要求，将代码与文档分开提交，均带 sign-off。代码提交为 `1624aa267`（实现、回归测试和微基准），本工作记录随独立文档提交保存；尚未推送或创建 PR。上述验证限制仍然存在，后续应在具备依赖和 NPU 的 Linux checkout 完成验证。
