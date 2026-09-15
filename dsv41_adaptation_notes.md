# DeepSeek-V4.1 (dsv41) 在 mrv2 上的适配记录

- 适配对象：DeepSeek-V4.1 (Aurora) 在 MRV2 (Model Runner V2) worker 路径上的部署运行
- 环境：远程主机 80.5.9.136，容器 `mrv2-dsv41`（Ascend NPU 16 chip，DP2 x TP8）
- 代码路径：`/mnt/share/l00960935/dev/mrv2-dsv41/vllm-ascend`（PYTHONPATH 优先加载）
- 启动脚本：`/mnt/share/l00960935/dev/mrv2-dsv41/mrv2-dsv41.sh`；日志目录 `logs/`
- 分支：`mrv2-dsv41`；本文覆盖 commit 范围 `cf01edfa6^..HEAD`（8 个 commit，2026-09-12 ~ 09-14）

---

## 1. 修改概述

V4.1 此前只能在 MRV1 上运行（`validate_cache_runtime` 对 `use_v2_model_runner` 直接抛
`NotImplementedError`）。本系列变更的总体目标是**让 V4.1 完整跑通 MRV2 worker 路径**，
覆盖四条主线：

1. **缓存体系接入 MRV2**：V4.1 的 layer-outermost 混合缓存（LongKV/Indexer/SWA/state/draft
   多 placement 共享槽位）接入 MRV2 的 KV cache 分配、reshape、绑定流程
2. **执行路径适配**：runtime-NONE 步骤（prefill、非均匀 decode）绕过编译 wrapper 强制 eager，
   仅均匀 decode 走 FULL ACL graph；dummy run 的 ring state 处理对齐 MRV1 语义
3. **元数据内核执行修复**：AICPU 元数据 kernel 在 MRV2 内联执行模式下的 stream/allocator
   适配与防重复发布
4. **算子内核修复**：`VllmQuantLightningIndexerMetadata` 的 `num_heads_q` 硬校验放宽，
   修复 TP 切分场景 AICPU errcode 22007

### Commit 时间线

| Commit | 时间 | 说明 | 变更量 |
| --- | --- | --- | --- |
| `cf01edfa6` try mrv2 | 09-12 11:59 | 主体：V4.1 接入 MRV2（缓存/元数据/dummy run/eager fallback） | 7 文件 +418/-5 |
| `4463f540d` test(deepseek-v41) | 09-12 14:32 | UT 对齐 PD(kv_transfer) 已启用现状 | 1 文件 +10/-3 |
| `7afeea8e8` try2 | 09-12 15:49 | dummy ring-state ContextVar 贯通；`prepare_source_rope` 拆分 | 4 文件 +158/-21 |
| `19932d5fd` try3 | 09-12 16:45 | AICPU 默认流路由（507018）；`_publish_task` 去重雏形 | 2 文件 +97/-1 |
| `d00d121aa` fix | 09-12 17:17 | UT 修正 | 1 文件 +18/-6 |
| `2dcc3c0d7` fix(aicpu) | 09-14 09:48 | `num_heads_q` 校验放宽（修复 22007） | 1 文件 +4/-3 |
| `40d50950a` fix(dsa_v41) | 09-14 09:49 | `_published_tasks` 改实例级注册表 | 1 文件 +11 |
| `4e484bd9f` chore | 09-14 09:49 | .gitignore 忽略 `.tmp_*` 调试临时文件 | 1 文件 +3 |

整体净变更（不含本文档）：10 个文件，约 +640/-30。

---

## 2. 详细变更说明（按功能模块分类）

### 2.1 缓存体系：V4.1 接入 MRV2 KV cache 流程

#### [vllm_ascend/core/deepseek_v41.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/core/deepseek_v41.py)（+2/-3）

- `validate_cache_runtime` 删除对 `use_v2_model_runner` 的拒绝。MRV2 的 V4.1
  cache 分配/reshape 经由 `vllm_ascend/worker/v2/attn_utils.py` 的 patch 路径执行，
  与 MRV1 共享同一套 `plan_cache_slots` / `allocate_cache_config` 槽位规划契约
- 说明：PD 分离（kv_transfer）的支持由范围外的 `9e9a0d622`（09-10 enable
  kv_transfer for dsv4.1）完成，`4463f540d` 仅将 UT 与该现状对齐

#### [vllm_ascend/worker/v2/attn_utils.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/v2/attn_utils.py)（净 +100）

1. **`_allocate_kv_cache` V4.1 专用分配路径**
   - 检测到 V4.1 spec 时拒绝混合布局（`Mixed V4.1 cache allocation is not supported`）
   - `plan_cache_slots()` 计算槽位，每槽一块 backing tensor，由该槽全部 placement
     （source / SWA 别名 / compressor state / draft）共享，各 placement 以不同
     live block ID 映射
   - 逐项校验 allocation 与槽位契约一致（offset=0、block_stride=page_size、
     size=num_blocks*page_size、shared_by 与 placement 名单一致）
   - 配置 kv_transfer（PD 分离）时追加 2MB 对齐（预填充分离需 cache 地址 2M 对齐）
2. **`_reshape_kv_cache_v2` V4.1 reshape 路径**
   - 按 `(offset, page_size_bytes)` placement 建立 `layer_placements` 映射，
     V4.1 层走 `reshape_cache()` 自身的 page-stride as_strided 视图，不进入
     通用 hybrid reshape 分支
3. **`build_attn_metadata` V4.1 元数据注入**
   - 新增 `full_graph_mode` 参数：V4.1 builder 依据运行时图模式选择
     graph-friendly 或 eager 元数据路径（由 `model_states/default.py` 传入
     `cudagraph_mode == CUDAGraphMode.FULL`）
   - 新增 `skip_ring_state_update` 参数（默认从 dummy-run 作用域 ContextVar 解析）
   - 识别 `DeepseekV41MetadataBuilder`，注入 `num_actual_reqs`、
     `common_v41_metadata`（**每个 KV cache group 独立一份**，保证同一框架组内
     source 的 LongKV 与 Indexer 复用同一 `[T, 2]` 映射，且绝不与 SWA 组别名）、
     `common_v41_batch_metadata`（**跨组共享**的 batch 级条目）；kwargs 同时
     流入 `build_for_cudagraph_capture`，FULL graph capture 亦可感知
   - 对齐 MRV1 `model_runner_v1` 的共享语义

#### [vllm_ascend/patch/worker/patch_bind_kv_cache.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/patch/worker/patch_bind_kv_cache.py)（+11）

- `bind_kv_cache` 增加 V4.1 早退分支：V4.1 cache 资源是 nn.Module，按
  `kv_cache[0]` 索引存储，因此绑定值必须包一层 list；按层名排序保证共享槽位
  上确定性绑定顺序；否则原逻辑会对 V4.1 抛 NotImplementedError

#### [vllm_ascend/worker/v2/model_states/default.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/v2/model_states/default.py)（+2）

- `AscendModelState` 构建 attn metadata 时传入
  `full_graph_mode=(cudagraph_mode == CUDAGraphMode.FULL)`

### 2.2 执行路径：eager fallback 与 dummy run 适配

#### [vllm_ascend/worker/v2/model_runner.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/v2/model_runner.py)（净 +142）

1. **`_install_v41_eager_fallback()`（`__init__` 中安装，进程内仅一次）**
   - V4.1 的 Python 参考实现 compressor/indexer 路径仅在 eager 下保证正确性，
     FULL_DECODE_ONLY 下 prefill 与不支持的 decode 形态会以 runtime NONE 派发
   - 上游 `skip_compiled` 只覆盖 encoder-decoder 步骤，因此 patch 模块级
     `vllm_model_runner.set_forward_context`：当 `cudagraph_runtime_mode==NONE`
     且模型为 V4.1（`model_type` 属于 `deepseek_v4.1/deepseek_v41` 及 text 变体）
     时强制 `skip_compiled=True`
   - 每次调用都检查 model type，同进程内非 V4.1 runner 不受影响
2. **`prepare_dummy_attn()` 重写**
   - V4.1 compressor ring state 每请求独占一个私有 page；上游 dummy 零填充
     block table 会把所有 dummy 请求别名到 page 0
   - 修正：为 circular spec 的 cache group 分配独立 live state ID `1..num_reqs`
     并清零对应 ring pages，保证 graph capture/replay 看到干净的 ring
   - `ring_state_update_skipped()` 为真时整体跳过（与 MRV1 dummy 语义一致）
3. **`_dummy_run()` 接线**
   - 上游会丢弃 runner 特有 kwargs（如 `skip_gdn_state_update`），改为
     `kwargs.pop` 后经 `skip_ring_state_update` ContextVar 传递给
     `build_attn_metadata` 与 `prepare_dummy_attn`
4. **`initialize_kv_cache()` 扩展**
   - 检测 circular spec：对所有 `ratio == 2` 的 `DeepseekV41Compressor` 调用
     `prepare_ring_compressor(max_num_tokens, device)`（持久 buffer 校验 +
     Triton core 解析），须在任何 graph capture 之前完成
   - 调用 `_prepare_v41_source_rope()`：遍历全部 attn groups 的 metadata
     builders，对 V4.1 builder 执行 RoPE 表初始化（见 2.3）

### 2.3 V4.1 元数据 Builder：[vllm_ascend/attention/dsa_v41.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py)（净 +75）

1. **`prepare_source_rope()`（自 `enable_device_metadata` 拆出，`7afeea8e8`）**
   - 校验全部 ratio-2 source 层共享同一 RoPE 表（data_ptr 一致性）并缓存到
     `_c2_full_source_rope`
   - MRV1 经 `enable_device_metadata`（异步任务模式）初始化；MRV2 元数据任务
     是同步内联执行，只能在首次 `build()` 前由 runner 直接调用，故拆分独立入口
2. **`_run_task_on_registered_stream()`（`19932d5fd`）**
   - 问题：AICPU kernel 通过当前 stream 注册的 CANN allocator
     （`aclrtAllocatorGetByStream`）分配输出；vLLM 为 graph capture 创建的
     stream（FULL decode / PIECEWISE / draft graph）从未注册 allocator，
     在其上提交会**静默失败**，中毒的 task 在下一次 synchronize 时级联abort
     整个 context（507018）
   - MRV1 的 DeviceMetadataExecutor 始终在 worker 专用 stream 上执行故不触发；
     MRV2 内联执行，因此在任何非默认 stream 上调用时：先 `torch.npu.synchronize()`
     + `wait_stream` 围栏，切默认 stream 执行，再切回并 `wait_stream` 排序
   - 附 `warning_once` 日志便于观测
3. **`_publish_task` 防重复发布（`19932d5fd` 字典级 → `40d50950a` 实例级）**
   - 元数据 kernel 必须每步重跑（编码了当步 batch 坐标），去重只应作用于
     单个 `build()` 周期内对同一 key 的重复发布
   - 原实现只缓存调用方传入的共享字典，共享字典被更换后同一 task 被重复
     发布并重复执行内核（单测 `test_v41_publish_task_runs_inline_on_default_stream`
     捕获）
   - 最终方案：实例级 `_published_tasks: dict[str, torch.Tensor]`，
     `__init__` 初始化、`build()` 开头重置；同 key 同 buffer 的重复发布
     直接复用已发布 buffer，不重跑内核
4. **执行分支**
   - `_device_metadata_enabled`（MRV1 异步）：追加 DeviceMetadataTask
   - `_supports_device_ops`（MRV2 同步 + 设备算子）：`_run_task_on_registered_stream`
   - 其余：直接内联 `run()`

### 2.4 AICPU 算子内核：22007 修复

[csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp](file:///d:/code/workspace/vllm-ascend/csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp)（+4/-3，`2dcc3c0d7`）

- 原校验 `numHeadsQ_ != 64` 直接拒绝；TP 切分下（deepseek v4.1 dspark draft，
  TP2 时 `index_n_heads=32`）heads 数不再是 64，dummy run 即触发
  AICPU errcode 22007（retCode 0x2a）
- 放宽为 `numHeadsQ_ <= 0` 拒绝，错误信息同步更新，并注释说明 TP 切分场景

### 2.5 测试

**[tests/ut/worker/test_attn_utils_v2.py](file:///d:/code/workspace/vllm-ascend/tests/ut/worker/test_attn_utils_v2.py)**（新建，净 +371，20/20 通过）

| 测试 | 覆盖点 |
| --- | --- |
| `test_build_attn_metadata_injects_v41_shared_dicts_across_groups` | V4.1 共享字典注入、组内共享/跨组隔离 |
| `test_build_attn_metadata_resolves_skip_ring_from_dummy_run_scope` | ContextVar 从 dummy run 作用域解析 |
| `test_v41_prepare_source_rope_initializes_cache_without_async_tasks` | RoPE 初始化拆分后同步路径可用 |
| `test_v41_publish_task_runs_inline_on_default_stream` | 默认流路由 + 单周期内防重复发布 |
| `test_mrv2_initializes_v41_cache_layers_end_to_end` | MRV2 端到端初始化 V4.1 cache 层 |
| `test_allocate_kv_cache_v41_rejects_mixed_specs` | 混合布局拒绝 |
| `test_v41_eager_fallback_forces_skip_compiled_for_runtime_none` | runtime-NONE 强制 skip_compiled |

**[tests/ut/models/test_deepseek_v41_cache.py](file:///d:/code/workspace/vllm-ascend/tests/ut/models/test_deepseek_v41_cache.py)**（+22/-10）

- `cf01edfa6`：unsupported 特性列表移除 "v2"，新增
  `test_v2_model_runner_runtime_is_supported`
- `4463f540d`：移除过期的 "pd" 拒绝预期（kv_transfer 已于 `9e9a0d622` 启用），
  新增 `test_kv_transfer_runtime_is_supported`

### 2.6 杂项

- [.gitignore](file:///d:/code/workspace/vllm-ascend/.gitignore)：忽略 `.tmp_*` 一次性调试/测试脚本（`4e484bd9f`）

---

## 3. 功能影响分析

| 能力 | 变更前 | 变更后 |
| --- | --- | --- |
| V4.1 + MRV2 | 直接 NotImplementedError | 完整支持（缓存分配/reshape/绑定/元数据/dummy run） |
| V4.1 + PD 分离 (kv_transfer) | 已启用但 UT 预期过期（UT 红） | UT 对齐；MRV2 分配路径含 2MB 对齐支持 |
| V4.1 图模式 | 仅 MRV1 语义 | MRV2：仅均匀 decode 走 FULL ACL graph，其余强制 eager（编译 wrapper 绕过） |
| TP 切分 draft（dspark TP2） | QLI metadata 算子 22007 崩溃 | heads 校验放宽，正常运行 |
| MRV2 graph capture 下的元数据 kernel | 未注册 stream 静默失败 → 507018 级联 | 统一路由默认 stream + 围栏 |
| 元数据任务发布 | 共享字典更换后重复执行 | 单 build 周期实例级去重 |

**对外行为**：无用户侧接口变化；服务启动参数（DP2 x TP8 + dspark5 +
FULL_DECODE_ONLY + enforce-eager）下 MRV2 路径功能与 MRV1 对齐。

---

## 4. 关键问题与解决记录

### 4.1 AICPU errcode 22007（`VllmQuantLightningIndexerMetadata`）

- **现象**：dummy run 阶段 AICPU 异常，设备 6/10（两 DP 的同一 TP rank）报错
- **定位**：在 `dsa_v1.py::_build_qli_metadata` 加临时探针打印 QLI 入参 +
  调用栈，证实 TP2 下实际传入 `num_heads_q=32`，而内核硬校验 `== 64`
- **修复**：内核校验放宽为 `>0`（见 2.4）
- **验证**：重编 custom_opp 后 22007 归零，curl 200 正常生成

### 4.2 编译后 so 未生效

- ninja 判定无需重链，修改内核后旧 so 仍在用。手动删除旧 so 再编译生效
- **经验**：改 AICPU 内核后若行为未变，先对比 so 构建时间戳与源码修改时间，
  必要时删 so 触发完整重链

### 4.3 MRV2 内联元数据任务在 graph capture stream 上失败（507018 级联）

- AICPU 输出分配依赖 stream 注册的 allocator；capture stream 未注册 →
  提交期静默失败 → 同步时 abort 整个 context
- **修复**：`_run_task_on_registered_stream` 路由默认流（见 2.3.2）

### 4.4 dummy run ring state 别名

- 上游 dummy block table 零填充使全部 dummy 请求共用 page 0，污染 ring state
- **修复**：`prepare_dummy_attn` 分配独立 state ID 并清零 ring pages（见 2.2.2）

### 4.5 元数据任务重复发布（UT 回归）

- `test_v41_publish_task_runs_inline_on_default_stream` 断言 `len(executed_on)==2`
  失败（第三次执行）：共享字典更换导致缓存失效
- **修复**：实例级 `_published_tasks`，`build()` 重置（见 2.3.3）

### 4.6 调试探针清理

- 定位完成后已移除 `dsa_v1.py` 中全部探针（`_vllm_logger` 导入与
  QLI_META_DEBUG 日志），文件与 git HEAD 一致，已同步容器（md5 一致）并重启验证

---

## 5. 关键适配点总结

1. **槽位规划契约共享**：MRV2 与 MRV1 共用 `plan_cache_slots` /
   `allocate_cache_config`，V4.1 的 layer-outermost 分配在 MRV2 侧按槽位契约
   逐项校验，两个 runner 行为一致
2. **组内共享、跨组隔离**：V4.1 slot 坐标（LongKV/Indexer `[T,2]` 映射）仅在
   单个框架 KV cache group 内共享，batch 级条目跨组共享——与 MRV1 语义对齐，
   防止 SWA 组别名
3. **graph 模式二分**：V4.1 只有均匀 decode 具备 FULL graph 条件；其余步骤
   必须绕过编译 wrapper。patch 点选在 `set_forward_context`（模块级、按 model
   type 过滤），对非 V4.1 零侵入
4. **kwargs 丢失绕行**：上游 `_dummy_run` 丢弃 runner 特有 kwargs，跨层标志
   统一走 ContextVar（`skip_ring_state_update`），与既有
   `override_mrv2_in_profile_run` 模式一致
5. **AICPU allocator 与 stream 的隐式耦合**：任何在自建 stream 上提交 AICPU
   kernel 的路径都必须确保 allocator 注册，或路由默认流——MRV1 专用执行流 /
   MRV2 默认流路由是同一问题的两种解
6. **内核参数校验应匹配模型派生值**：`num_heads_q` 等由 model config/TP 推导
   的参数不应硬编码校验；校验意图是防非法输入而非锁定单一模型形态

---

## 6. 验证结果汇总（2026-09-12 ~ 09-14）

| 项目 | 结果 |
| --- | --- |
| 服务启动（DP2 x TP8 + dspark5，MRV2） | 成功，无致命错误 |
| curl /v1/chat/completions | HTTP 200，正常生成（max_completion_tokens=10 截断符合预期） |
| /v1/models | HTTP 200 |
| 启动日志算子错误 | 22007 / AICPU / 507018 计数 0 |
| tests/ut/worker/test_attn_utils_v2.py | 20/20 通过 |
| vendor custom_opp 一致性 | `libtransformer_aicpu_kernels.so`（5449544 字节，2026-09-12 13:09:44 构建）与代码一致 |

## 7. 遗留事项

1. `prepare_source_rope`、`prepare_ring_compressor` 等 runner 驱动的初始化
   契约建议后续向上游收敛（当前为 patch/子类内实现）
2. V4.1 prefix caching 仍未实现（`build()` 显式 raise），按需排期
3. `.tmp_*` 调试脚本已保留本地并被 gitignore；容器 workdir 下另有
   `repro_qli_metadata.py`、`apitest.sh`、`perftest.sh`、`test_api.sh`、
   `req_test_clean.json` 等非仓库文件
4. 本分支领先 `origin/mrv2-dsv41` 3 个 commit（`2dcc3c0d7`、`40d50950a`、
   `4e484bd9f`），尚未推送

## 8. 复现/回归命令

```bash
# 启动（容器内）
bash /mnt/share/l00960935/dev/mrv2-dsv41/mrv2-dsv41.sh

# 功能测试（容器内）
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/mnt/share/l00960935/dev/mrv2-dsv41/req_test_clean.json

# 单元测试
pytest -sv tests/ut/worker/test_attn_utils_v2.py
pytest -sv tests/ut/models/test_deepseek_v41_cache.py

# 错误检查
grep -aE "22007|AiCPU error|507018" /mnt/share/l00960935/dev/mrv2-dsv41/logs/<最新日志>
```

---

## 9. num_reqs / num_tokens 维度约定对照表（上游 / MRV1 / MRV2 / V4.1 builder）

> 近期三个坑（is_prefilling 切片 IndexError、上游 PR #55458 draft query graph
> token 计数、dcp draft 元数据切片）本质都是同一类问题：**四处对
> request 维 / token 维的"真实值 vs padding 值"约定不一致**。本节作为后续
> 排查的速查表。核实基于当前分支代码。

### 9.1 上游 vLLM（MRV2 继承的基类约定）

| 维度 | 约定 |
| --- | --- |
| `CommonAttentionMetadata.num_reqs` | **真实请求数**，不随图 padding 扩展 |
| `num_actual_tokens` | **真实 token 数**（不含 FULL 图 padding） |
| `query_start_loc` | 长度 `num_reqs+1`，末元素 = 真实 token 数；**不插入 dummy request** |
| FULL/DECODE_ONLY 图 padding | 只 pad token 形状张量（input_ids/positions/slot_mapping）到 `batch_desc.num_tokens`；request 形状张量（seq_lens/block_tables）保持真实 num_reqs |
| `is_prefilling` / `query_start_loc_cpu` / `num_reqs_after_padding` | **上游不存在**，均为 Ascend 扩展字段 |

vllm-ascend 对此的偏离点：`AscendInputBuffers` 把 query_start_loc 缓冲区从
`max_num_reqs+1` 扩到 `max_num_reqs+2`
（[input_batch.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/v2/input_batch.py#L31-L39)），
因为 TND 布局要求 `hidden_states` 首维 == `query_start_loc[-1]`，而上游末元素
是真实 token 数、与 padded hidden_states 不匹配——dummy request padding 是
**Ascend 专属**（代码注释 "only required for vllm-ascend"）。

### 9.2 四处约定对照

| 字段 / 概念 | 上游 | MRV1（[model_runner_v1.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L3278-L3285)） | MRV2（[v2/model_runner.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/v2/model_runner.py#L539-L545)） | V4.1 builder（[dsa_v41.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py#L827-L830)） |
| --- | --- | --- | --- | --- |
| metadata `num_reqs` | 真实 | **padded**（`num_reqs=num_reqs_padded`，混合批次含 dummy） | FULL 模式 = `num_reqs_after_padding`（混合批次含 dummy = 真实+1；均匀 decode = graph capture size）；piecewise/eager = 真实 | 入口取 `common.num_reqs`，**不假设语义**，用 `num_actual_reqs` 界定真实前缀 |
| `num_actual_reqs` | 无此字段 | 无（padded 即语义） | 真实请求数（`input_batch.num_reqs`） | `min(common.num_actual_reqs, num_reqs)`，padding 行掩码用 |
| token 数（padded） | `num_tokens_padded`（token 形状 pad 用） | `num_input_tokens=num_tokens_padded` | `num_input_tokens = num_tokens_after_padding`（PCP eager 也 pad） | token 形状切 `[:num_input_tokens]`，真实前缀用 `num_actual_tokens` 界定 |
| token 数（真实） | `num_actual_tokens` | 无独立字段（= `num_actual_tokens` 传入） | `num_actual_tokens`（真实） | `valid_end = query_start_loc[num_actual_reqs].clamp_max(num_actual_tokens)` |
| `query_start_loc` | 真实 num_reqs+1，末元素 = 真实 tokens | **padded**：`[:num_reqs_padded+1]`，混合批次插 dummy request，末元素 = padded tokens | **padded**：`_pad_query_start_loc_for_fia` 后长度 = padded num_reqs+1（混合批次 dummy 行吸收全部 padding token） | 按 padded `num_reqs+1` 切，配合 `num_actual_reqs` |
| `is_prefilling` | 无此字段 | **长度 = num_reqs_padded**，padding 行显式置 False（与 query_start_loc 天然对齐） | **长度 = 真实请求数**（按 `idx_mapping_np` 逐请求构造，不随 padding 扩展） | ⚠️ 唯一"反 padding"字段：`_request_counts` 按 `min(num_reqs, is_prefilling.shape[0])` 取界，dummy/padding 行归入 decode |
| `seq_lens`（request 形状） | 真实 num_reqs | `[:num_reqs_padded]` | `[:num_reqs_padded]`，padding 行由 `seq_lens_np[num_reqs_padded:]=0` 清零 | `[:num_reqs]` 后对 `seq_lens[num_actual_reqs:num_reqs]` 显式 zero_() |

### 9.3 核心矛盾与排查要点

**核心矛盾**：MRV1 把 `is_prefilling` 一起 pad（长度恒等于 metadata
`num_reqs`），MRV2 继承上游只按真实请求构造 `is_prefilling`，而 request 形状
张量（`query_start_loc_cpu`）却是 padded 的——**同一个 batch 里两种长度并存**
。任何把 `is_prefilling` 直接按 `num_reqs`（padded）切分的代码，在
MRV2 FULL 图混合批次下必然 IndexError。

排查 checklist（按风险从高到低）：

1. **CPU 侧逐请求切片**：凡 `is_prefilling[:num_reqs]`、
   `is_prefilling[:num_reqs_after_padding]` 的写法都要警惕，改用
   `min(num_reqs, is_prefilling.shape[0])` 取界（参考
   [dsa_v41.py `_request_counts`](file:///d:/code/workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py#L188-L210)、
   [dcp_utils.py](file:///d:/code/workspace/vllm-ascend/vllm_ascend/worker/dcp_utils.py#L508-L523) 的防御写法）
2. **MRV1↔MRV2 共用的 metadata 消费端**：MRV1 下能跑通不代表 MRV2 正确——
   MRV1 的 `is_prefilling` 长度兜底了越界，MRV2 会暴露
3. **dummy request 的归属**：混合批次 FULL 图下 `num_reqs_after_padding`
   含 1 个 dummy 行（query_len = 全部 padding token），派生计数
   （num_decodes/num_prefills 等）必须决定 dummy 算 decode 还是剔除
4. **均匀 decode 的 padded 行**：非混合批次下 padded 行 query_len 按
   `decode_query_len` 步进填充，同为 decode 语义，通常无害；但逐请求
   CPU 循环若遍历到 padded 行需跳过
5. **图模式边界**：piecewise/eager 下 MRV2 `num_reqs` 回归真实值，同一
   代码在两种模式下的 `num_reqs` 语义不同，条件分支要覆盖两种取值
