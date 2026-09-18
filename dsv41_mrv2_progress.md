# DeepSeek V4.1 × MRV2 适配进度文档

> 分支：`main-mrv2-dsv41`（基于 vllm-ascend/main + PR #16544 集成）
> 代码仓库：本地 `D:\code\workspace\vllm-ascend` 与远端 `80.5.9.148:/mnt/share/l00960935/dev/main-mrv2-dsv41/vllm-ascend`（容器 `main-mrv2-dsv41`，服务端口 8900；IP 已于 09-16 晚从 136 迁移到 148）
> 更新时间：2026-09-18 凌晨（**P1→P6 全部完成**：全量特性 + FULL_DECODE_ONLY 图模式打通 + 64 并发压测 + gsm8k 87.5%；第 9 章为 P5/P6 最终结果）

---

## 1. 已完成修改清单

### Commit 1 — `acda01ac2` feat(worker/v2): wire V4.1 cache allocation and metadata into MRV2

| 文件 | 修改内容 |
|------|----------|
| `vllm_ascend/worker/v2/attn_utils.py` | ① `_allocate_kv_cache` 新增 V4.1 分支：layer-outermost 分配，每个 planned slot 一块 backing tensor，所有 placements（source/SWA 别名/state/draft）以不同 live block ID 共享；校验 `allocation.offset/block_stride/layers` 与 slot 规划一致。② `_reshape_kv_cache_v2` 新增 V4.1 分支：按 `(offset, page_size_bytes)` placement 做 `as_strided` 视图（`reshape_cache`）。③ `build_attn_metadata` 新增 V4.1 builder 分支：注入 `num_actual_reqs`、`skip_ring_state_update`、`common_v41_metadata`（组内共享，LongKV/Indexer 不跨 SWA 组别名）、`common_v41_batch_metadata`（跨组共享）、`full_graph_mode`。④ 新增 ContextVar 机制 `skip_ring_state_update` / `ring_state_update_skipped`（上游 `_dummy_run` 丢弃 runner 私有 kwargs，标志改走 ContextVar）。 |
| `vllm_ascend/worker/v2/model_runner.py` | ① `_install_v41_eager_fallback`：包装模块级 `set_forward_context`，V4.1 且 `cudagraph_runtime_mode=NONE` 的步骤（prefill/非均匀 decode）注入 `skip_compiled=True`，仅均匀 decode 走 FULL ACL graph。② `_prepare_v41_source_rope`：遍历 attn_groups 对所有 V4.1 builder 调 `prepare_source_rope()`（对齐 MRV1 在 `initialize_attn_backend` 中经 `enable_device_metadata` 的行为）。③ `prepare_dummy_attn` + `_prepare_v41_dummy_ring_state`：dummy 请求分配独占 ring 页（1..num_reqs）并清零，避免图捕获/回放时所有 dummy 请求别名到 page 0；`skip_gdn_state_update` 的 dummy 批完全跳过。④ `_dummy_run`：从 kwargs pop `skip_gdn_state_update` 后进入 ContextVar 作用域。⑤ circular spec 存在时，图捕获前调用 `prepare_ring_compressor`（持久 buffer 校验 + Triton 核解析）。 |
| `vllm_ascend/core/deepseek_v41.py` | `validate_cache_runtime` 移除 `use_v2_model_runner → NotImplementedError` 闸门（两个 runner 共享 `plan_cache_slots`/`allocate_cache_config` 的 slot 规划契约）。 |
| `vllm_ascend/patch/worker/patch_bind_kv_cache.py` | V4.1 `DeepseekV41CacheLayer` 的 kv_cache 绑定包装为单元素列表（layer 以 `kv_cache[0]` 索引），排序保证确定性绑定顺序。 |

### Commit 2 — `bd32724b8` feat(dsa_v41): split prepare_source_rope from enable_device_metadata

| 文件 | 修改内容 |
|------|----------|
| `vllm_ascend/attention/dsa_v41.py` | ① 新增 `prepare_source_rope()`：V4.1 source RoPE 表校验 + 缓存（ratio-2 多层必须共享同一 RoPE 表），**不翻转** `_device_metadata_enabled`。② `enable_device_metadata()` 收窄为仅翻转异步开关 + 调用 prepare_source_rope。③ 埋点：两 hook 进出均打印 `device_metadata_enabled` 与任务数，`prepare_source_rope` 内检测开关翻转并打 ERROR。 |
| `vllm_ascend/attention/context_parallel/dsa_v41_cp.py` | `_ReplicatedCacheMetadataBuilder.prepare_source_rope`：先转发内层 `_global_builder` 再调自身；`enable_device_metadata` 加转发日志。 |

### Commit 3 — `ed98001a1` fix(models/deepseek_v41): survive MRV2 metadata and forward-context gaps

| 文件 | 修改内容 |
|------|----------|
| `vllm_ascend/models/deepseek_v41/engram_hash.py` | `engram_history_metadata`：`block_table_cpu` 缺失时回退镜像 device 侧 `block_table`（MRV2 从不填充该 CPU 镜像；MRV1 在 model_runner_v1.py:3657 显式挂载）。镜像发生在 engram 自身已有的 eager 同步边界（`input_ids.cpu()` 等），无额外流水线代价。 |
| `vllm_ascend/models/deepseek_v41/model.py` | `prepare_engram` 增加 `not ring_state_update_skipped()` 守卫：MRV2 dummy 批会构建完整 attn metadata（与 MRV1 metadata-free dummy 不同），必须跳过 n-gram 历史更新，防止 dummy token 污染按请求索引的历史存储。 |
| `vllm_ascend/ops/fused_moe/force_eplb.py` | `get_force_eplb_topk`：`getattr(forward_context, "moe_comm_method", None)`；None 时返回原 `topk_ids`（原代码 `return None` —— 调用方 routed_experts.py:606 对 None 无处理，属潜在 bug，新行为同时修复该契约）。 |

### 验证性埋点结论（2026-09-16 09:45 启动日志）

- `prepare_source_rope enter/exit` 共 832 条（16 worker × builder 链），顺序符合预期：CP builder 转发 → inner global builder → CP 自身。
- 启动阶段 `enable_device_metadata` 0 次调用（符合预期：MRV2 任务内联，该 hook 仅异步交接时触发）。
- **0 条** `flipped device_metadata_enabled` ERROR —— 异步任务开关未被翻转，确认无回归。

---

## 2. 剩余任务状态与完成度【2026-09-16 历史快照，最新进展见第 7/8 章】

### 2026-09-16 端到端验证结果（12:20:35 启动 → 12:27:13 READY）

| 验证项 | 结果 |
|--------|------|
| 模型注册表加载（架构解析） | ✅ 0 错误（lazy-import 修复后） |
| 启动全程 Traceback | ✅ 0 |
| `prepare_source_rope` 埋点 | ✅ 832 次，顺序正确（CP → inner global → CP 自身） |
| `enable_device_metadata` | ✅ 0 次调用（MRV2 任务内联，符合预期） |
| 异步开关翻转检测（ERROR 守卫） | ✅ 0 次触发 —— 开关未被翻转 |
| curl 回归（短 prompt × 3） | ✅ 3× 200 OK，输出正常，`finish_reason=stop` |
| 长 prompt（压缩缓存非空路径） | ✅ 200 OK，结构化回答正常 |
| 崩溃后 NPU 显存清理 | ✅ 发现 `VLLM::Worker_DP` 进程需按 host PID 清理（setproctitle 改名，pkill -f vllm 匹配不到） |

### 新增修复（本轮验证中发现）

| Commit | 内容 |
|--------|------|
| `caab59081` | fix(models/deepseek_v41): lazy-import ring-state flag —— attn_utils 模块级代码依赖 worker 路径导入顺序，model.py 改为 prepare_engram 内延迟导入，修复注册表加载 AttributeError |
| `94c34c556` / 远端 `2f89d1f29` | fix(dsa_v41): 空压缩源（max_cache_seq_len==0，短 prompt 首 decode）返回零宽 [T,0] 选择时，buffer 行按 -1 填充（pad_sparse_indices 约定）而非 copy_，修复 aclnnInplaceCopy 161002 |

### 剩余工作

| 工作流 | 完成度 | 状态 |
|--------|--------|------|
| 环境搭建 + 分支集成 | 100% | 完成 |
| V4.1 × MRV2 代码适配 | 100% | 5 commit 已提交（本地/远端历史一致） |
| 服务启动链路 | 100% | READY，无错误 |
| 功能回归（短/长 prompt） | 100% | 短 prompt ×3 + 长 prompt ×1 全部 200 OK |
| 埋点顺序复核 | 100% | 启动段 + 请求段均确认开关未翻转 |
| 并发压测（8/16/64，FULL_DECODE_ONLY 需 ≥64 触发混合批） | 0% | 依赖 aisbench，下一步执行 |
| MRV1 基线性能对比 | 0% | 依赖压测 |
| 单元测试补齐（ContextVar / V4.1 分配分支 / engram 回退 / 零宽选择） | 0% | 可并行开发 |
| 适配文档迁移 + PR 描述 | 40% | 本文档 + 旧分支约定表待合并入库 |

**整体完成度：约 85%**（代码适配与功能验证完成，压测与文档收尾待做）

---

## 3. 挑战与解决方案

| # | 挑战 | 根因 | 解决方案 |
|---|------|------|----------|
| 1 | 启动即崩：`validate_cache_runtime` 拒绝 MRV2 | V4.1 cache 初始化曾硬性要求 MRV1 | 移除闸门，V4.1 分配/reshape 移植进 MRV2 worker 路径（Commit 1） |
| 2 | host OOM（error 207001，128B pinned 分配失败） | KV cache 初始化阶段瞬时 host 内存耗尽（暂态） | 确认无残留进程 + host 内存充足后重试即恢复；留意容器 `ulimit -l` 仅 64MB |
| 3 | READY 后首次请求崩全引擎：`Engram requires query_start_loc_cpu and block_table_cpu` | MRV2 不构建 `block_table_cpu`；崩溃点在 DP 对端 rank 的 `execute_dummy_batch`（MRV2 dummy 批带完整 metadata，与 MRV1 不同） | engram 回退镜像 device block_table + ContextVar 守卫 dummy 批历史更新（Commit 3） |
| 4 | `AttributeError: ForwardContext has no attribute moe_comm_method` | MRV2 构建上游 ForwardContext，无该 Ascend 扩展属性 | getattr 容错 + 透传 topk_ids（Commit 3） |
| 5 | `KVCacheTensor has no attribute shared_by` | 新 vllm 版本字段更名 `shared_by → layers` | 全量替换（Commit 1 内） |
| 6 | AICPU 22007：`vllm_quant_lightning_indexer_metadata` num_heads_q 硬编码 64 | TP 切分下 index_n_heads=32 | 内核校验改为仅要求为正数（前序 commit `2dcc3c0d7`，随分支携带） |
| 7 | 本地/远端双仓库行尾不一致 | scp 传输本地 CRLF 文件到 LF 仓库造成全量假差异 | 改用 `git format-patch` + `git am` 同步，两边 commit 历史完全一致 |
| 8 | PowerShell/ssh 引号吞噬导致 curl 400 | PS 解析 ssh 参数中的内嵌双引号 | payload 走文件 + scp，脚本化执行 |

---

## 4. 关键技术决策与理由

1. **`skip_gdn_state_update` 走 ContextVar 而非改调用链**：上游 `_dummy_run` 会丢弃 runner 私有 kwargs（与 `override_mrv2_in_profile_run` 同款问题），ContextVar 是对上游零侵入的等价方案，且同时覆盖 `build_attn_metadata` 与 `prepare_dummy_attn` 两个消费点。
2. **V4.1 eager fallback 打在模块级 `set_forward_context`**：upstream `skip_compiled` 只覆盖 encoder-decoder 场景；V4.1 的 Python 参考 compressor/indexer 路径 eager 下正确性安全，仅均匀 decode 具备 FULL graph 条件。包装器每次调用检查 model type，非 V4.1 runner 不受影响。
3. **`prepare_source_rope` / `enable_device_metadata` 分离**：MRV1 把 RoPE 初始化绑在 `enable_device_metadata`（异步交接点）内；MRV2 任务内联执行，RoPE 缓存必须在 builder 初始化期完成，而异步开关必须只能由交接路径翻转 —— 分离后两者职责单一，并以埋点 + 翻转检测 ERROR 双重保障。
4. **engram 回退镜像 device block_table 而非改造 MRV2 metadata**：MRV2 的 block table 生命周期由 StagedWriteTensor/UVA 管理，为 engram 单独维护 CPU 镜像会引入跨组件一致性负担；engram 每步本就同步 `input_ids/positions` 到 host，在同一边界顺带镜像 [num_reqs, blocks]（≤几十 KB）代价可忽略。
5. **force_eplb 返回 `topk_ids` 而非维持 `return None`**：调用方对 None 无处理（返回后 `topk_ids=None` 会向下传播崩溃），旧分支实为死代码/潜在 bug；透传是唯一使调用方契约成立的行为。
6. **保留 `full_graph_mode` 死参数**（经确认）：作为 graph/eager 元数据分路径的预留接口，已在 commit body 中说明当前无消费者。

---

## 5. 遗留风险

- `_prepare_v41_dummy_ring_state` 依赖上游 `BlockTables.input_block_tables` 属性名与 `static_forward_context` 的 kv_cache 列表结构（隐式契约），上游重构时需同步。
- `enable_cpu_binding + engram_ple_offload + FULL_DECODE_ONLY` 组合下的 host pinned 内存水位需在压测中持续观察（暂态 OOM 已复现一次）。
- 4 个 `.bak` 调试备份残留在远端工作区（untracked），评审前应清理。

---

## 6. 图模式适配（2026-09-16 ~ 09-17）——当前死锁分析与交接

> 本章为最新交接内容。eager mode 功能/精度已验证通过（见第 2 章），图模式（`cudagraph_mode: FULL_DECODE_ONLY`）适配推进到**首请求阶段被 DP 间 padding 不一致死锁卡住**，排查陷入瓶颈，以下是完整过程、证据与下一步建议。

### 6.1 图模式适配总体思路

启动参数：DP2×TP8×EP8（EP 域跨两个 DP 组共 16 rank），dspark 投机解码（draft=Aurora-W8A8-144，`enforce_eager=true`，5 token），主模型 `FULL_DECODE_ONLY`（均匀 decode 走 FULL 图，prefill/混合批走 eager fallback），同时开启 engram / DSA-CP / flashcomm1 / force_eplb / shared-expert-DP / SP（`replace_allreduce=True` 的 MC2 路径）。

适配原则：主模型前向在图捕获/回放路径下不能有任何 dynamo 无法处理的宿主侧分支（ContextVar、动态形状算术、eager-only 同步点），因此：

1. **Engram 输入外置**：图捕获时 dynamo 会追踪 `prepare_engram` 的 eager 分支并踩 `ContextVar.get()`（dynamo 不支持）。方案是不让该路径进入前向——通过 MRV2 `AscendModelState` 钩子（`prepare_dummy_inputs` 捕获期注入固定地址 addressing buffer、`prepare_inputs` 每个真实步前刷新 buffer），图内只读固定地址缓冲，engram 寻址全部移到图外。
2. **形状逻辑算子化**：SP 分片的 Python 形状算术（modulo padding / chunk 切分）会被 dynamo 按首次追踪的形状烘焙，后续捕获尺寸全部错位。方案是注册自定义算子把形状逻辑对 dynamo 隐藏，fake impl 用 cdiv 动态推导。
3. **元数据计数解耦 padding 约定**：MRV1 与 MRV2 的 `is_prefilling`/`query_start_loc_cpu` 长度约定不同（详见第 4 章/第 3 章），图模式混合批暴露了该错位，按 mask 自身长度切片即可同时兼容两者。

### 6.2 图捕获阶段已解决的问题链（按时间序）

| # | 问题 | 根因 | 修复 | 状态 |
|---|------|------|------|------|
| G1 | 图捕获挂死（首轮，log `mrv2-dsv41-20260916-150854.log`） | dynamo 追踪 `prepare_engram` eager 分支，`ContextVar.get()` 不支持 | `882e925d5`(远端)/`e525c0537`(本地)：AscendModelState 钩子注入 engram graph inputs，`prepare_engram` 增加 `attn_metadata` 显式参数 | ✅ 修复 |
| G2 | `AssertionError: Current vLLM config is not set` | 图参数更新时 `get_impl_cls()` 在无 vllm config 上下文调用 | `aclgraph_utils.py` 捕获/回放统一包在 `set_current_vllm_config` + `set_forward_context` 作用域内（分支基线已带，见 aclgraph_utils.py:161-167） | ✅ 修复 |
| G3 | `input_ids.numel() must equal x.size(0). input_ids.numel()=1, rows=24` | dynamo 烘焙 `sp_shard` 的 Python 形状算术，跨捕获尺寸错位分片 | `d0af3d753`：`ascend_sp_shard_impl`/`ascend_sp_reduce_scatter_impl`/`ascend_sp_padding_mask_impl` 自定义算子 + 动态 fake impl | ✅ 修复 |
| G4 | `Padding length should be <= 2x input dimension ... padding length 4, input of dimension 1` | 上游 `sequence_parallel_chunk_impl` 只支持 2D，1D input_ids/token_mask 崩 | G3 的 shard impl 支持任意维度 | ✅ 修复 |
| G5 | 首请求崩：`mask [7] vs indexed tensor [8]` IndexError | 混合批 FULL padding：`is_prefilling`（真实请求数）与 `query_start_loc_cpu`（padded+1）长度错位 | `fdd8612ee`：`_request_counts` 按 mask 自身长度切片 + UT（`tests/ut/attention/test_dsa_v41_request_counts.py`，覆盖 MRV1/MRV2 两种约定） | ✅ 修复 |
| G6 | **首请求挂死（无 crash、无 traceback）** | **SP 分片形状算术被 dynamo 烘焙，跨捕获尺寸错位分片 → DP0/DP1 padded token 数不一致（8 vs 16）→ MC2/A2A 不匹配死锁** | `c5bff3db4`：`ascend_sp_shard_impl` 等自定义算子（形状逻辑对 dynamo 透明） | ✅ 已修复（P4 实证） |

最终图捕获结果（V8 轮，log `mrv2-dsv41-20260917-092837.log`）：**16 rank 图捕获全部完成（454s，0.74 GiB，0 Traceback）**，`Application startup complete`，服务正常 READY——即图捕获阶段已全部打通，问题完全收敛在**运行期首个请求**。

### 6.3 当前卡点：首请求 DP 死锁（证据 + 分析）【✅ 已结案 2026-09-17 P4，见 7.5】

> **结案结论**：G6 死锁**不是**架构契约问题，根因是 **SP 分片形状算术被 dynamo 烘焙**——`sp_shard` 的纯 Python pad+chunk 按首次追踪形状固化，图回放到其它捕获 bucket 时各 DP 组分片行数错位，进入 MC2 的 padded token 数随之失配（DP0=8 vs DP1=16）。P4 中以自定义算子 `ascend_sp_shard_impl` 将形状逻辑对 dynamo 透明后死锁消失（`c5bff3db4`），下列"方向 A/B/C"分析不再需要，仅作历史记录保留。

**时间线**（09-17）：09:41:47 图捕获完成 → 09:43 READY → 用户手动发首请求 → ~09:49:59 起失速 → 09:50:59/09:51:59 周期性 shm_broadcast 60s 警告 → 服务永久挂死（进程全部存活，无 crash）。

**决定性证据**（V41DBG 埋点，log 尾部 101245-101250 行）：

```
# DP1 各 TP rank 最后输出（重复出现，padded=16，每 rank 2 token）：
(Worker_DP1_TP3_EP11) [V41DBG][mc2.prepare.in] hs=[2,5120] rl=[2,128] repl_ar=True tp=8 rank=3 padded=16 mc2_mask=(16,)
(Worker_DP1_TP4_EP12) [V41DBG][mc2.prepare.in] hs=[2,5120] ... padded=16 ...
# DP0 各 TP rank 最后输出（padded=8，每 rank 1 token）：
(Worker_DP0_TP4_EP4)  [V41DBG][mc2.prepare.in] hs=[1,5120] ... padded=8  mc2_mask=(8,)
(Worker_DP0_TP6_EP6)  [V41DBG][mc2.prepare.in] hs=[1,5120] ... padded=8  ...
```

**同一执行步**内，DP0 组按 8 token padding、DP1 组按 16 token padding 进入 `PrepareAndFinalizeWithMC2`（`repl_ar=True` 即 SP-MC2 路径）。而 `enable_expert_parallel` 下 EP 域跨两个 DP 组共 16 rank（进程名 `Worker_DP1_TP3_EP11` 即 EP=8+3），**MoE 集合通信（A2A/MC2）的 token 维度必须在 EP 域内一致**——8 vs 16 的不一致使 16 rank A2A 两阶段对不齐 → 部分 rank 在集合通信内自旋等待（宿主机观察 `do_sched_yield` 忙等，Worker 477692 futex），EngineCore 全部 idle 在 ppoll，形成永久死锁。

**内核栈佐证**（/proc/<pid>/stack，宿主机）：
- Worker 477728 / 481543：`do_sched_yield` 自旋（HCCL host 侧轮询特征，即在等集合通信完成）
- Worker 477692：`futex_wait`（等待同伴线程）
- EngineCore ×2 / APIServer ×2：ppoll / epoll idle（上游调度链正常，纯等 worker）

**为什么 eager mode 没暴露**：eager 下 MC2 输入是动态形状，每组内按各自 token 数调用（aclnn 侧逐次协商或 mask 对齐），MRV1 时代同配置也从未在 eager 出问题；图模式下 padding 尺寸被捕获图固化并且回放尺寸按各 DP 组**独立**对齐（DP0 撞 8 的捕获尺寸、DP1 撞 16 的），差异被固化放大。

**死胡同所在**：这不再是"修 bug"而是**架构契约问题**——MRV2 的 padding 决策（`num_tokens_after_padding` per-DP-group）与 V4.1 的 EP16 MoE 通信域（跨 DP 组）在图模式下冲突。修复需要二选一：
- **方向 A（推荐探索）**：图回放尺寸跨 DP 组统一——同一 step 全 EP 域 pad 到相同 token 数（如取 `max(各DP组)` 或统一 bucket），需要 MRV2 调度/dummy 批构造处感知跨 DP 的对齐约束（DP 组间可能需要一次轻量协商或利用现有的 DP coordinator 同步点）；
- **方向 B**：把 V4.1 MoE 通信域收窄回 DP 组内（TP8），但 EP 分片本身就是跨组 16 rank，等于改 EP 拓扑，代价大基本不可行；
- **方向 C（临时绕过）**：`cudagraph_mode` 退回 `PIECEWISE` 或关闭 FULL_DECODE_ONLY，先验证 A/B 假设（若 PIECEWISE 也挂，说明问题在 EP 契约本身而非 FULL 图）。

### 6.4 仓库状态与同步步骤（重要）【2026-09-17 深夜更新】

**本地与远端哈希链已完全统一**（基于 `94c34c556` 重置后重新 cherry-pick 的链，本地 `main-mrv2-dsv41` HEAD = 远端工作区 HEAD = `16653528d`）：

| 本地 commit（= 远端） | 内容 |
|-------------|------|
| `94c34c556` | 基准：eager 全特性验证通过（origin/main-mrv2-dsv41） |
| `c5bff3db4` | fix(sequence_parallel): SP 形状算术对 dynamo 透明（G3/G4/G6 根因修复） |
| `ba81407f9` | fix(sequence_parallel): 3D draft 输入按 token 轴分片（G4 draft 路径） |
| `87a76f84e` | fix(dsa_v41): request counters 按 is_prefilling 长度派生（G5） |
| `b7f7c05ce` | feat(model_states): AscendModelState 钩子注入 engram 图输入（G1/P5a） |
| `16653528d` | test(worker): 修复 dummy-attn 测试既有 mock 缺口 |

**同步机制（现行）**：本地 remote `r148`（ssh git 协议，scp 风格 URL `80.5.9.148:/mnt/share/l00960935/dev/main-mrv2-dsv41/vllm-ascend`），日常同步一条命令：`powershell .\.tmp_main_mrv2\sync-to-148.ps1`（push 临时分支 `sync-from-local` → 远端 `reset --hard` → 校验哈希）。**format-patch 流程已作废**；远端 git 2.33.0 的 `safe.directory` 必须用带 `.git` 后缀的精确路径（已配好）。远端工作区只作部署区，禁止直接改文件。

**归档分支**（旧调试链，勿混入）：本地 `archive/mrv2-dsv41-graph-r1`（含旧哈希 `fdd8612ee`/`d0af3d753`/`6613a0d77`/`879a9cba3`，V41DBG 埋点按需 cherry-pick）；远端 `archive/am-chain-882e925d5`。

**未提交的无关文件**（本地 untracked，未入库）：`vllm_ascend/worker/anomaly_monitor.py` + 其 UT（独立的 NaN/inf logits 监控功能，与本次适配无关）；`.tmp_*` 调试脚本（本目录，可清理）。

### 6.5 下一步排查建议（按优先级）【已过时：G6 结案，P4 全过，建议 1/2/3/5 自动作废，仅 4（MRV1 对照）留作性能对比参考】

1. **拿到 Python 栈确认卡点**：宿主机无外网装不了 py-spy（pip Errno 101 network unreachable）。需在能上外网的机器下载 aarch64 wheel（`py_spy-*-py3-none-manylinux*_aarch64.whl`）scp 上去安装，对 `docker top`/宿主机 worker PID dump。重点 dump 自旋中的 Worker（本轮为宿主机 477728/481543）与 futex 的 477692。
2. **直接验证 DP padding 不一致假设**：在 6.3 证据基础上复跑一次，挂死时对比两组最后 `mc2.prepare.in` 的 `padded` 值——若重现 8 vs 16 类不一致，即坐实。可在 `_EXTRA_CTX.padded_num_tokens` 产生处（V4.1 builder / MRV2 `num_tokens_after_padding`）加跨 DP 打印，确认各组的图回放尺寸选择逻辑。
3. **方向 C 快速二分**（改启动脚本即可，30 分钟量级）：
   - `cudagraph_mode: PIECEWISE` 重跑首请求 → 挂则说明与 FULL 图无关，EP 契约问题在 eager 图混合下也存在（可对比 MRV1 为何不挂）；
   - 去掉 `speculative-config`（dspark）重跑 → 排除投机解码 dummy/extend 步的参与；
   - 关 `enable_flashcomm1` 或 `enable_force_eplb` → 排除 MC2 变体路径。
4. **MRV1 对照实验**：同一 148 机器若还有 MRV1 基线环境，用相同 DP2×TP8×EP8+FULL_DECODE_ONLY 跑首请求，抓它的 per-DP padded 尺寸——确认 MRV1 是"跨 DP 统一 padding"还是"MC2 域不同"，直接指明方向 A 的实现位置。
5. **排查 dummy 批尺寸选择**：单请求只落在一个 DP 组，另一组跑 dummy；两组选择的回放尺寸 bucket 不同（8 vs 16）——检查 MRV2 dummy/execute_dummy_batch 的尺寸对齐逻辑与 `num_tokens_after_padding` 的来源（是否本应读跨 DP 全局值）。

### 6.6 操作注意事项（环境备忘）

- **服务**：容器 `main-mrv2-dsv41`，端口 **8900**（不是 8000），启动脚本 `/mnt/share/l00960935/dev/main-mrv2-dsv41/main-mrv2-dsv41.sh`（启动配置见 6.1），日志目录 `logs/`（最新 log 用 `ls -t logs/*.log | head -1`）。
- **重启服务**：必须 `pkill -f 'vllm serve'`（注意 worker 进程被 setproctitle 改名 `VLLM::Worker_DP*`，pkill vllm 匹配不到，需按 host PID 补刀），确认 NPU 显存释放后再拉起。
- **PowerShell/ssh 陷阱**（本轮反复踩）：① PS 双引号会展开 `$var` 和 `$(...)`，远程命令含 `$` 时整个命令用 PS 单引号包裹；② 远程命令内的双引号会被 PS 吃掉，多层嵌套时改写成单层或走脚本文件；③ 超过 ~30s 的 ssh 命令输出会丢失——长任务一律远端 `nohup ... > /tmp/xxx.out 2>&1 &` 后台跑，轮询结果文件；④ 本地写脚本 scp 上去比 heredoc 可靠（heredoc 里 `$`/引号同样会被 PS 吃掉）。
- **py-spy 抓栈脚本**：远端仓库根有 `hangdump.sh`（watch 最新 log，90s 无新日志即 dump 全部 worker 栈到 `/tmp/v9_hang_dump.txt`），但宿主机缺 py-spy，装好后即可复用。
- **本地 UT 运行**：本地 Windows 无 pytest；容器内跑（`docker exec main-mrv2-dsv41 python /mnt/.../vllm-ascend/.tmp_ut_runner.py` 风格）注意 vllm_ascend 导入链长（>1min），必须后台 + 结果文件方式取结果；CI 兜底。

---

## 7. 基准重置与图模式重启（2026-09-17 晚，当前状态）

> 应用户要求推倒重来：以 `94c34c556`（eager 全特性验证通过、精度达标的最后版本）为基准，先用**精简环境**（`tmp-mrv2-dsv41.sh`）打通最基础功能的图模式，再增量恢复特性。第 6 章的全量配置调试经验作为参考保留。

### 7.1 环境与同步机制（已就绪）

| 项 | 状态 |
|----|------|
| 本地仓库 | `D:\code\workspace\vllm-ascend`，分支 `main-mrv2-dsv41` 重置到 **`94c34c556`** |
| 远端仓库 | `/mnt/share/l00960935/dev/main-mrv2-dsv41/vllm-ascend`，同样 `94c34c556`（哈希已与本地完全统一） |
| 历史归档 | 本地 `archive/mrv2-dsv41-graph-r1`（=旧 f6a6388d5，含 engram 图输入/G3/G5/V41DBG 四提交）；远端 `archive/am-chain-882e925d5`（旧 git am 链）；远端工作区垃圾已备份 `../worktree-junk-backup-20260917.tar.gz` |
| 同步机制 | 本地添加 git remote `r148`（ssh git 协议直推，对象 LF 无 CRLF 问题）。日常同步：`powershell .\.tmp_main_mrv2\sync-to-148.ps1`（push 到临时分支 + 远端 reset --hard + 校验）。**远端工作区只作部署区，禁止直接改文件** |
| 坑 | 远端 git 2.33.0 为发行版回移植版，`safe.directory` 必须用带 `.git` 后缀的精确形式（已在 system+global 配置）；格式化 patch 流程作废 |
| 实验脚本 | `/mnt/share/l00960935/dev/main-mrv2-dsv41/tmp-mrv2-dsv41.sh`（本地副本 `.tmp_main_mrv2/tmp-mrv2-dsv41.sh`）。关闭：engram/DSA-CP/flashcomm1(SP)/EPLB/shared-expert-DP/multistream/dspark；保留：DP2×TP8×EP8、FULL_DECODE_ONLY、cpu_binding、fused MC2。PYTHONPATH 直指仓库工作区 |
| 服务 | 当前未运行（P5a 第三次尝试失败后已清理全部进程，等用户清理残留后重启；详见第 8 章）。另：同机有多个容器（`mrv2-dsv4`、`mrv2-dsv4-0907`、`vllm-ascend-dev-148`），与本任务无关但共享 host 内存/HCCL 端口段，勿动但需留意资源竞争 |
| 上游参考 | `.tmp_main_mrv2/upstream_ref/` 已下载 `gpu_model_runner.py` / `gpu_cudagraph_utils.py` / `gpu_dp_utils.py`（148 的 `/mnt/share/l00960935/dev/main-mrv2-dsv41/vllm` 检出） |

### 7.2 图模式 DP 死锁（G6）静态调研结论

1. **上游主模型路径本身是 DP 安全的**：`vllm/v1/worker/gpu/dp_utils.py::sync_cudagraph_and_dp_padding` 在 DP cpu_group 上 all_reduce `[6, dp_size]`（tokens/cg_mode/uniform/max_query_len/ubatch/reqs），取 `max(num_tokens)` 后全 rank 以同一 synced 值联合 dispatch → 两 DP 组拿到相同 `batch_desc`；**idle rank 的 dummy 批经 `_dummy_run → execute_model` 参与同一 all_reduce**（num_tokens=0）；任一 rank 要 eager 则全体 eager；eager 时 `num_tokens_across_dp` 为原始逐 rank 计数，Ascend 侧 `ascend_forward_context` 取 max 后 pad 到 tp 倍数——仍一致。
2. **Ascend 侧确凿缺陷（头号嫌疑）**：`vllm_ascend/worker/v2/aclgraph_utils.py::ModelAclGraphManager.run_fullgraph` 中 `num_tokens_across_dp = torch.full([dp_size], num_tokens)` **伪造** DP 张量（本地回放尺寸填充，注释自认"refer to sync_cudagraph_and_dp_padding to calculate"未完成）。仅当两组 desc 一致时无害；一旦任何环节脱钩（draft/SP 路径）即放大为 EP16 域 A2A 死锁。
3. **次级嫌疑（全量配置特有）**：① dspark speculator `build_draft_attn_metadatas` 用 **per-rank** `num_reqs_padded * num_query_per_req` 计算 draft token 数（其 propose 正确复用 target 的 dp_sync，但元数据尺寸独立计算）；② SP-MC2（`repl_ar=True`，V41DBG 挂死现场路径）`pad_and_split_input_ids` 依赖 `_EXTRA_CTX.padded_num_tokens`，该值若来自本地回退则各组不同。两者在精简环境中均已关闭。
4. **基线能力确认**：`prepare_engram` 被 `enable_engram` 门控（model.py:528 早退）→ 精简环境无需移植 G1（engram 图输入）修复；SP 关闭 → 无需 G3/G4（SP 自定义算子）；G5（`_request_counts` 长度切片）在混合批下仍可能触发，遇到时从归档移植。

### 7.3 P1 结果（2026-09-17 16:08-16:21，✅ 全部通过）

| 验证项 | 结果 |
|--------|------|
| 启动 | 16:08:32 → ~16:19 READY（~10.5 min），0 Traceback |
| 图捕获（FULL） | 7 批 ~2.5 min（全量配置 454s → 精简环境显著加快） |
| **首请求（旧死锁点）** | ✅ 200 OK，"1+1=? → 2"，10s（含预热），finish_reason=stop |
| **DP 一致性** | ✅ `run_fullgraph num_tokens=1` 同一步在 DP0+DP1 同尺寸触发（日志实证） |
| 短 prompt ×2 复测 | ✅ 各 1s（图回放加速明显），temp=0 输出确定 |
| 长 prompt（226 token） | ✅ 9s/200 token，推理连贯、三点总结正确（length 截断属预期） |
| 挂死征兆 | ✅ 无请求期 shm_broadcast 警告（4 条全部在捕获期，正常瞬态） |
| 进程 | 16 worker 存活，0 Traceback |

日志：`logs/mrv2-dsv41-20260917-160832.log`；冒烟结果：`p1_curl1.out` / `p1_curl2.out`（远端 dev 目录）。

**结论**：基准 `94c34c556` + 精简环境下图模式（FULL_DECODE_ONLY, DP2×TP8×EP8）**完整打通**——静态调研结论被实证：上游 `dispatch_cg_and_sync_dp` 主模型路径 DP 安全，全量配置的 G6 死锁来自 dspark/SP/engram 等特性交互（P3+ 逐个恢复定位）。另注：`run_fullgraph` 日志行是 `info_once`（每 DP 组仅首条），非回放计数。

### 7.3.1 P3 结果（2026-09-17 16:39-16:54，✅ 全部通过，dspark 未复现死锁）

| 验证项 | 结果 |
|--------|------|
| 启动（dspark on） | 16:39:39 → 16:53:18 READY（~13.5 min），0 Traceback |
| 图捕获 | **25 批**（P1 的 7 批 → spec decode 尺寸按 decode_query_len=6 向上取整产生更多 bucket），~6.5 min |
| **首请求** | ✅ 200 OK "2"，17s（含 draft 预热），finish_reason=stop |
| **spec verify FULL 回放** | ✅ `run_fullgraph num_tokens=6`（1+5 投机 token）DP0+DP1 同步触发 |
| 短 ×2 | ✅ 各 1s，输出与 P1 **逐字节一致**（temp=0 确定性保持，19 tokens） |
| 长 prompt | ✅ 3s/200 tokens（P1 无 dspark 为 9s → **3 倍加速**，接受率高） |
| 进程 | 0 Traceback 全程 |

日志：`logs/mrv2-dsv41-20260917-163939.log`；结果：`p3_curl1.out` / `p3_curl2.out`；启动脚本：远端 `p3-mrv2-dsv41.sh`（= tmp 脚本 + speculative-config）。

**关键结论**：dspark 单独开启**不触发** G6 死锁——`build_draft_attn_metadatas` 的 per-rank padding 嫌疑排除（draft rank-local + verify 步 target dp_sync 正确同步 + idle 组 dummy 批参与 all_reduce）。结合 V41DBG 挂死现场在 SP-MC2（`repl_ar=True`）路径的证据，**头号嫌疑转向 P4 的 flashcomm1/SP**。

### 7.4 分阶段打通计划（原 7.3；P1/P3/P4 ✅，P5a 🔄 进行中，详见第 8 章）

| 阶段 | 内容 | 需移植的归档修复 | 状态 |
|------|------|------------------|------|
| P1 | 基线 + tmp 脚本：图捕获 → 首请求 → curl 冒烟（短/长 prompt） | 预计无（遇 G5 移植 `fdd8612ee` 逻辑） | ✅ 完成（7.3） |
| P2 | V41DBG 风格埋点确认 DP 同步完整性 | — | ⊘ 并入 P4（G6 随 SP 修复结案，未单独执行） |
| P3 | 恢复 dspark | — | ✅ 完成（7.3.1，未复现死锁） |
| P4 | 恢复 flashcomm1/SP | `d0af3d753`（SP 自定义算子化） | ✅ 完成（8.1，G3/G4/G5 复现并修复，首请求 + 冒烟全过，G6 消失） |
| P5 | 恢复 engram（a）/ DSA-CP（b）/ EPLB + shared-expert-DP（c） | `879a9cba3`（engram 图输入钩子）+ ContextVar 修复 | 🔄 P5a 代码完成 + UT 全过，启动被 host pinned 内存问题卡住（8.2） |
| P6 | 全量配置回归 + 并发压测（≥64 触发混合批） | — | ⏳ 待 P5 |

### 7.4 遗留事项

- py-spy aarch64 wheel 仍未安装（宿主机无外网，需本地下载后 scp）——P2/P3 若需 Python 栈再装。
- 归档分支中的 V41DBG 埋点 commit（`6613a0d77`）按需 cherry-pick，勿整体 revert 到基线。

---

## 8. P4 完成 + P5a engram 卡点交接（2026-09-17 深夜，当前最新状态）

### 8.1 P4 结果：flashcomm1/SP + dspark + FULL 图模式 ✅ 全部通过

**过程**（19:48 最终轮，log `mrv2-dsv41-20260917-194810.log`）：17:03 首轮启动即在图捕获撞上 **G3 复现**（`input_ids.numel()=1, rows=24`，dynamo 烘焙 SP 形状）→ 移植修复 `c5bff3db4` 后 **G4 复现**（draft 3D 输入 `[T, hc_mult, H]` 被上游算子 pad 错维度 → 零行分片）→ 补 `ba81407f9`（token 轴分片）→ 首请求 **G5 复现**（`mask [7] vs indexed tensor [8]`，P4 场景下 is_prefilling/query_start_loc_cpu 长度错位）→ 补 `87a76f84e` → 19:48 轮完整通过。

**最终验证结果**：

| 验证项 | 结果 |
|--------|------|
| 启动（flashcomm1 + SP on） | 19:48 → READY，0 Traceback |
| 图捕获（FULL） | 24 批 × ~30s ≈ 12 min（SP 路径进图后捕获变重，属预期） |
| **首请求（G6 旧死锁场景）** | ✅ 200 OK，无挂死——**G6 随 SP 修复消失** |
| 短/长 prompt 冒烟 | ✅ 200 OK，推理正常 |
| FlashComm1 | ✅ 日志确认 enabled |

**结论**：`94c34c556` + SP/dynamo 修复（`c5bff3db4` + `ba81407f9`）+ G5 修复（`87a76f84e`）后，**dspark + flashcomm1/SP + FULL_DECODE_ONLY 完整打通**。V41DBG 埋点未再需要（G6 结案后从新链上剔除，归档分支留底）。

**根因复盘（G3/G4/G6 同源）**：SP 的 `sp_shard`/`sp_reduce_scatter` 等纯 Python 形状算术（`(-T) % tp` padding + chunk 切分）被 dynamo 按首捕形状烘焙；`sp_shard` 对 3D 输入还 pad 错轴（第二维而非 token 轴）。全部以 `direct_register_custom_op`（`dispatch_key="PrivateUse1"` + fake impl cdiv 动态推导）解决——**图内任何依赖 Python 算术的形状决策都必须算子化**，这是本次适配最重要的可复用结论。

### 8.2 P5a：engram 适配（代码完成 ✅，启动被 host pinned 内存卡住 ❌）

#### 8.2.1 代码修改（`b7f7c05ce`，已入库 + UT 全过）

| 文件 | 修改 |
|------|------|
| `vllm_ascend/worker/v2/model_states/default.py` | `AscendModelState` 新增 `prepare_inputs`：真实步调用 `model.prepare_engram_inputs`（显式传当步 `attn_metadata`，因 hook 先于 `set_forward_context`）；dummy/profile 批调用 `prepare_engram_graph_inputs`（只暴露固定地址捕获缓冲，不跑 eager 路由，防 dummy token 污染 n-gram 历史）。新增 `prepare_dummy_inputs`：图捕获期绑定 engram 固定地址缓冲 |
| `vllm_ascend/models/deepseek_v41/model.py` | `prepare_engram` / `prepare_engram_inputs` 增加 `metadata=None` 参数（显式优先，回退 forward context，兼容 MRV1 调用方）；dummy 批经 `ring_state_update_skipped()` 跳过历史更新 |
| `tests/ut/worker/v2/test_engram_state_hooks.py` | 新增 UT 8 项（真实步路由/显式 metadata/dummy 隔离/捕获绑定/no-model 回退等），容器内 `run_ut_p5a.sh` 全过 |

**设计要点**：图捕获/回放不能追踪 `prepare_engram` 的 eager 分支（ContextVar.get() 不 dynamo-safe），寻址全部移到图外经 AscendModelState 钩子注入；图内只读固定地址缓冲（`_engram_input_buffers`），真实步在 replay 前 `copy_` 刷新行数据。

#### 8.2.2 启动脚本

`p5a-mrv2-dsv41.sh`（本地 `.tmp_main_mrv2/` + 远端 dev 目录）= P4 脚本 + `enable_engram: true` + `enable_engram_ple_offload: true` + `engram_storage: int8` + `--skip-mm-profiling`（见 8.2.3 第 1 次尝试）。

#### 8.2.3 三次启动尝试与失败记录（时间线）

| # | 时间（log） | 配置 | 失败点 | 分析 |
|---|------|------|--------|------|
| 1 | 20:41/20:48 | engram + PLE offload + **mm profiling 开** | `aclrtMallocHostWithCfg` 207001（host pinned 分配失败，mm encoder profile 阶段） | mm profile 的 pinned 分配与 engram PLE offload host 需求叠加 → 加 `--skip-mm-profiling` |
| 2 | 21:12 | + skip-mm-profiling | 主进程 config 解析阶段被 **kernel OOM-killed**（SIGKILL，非 python 异常） | host 内存瞬时紧张（孤儿进程/多容器叠加），用户手动清理 |
| 3 | 21:13:38 | 同上（用户清理后） | 走到 KV cache init：`BlockTables → StagedWriteTensor → UvaBuffer`（`patch_uva.py:132` `pin_memory=True`）207001；**伴随 21:18:56 多 worker EI0019**（HCCL bind 60000-60003 失败）+ `rtsMallocHost driver error: out of memory` | 详见 8.3 |

### 8.3 当前卡点分析：207001 host pinned 内存（第 3 次尝试）

**已排除的因素**：
- ✗ host 物理内存不足：`free -g` 显示 available 1619GB（used 393GB / total 2013GB），充足
- ✗ npu-smi "残留进程"：host PID 1025670（43GB）/1048482（56GB）经 `ps` 确认为**当前实例自己的 worker**（elapsed ~7min，与启动时间吻合），非残留
- ✗ engram 启动路径新增 pinned：`NodeShardedEngram` 的 weight/scale 均 `pin_memory=False`（non-pinned CPU）；唯一的 pinned offload buffers（上限 512MB/层）是**运行时 lookup 才分配**，启动不触发
- ✗ 纯 ulimit 说法：容器 `ulimit -l` = 64MB，但 **P4 同 ulimit 下 UvaBuffer 分配成功**，说明 64MB 并非硬性瓶颈（HCCL/CANN 驱动路径的分配方式与 torch pin_memory 不同）

**🔑 关键对比证据（2026-09-17 深夜补充）**：P4 成功日志 `194810` 中 EI0019 = **0** 条、207001 = 0 条、rtsMallocHost = 0 条（三项全零）；而 P5a 失败日志 `211338` 三项全部出现。**EI0019（HCCL 端口冲突）是 P5a 特有的**，且占用窗口在 P4 结束（~20:30）与 P5a 首启（20:41）之间打开——时间上与同机容器 `vllm-ascend-dev-148`（docker ps 显示 Up 2 hours，即 ~19:40 前后启动）高度吻合，**最大嫌疑是其他容器的 HCCL 实例占用了 60000 端口段**。EI0019 → HCCL 连接重建/重试 → pinned host 内存反复分配 → CANN 驱动级池耗尽（`rtsMallocHost: driver error: out of memory`）→ torch 层 207001，因果链完整。

**现存嫌疑（按可能性排序，已按上述证据更新）**：
1. **HCCL 端口冲突诱发（头号，证据充分）**：见上。占用者在其他容器 netns（host `ss -tln` 查不到 60000 段监听）
2. **engram CPU 表的 NUMA/驱动内存布局影响**：PLE offload 的 engram 表（int8，多层 × 大行数）分配在 worker 进程的 CPU 侧，可能挤占 CANN 驱动 host 内存池的 NUMA 局部性（P4 无此分配）
3. **瞬时 host 内存压力**：21:12 轮 OOM-kill 说明机器 host 内存水位确实紧张过（16 worker × ~18GB RSS + 其他容器）

**下一步排查（按优先级，已按对比证据更新）**：
1. **【首选】换 HCCL 端口段**：启动脚本加 `export HCCL_IF_BASE_PORT=60100`（EI0019 报错信息给出的官方方案），避开被占用的默认 60000 段——一条环境变量即可验证/绕过头号嫌疑
2. **确认端口占用者**：用户在宿主机侧检查 `vllm-ascend-dev-148` / `mrv2-dsv4-0907` / `mrv2-dsv4` 三个容器是否有活跃 vllm/HCCL 实例（`docker exec <c> ss -tln | grep 600` 或查 npu-smi 进程），协调停掉或错峰
3. **降级二分**（若换端口后仍 207001）：`enable_engram: true` + `enable_engram_ple_offload: false`（表放 device）——先验证 engram 逻辑与图模式的交互（AscendModelState 钩子链路），PLE offload 内存问题单独跟进；device 侧内存预算需先确认（engram int8 表 vs 0.9 gpu-mem-util）
4. 若仍失败：CANN 侧 host 内存池观察（`npu-smi info -t host-mem` 类命令）、dmesg OOM 记录核对

### 8.4 交接清单（当前时刻）

**环境状态**：
- 服务**已全部停止**：`pkill -9 'vllm serve'` + `spawn_main` 已执行，容器内 vllm 进程数 0；远端 watch/poll 循环脚本已停（`pkill -f p5a_watch_loop / p5a_start_watch`）
- **等待用户手动清理残留进程**（用户 09-17 深夜接手清理），清理完成后从"重启 P5a"继续
- 同机 4 容器并存：`main-mrv2-dsv41`（本任务）+ `mrv2-dsv4` + `mrv2-dsv4-0907` + `vllm-ascend-dev-148`（后三者勿动）

**本地脚本清单**（`.tmp_main_mrv2/`，均已 scp 到远端）：
| 脚本 | 用途 |
|------|------|
| `p5a-mrv2-dsv41.sh` | P5a 启动脚本（engram 全开 + skip-mm-profiling） |
| `p5a_clean_orphans.sh` | 容器内清孤儿进程（spawn_main/resource_tracker） |
| `p5a_start_watch.sh` | 后台启动 + 监视 READY/Traceback，结果写 `/tmp/p5a_start_result.txt` |
| `p5a_poll.sh` / `p5a_watch_loop.sh` | 轮询启动进度（log 尾部 + 关键里程碑 + 进程数） |
| `p5a_diag2.sh` | 207001 诊断（engram 日志行/残留进程/host free/端口/memlock） |
| `run_ut_p5a.sh` | 容器内跑 engram hooks UT |

**重启命令**（用户清理完成后）：
```powershell
# 同步代码（若本地有新提交）
powershell .\.tmp_main_mrv2\sync-to-148.ps1
# 清理 + 启动 + 监视
scp .tmp_main_mrv2/p5a-mrv2-dsv41.sh 80.5.9.148:/mnt/share/l00960935/dev/main-mrv2-dsv41/p5a-mrv2-dsv41.sh
ssh 80.5.9.148 "docker exec -d main-mrv2-dsv41 bash /tmp/p5a_start_watch.sh"
# 轮询（每 2-3 分钟）
ssh 80.5.9.148 "docker exec main-mrv2-dsv41 cat /tmp/p5a_start_result.txt"
```

**注意**：单条 ssh/本地命令总时长 **≤35s** 否则输出丢失；含 `$` 的远程命令用 PS 单引号；启动一次约 13-20 min（模型加载 ~10min + 图捕获 ~3-12min），用 watcher + 轮询模式勿阻塞等待。

**远端验证日志索引**（`logs/`）：`211338`=P5a 第 3 次（207001@UvaBuffer+EI0019）、`211200`=第 2 次（OOM-killed）、`204142/204834`=第 1 次（mm profiling 207001）、`194810`=P4 最终轮（✅ 全过基准）。

---

## 9. P5/P6 最终结果（2026-09-17 深夜 ~ 09-18 凌晨，✅ 全部完成）

### 9.1 P5a engram：两处修复后全过

**启动链路修复（与 engram 代码无关的环境问题）**：
1. mm encoder profile 的 pinned 分配 + engram PLE offload 叠加超限 → `--skip-mm-profiling`
2. **EI0019 HCCL 端口冲突**（60000-60003 被同机其他容器占用）→ HCCL 重建循环 → 驱动级 pinned host 池耗尽（`rtsMallocHost OOM`）→ 207001。修复：启动脚本 `export HCCL_IF_BASE_PORT=60100`。**对照证据**：P4 成功日志 EI0019=0 条，P5a 失败日志全出现——端口冲突是 P5a 特有
3. **残留进程陷阱再次验证**：`pkill -f 'vllm serve'` 杀不到 setproctitle 改名的 `VLLM::EngineCore/Worker/DPCoordinator`，导致下一次启动撞车（ERR99999）。修复：`p5a_kill_all.sh` 按 `VLLM::` 模式补刀

**代码修复（2 个 commit）**：
| Commit | 内容 |
|--------|------|
| `341349388` | fix(model_states): dummy 批也参与 engram 路由集合通信。**根因**：`route_many` 的 `all_gather(cpu_group)` 域为节点 16 rank（跨 DP 组）；dummy 批原先只暴露捕获缓冲不路由 → 真实组 8 rank 进 16 rank all_gather → 首请求死锁（worker 63-115% CPU 自旋 + 周期 shm_broadcast 警告，与 G6 同型但根因不同）。历史更新污染由 `ring_state_update_skipped()`（execute_dummy_batch）/`metadata=None`（profile dummy）双重守卫，与路由解耦 |
| `ef2052648` | fix(model_states): `getattr(self, "attn_metadata", None)`——profile dummy（skip_attn=True）不跑 prepare_attn，属性不存在导致 AttributeError |

**UT**：`test_engram_state_hooks.py` 更新 + 新增 profile-dummy 用例（2 修复均带回归测试）。

### 9.2 P5c / P5b：开关级恢复，一次通过

| 阶段 | 配置 | 结果 |
|------|------|------|
| P5c | + `enable_force_eplb` + `enable_shared_expert_dp` + `multistream_overlap_shared_expert` | ✅ READY + 冒烟通过（"Paris" 与 P5a 逐字节一致） |
| P5b | + `enable_dsa_cp: true`（全量配置） | ✅ READY + 冒烟通过。注意：`ascend_config.py:659` 提示 enable_dsa_cp 将被 PCP 取代（deprecation 警告，非错误）；SP 开启下捕获尺寸 [1,4] 非 tp_size 倍数被自动移除 → 实际捕获 [8,16,32] |

### 9.3 P6：压测与精度

| 项 | 结果 |
|----|------|
| **P6a 64 并发混合压测** | ✅ **64/64 全部 200 OK、0 失败**（~2 min，长短 prompt 混合触发 prefill+decode 混合批）；压测后 0 Traceback，服务健康 |
| **P6b aisbench gsm8k_lite** | ✅ **accuracy 87.50%**（28/32，temperature=0.6 与历史基线同口径，±1-2 项抖动范围内）；aisbench 位于 148 `/mnt/share/l00960935/test/benchmark/ais_bench`，容器 site-packages 的 `ais_bench/datasets/gsm8k` 软链到源树 `gsm8k_lite` |

**aisbench 运行要点**（踩坑记录）：
- 必须从非源码目录启动（`cd /tmp`），否则源树的 `datasets/` 数据目录遮蔽 HF `datasets` 包 → ImportError
- config 是**位置参数**：`ais_bench <config.py>`（不是 `--config`）
- 配置：`configs/api_examples/p6_gsm8k_dsv41.py`（VLLMCustomAPIChat → 127.0.0.1:8900, batch_size=64, temp=0.6），输出 `p6_aisbench_out/`

### 9.4 最终状态汇总（适配完成）

- **分支链**（本地 = 远端 `main-mrv2-dsv41`）：`94c34c556`（基准）→ `c5bff3db4`（SP dynamo）→ `ba81407f9`（SP 3D）→ `87a76f84e`（G5 计数）→ `b7f7c05ce`（engram 钩子）→ `16653528d`（UT 修复）→ `341349388`（dummy 路由）→ `ef2052648`（getattr）→ HEAD
- **全量配置**：DP2×TP8×EP8 + FULL_DECODE_ONLY（capture [8,16,32]）+ dspark(5) + engram(int8+PLE offload) + flashcomm1/SP + DSA-CP + force_eplb + shared-expert-DP + multistream + cpu_binding —— 全部与图模式共存
- **启动脚本**：`p5c-mrv2-dsv41.sh`（全量）；日志 `logs/mrv2-dsv41-20260917-225705.log`
- **遗留跟进项**：① HCCL_IF_BASE_PORT=60100 是规避措施，长期应协调同机容器的 HCCL 端口分配；② enable_dsa_cp 走 PCP 迁移是上游方向；③ MRV1 基线性能对比（吞吐数字）→ 已在 P7 补齐（第 10 章）；④ V41DBG 埋点在归档分支按需取用

---

## 10. P7：MRV1 基线性能对比（2026-09-18 上午，✅ 数据齐 + ⚠️ 发现 MRV1 EOS 行为差异）

### 10.1 对比方法

- **同口径**：同一数据集（GSM8K test.jsonl 前 256 题）、同一服务参数（= `p5c-mrv2-dsv41.sh` 全量配置）、同一请求参数（temp=0.6 / top_k=20 / top_p=0.95 / max_tokens=512）、64 并发、1 条预热不计入。
- **MRV2 侧**：`p5c-mrv2-dsv41.sh`（MRV2 + FULL_DECODE_ONLY 图模式，capture [8,16,32]），log `20260917-225705`。
- **MRV1 侧**：`p7-mrv1-dsv41.sh`（= P5c 但 `VLLM_USE_V2_MODEL_RUNNER=0` + `--enforce-eager`），log `20260917-234152`（日志确认 `V1 LLM engine` + `CUDAGraphMode.NONE`）。
- **runner 切换机制**：`VLLM_USE_V2_MODEL_RUNNER` 环境变量（`patch_use_v2_model_runner.py` 直接返回该值，未设置默认 MRV1）。
- 测量脚本 `p7_tput.py`（OpenAI chat API + usage token 统计 + 延迟分位），结果 JSON 容器内 `/tmp/p7_tput_mrv2.json` / `/tmp/p7_tput_mrv1.json`。

### 10.2 结果

| 指标 | MRV2 + FULL 图 | MRV1 eager |
|------|---------------|-----------|
| 成功请求 | 256/256 | 256/256 |
| 总墙钟 | **141.3 s** | 430.0 s |
| 端到端完成速度 | **3.0×** | 1× |
| 输出 tok/s | **362.0** | 304.8（~1.19×） |
| req/s | 1.81 | 0.60 |
| 延迟 p50 / p99 | 22.7 s / 81.7 s | 52.4 s / 167.2 s |
| 总输出 tokens | 51,162 | 131,072（=256×512 全部顶格） |

### 10.3 ⚠️ 重要发现：MRV1 路径 EOS 从不触发（口径偏差 + 独立跟进项）

- **现象**：MRV2 下 92%（235/256）请求 `finish_reason=stop`（平均 200 tokens，min 41 / p50 158 / max 512）；MRV1 下 **100%（256/256）`finish_reason=length` 顶格 512 tokens**，且单请求探针显示 `reasoning_tokens=512` —— 模型全程处于 `<think>` 推理，从不产出 `</think>`+EOS。
- **影响**：两轮 workload 不完全对等（MRV1 每请求多生成 2.56× tokens）。表 10.2 中「输出 tok/s」1.19× 是归一化后的保守对比；3.0× 是端到端完成速度（含 MRV1 额外 token 工作量）。
- **嫌疑**（未排查，列为跟进项）：MRV1 + dspark 投机路径的 EOS/stop-token 处理（verify 步未把 EOS 计为停止？），或 MRV1 采样参数（top_k/top_p）应用差异。**注意这不影响 MRV2 侧结论**：MRV2 的 gsm8k 87.5%（P6b）与 92% stop 率自洽。
- **跟进建议**：在 MRV1 服务上用同题对拍 `temperature=0`（确定性）+ 检查 draft verify 的 stop-token 分支；如确认 EOS 缺失，属 MRV1+dspark 功能 bug，与本次 MRV2 适配无关。

### 10.4 脚本清单（本次新增，均已在远端 dev 目录）

| 脚本 | 用途 |
|------|------|
| `p7-mrv1-dsv41.sh` | MRV1 eager 基线启动脚本（= P5c + MRV1 runner + enforce-eager） |
| `p7mrv1_start_watch.sh` | MRV1 启动 + READY/Traceback watcher（结果写容器 `/tmp/p7mrv1_start_result.txt`） |
| `p7_run_bench.sh` | 吞吐基准启动包装（nohup + 结果落盘） |
| `p7_tput.py` | 吞吐测量（固定请求集 + 64 并发 + usage 统计 + 延迟分位） |
| `p7_compare.py` | 两轮结果对比（finish_reason 分布 + token 分布） |
| `p7_probe.sh` / `p7_wait_mrv1.sh` | 单请求行为探针 / 远端等待标记 |

### 10.5 结论

- P5/P6 的 MRV2 适配在端到端吞吐维度验证通过：同配置下完成同 workload 比 MRV1 eager 快 3.0×，归一化输出吞吐 ~1.19×（且 MRV2 延迟 p50 减半以上）。
- 遗留新跟进项：MRV1+dspark 的 EOS 行为（10.3）。至此 MRV2 适配（功能 + 精度 + 性能 + 压测）全部收尾。
