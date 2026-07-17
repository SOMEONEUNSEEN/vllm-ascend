# MRV2 每日报告
生成时间: 2026-07-17 11:35:29
统计范围: 最近 7 天

**MRV2 定义**: `vllm/v1/worker/gpu/model_runner.py` 及其依赖的所有组件

MRV2 相关 commits 总数: 19

## 2026-07-16
### vllm
- **[8bfd6839](https://github.com/vllm-project/vllm/commit/8bfd6839016f9670508cf07c16c68891f83ea117)** ([#48787](https://github.com/vllm-project/vllm/pull/48787)) [Spec Decode] 新增：添加 kv_cache_dtype speculative_config control separately from target
  - 标签: `feature`, `low-risk`, `spec-decode`
  - 变更文件:
  - 修改 `vllm/config/speculative.py` (+4/-0)
  - 修改 `vllm/engine/arg_utils.py` (+4/-0)
  - 修改 `vllm/v1/spec_decode/llm_base_proposer.py` (+9/-0)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` (+8/-0)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` (+8/-0)
  - 修改 `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` (+9/-1)
  - Ascend 影响: ✓ 无影响

- **[6a9f24aa](https://github.com/vllm-project/vllm/commit/6a9f24aa8cb856235528d01a829a4ba85fc1c19d)** ([#48764](https://github.com/vllm-project/vllm/pull/48764)) [ROCm] [CI] 修复：修复 CUDA graph mem profile issue
  - 标签: `bugfix`, `mrv2`, `high-risk`, `model-runner`
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu_model_runner.py` (+13/-10)
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 修改了 vLLM 核心接口文件 vllm/v1/worker/gpu_model_runner.py，可能影响 vllm-ascend 的适配实现
    - 建议测试区域: `整体功能回归测试`

- **[ecf4aa5c](https://github.com/vllm-project/vllm/commit/ecf4aa5ce2ccd4069f12318ca9d3fcef7c9f6257)** ([#48167](https://github.com/vllm-project/vllm/pull/48167)) [Bugfix] 修复：修复 FlashInfer non-causal draft 注意力 DFlash/DSpark Blackwell
  - 标签: `bugfix`, `mrv2`, `high-risk`, `model-runner`, `attention`, `spec-decode`, `tests`
  - 变更文件（共 12 个）:
  - 新增 `tests/v1/spec_decode/test_dflash_causality.py` (+55/-0)
  - 修改 `tools/pre_commit/generate_attention_backend_docs.py` (+3/-1)
  - 修改 `vllm/model_executor/models/qwen3_dflash.py` (+24/-12)
  - 修改 `vllm/platforms/cuda.py` (+6/-1)
  - 修改 `vllm/v1/attention/backends/flashinfer.py` (+2/-1)
  - 修改 `vllm/v1/spec_decode/dflash.py` (+5/-1)
  - 修改 `vllm/v1/worker/gpu/model_runner.py` (+4/-3)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (+35/-10)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` (+5/-11)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (+0/-2)
  - ... 及其他 2 个文件
  - Ascend 影响: ✓ 无影响

- **[b7950e79](https://github.com/vllm-project/vllm/commit/b7950e798f2094b1163c22787c0ba2e3231bf01b)** ([#47460](https://github.com/vllm-project/vllm/pull/47460)) [Bugfix] 更新：Initialize draft CUDA-graph keys native draft_model proposer
  - 标签: `bugfix`, `mrv2`, `high-risk`, `model-runner`
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu_model_runner.py` (+2/-0)
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 修改了 vLLM 核心接口文件 vllm/v1/worker/gpu_model_runner.py，可能影响 vllm-ascend 的适配实现
    - 建议测试区域: `整体功能回归测试`

### vllm-ascend
- **[e77502ef](https://github.com/vllm-project/vllm-ascend/commit/e77502ef715f612f2cc6f3c833174431f26cb765)** ([#11949](https://github.com/vllm-project/vllm-ascend/pull/11949)) [Refactor] 更新：移除 weight prefetch config
  - 标签: `refactor`, `mrv2`, `high-risk`, `model-runner`, `attention`, `spec-decode`, `ops`, `quantization`, `ci`, `tests`, `docs`
  - 变更文件（共 43 个）:
  - 修改 `.github/workflows/scripts/test_config.yaml` (+2/-2)
  - 修改 `docs/hooks/nav_titles.py` (+0/-1)
  - 修改 `docs/source/_templates/Model-Deployment-Tutorial-Template.md` (+7/-8)
  - 修改 `docs/source/_templates/Model-Deployment-Tutorial-Template.zh.md` (+5/-6)
  - 修改 `docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po` (+0/-29)
  - 删除 `docs/source/locale/zh_CN/LC_MESSAGES/user_guide/feature_guide/weight_prefetch.po` (+0/-137)
  - 修改 `docs/source/locale/zh_CN/LC_MESSAGES/user_guide/support_matrix/supported_models.po` (+4/-6)
  - 修改 `docs/source/user_guide/configuration/additional_config.md` (+0/-24)
  - 删除 `docs/source/user_guide/feature_guide/weight_prefetch.md` (+0/-73)
  - 修改 `docs/source/user_guide/support_matrix/supported_models.md` (+29/-29)
  - ... 及其他 33 个文件
  - Ascend 影响: ✓ 无影响

---

## 2026-07-15
### vllm
- **[05eed72a](https://github.com/vllm-project/vllm/commit/05eed72aec6c05e6d500c7276b47f7652bb37af6)** ([#48526](https://github.com/vllm-project/vllm/pull/48526)) [ROCm] 更新：Re-enable cudagraph 内存 profiling captured current stream
  - 标签: `mrv2`, `high-risk`, `model-runner`, `distributed`
  - 变更文件:
  - 修改 `vllm/distributed/parallel_state.py` (+10/-2)
  - 修改 `vllm/v1/worker/gpu_model_runner.py` (+20/-1)
  - 修改 `vllm/v1/worker/gpu_worker.py` (+7/-5)
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 修改了 vLLM 核心接口文件 vllm/distributed/parallel_state.py，可能影响 vllm-ascend 的适配实现
    - 建议测试区域: `整体功能回归测试`

- **[3ca242d1](https://github.com/vllm-project/vllm/commit/3ca242d1b6282084b2a41e247810e632450c639a)** ([#48622](https://github.com/vllm-project/vllm/pull/48622)) [Bugfix] [R3] 更新：Exclude draft routers from expert capture
  - 标签: `bugfix`, `mrv2`, `high-risk`, `model-runner`, `tests`
  - 变更文件:
  - 修改 `tests/model_executor/test_routed_experts_capture.py` (+34/-6)
  - 修改 `vllm/v1/worker/gpu_model_runner.py` (+1/-1)
  - Ascend 影响: ✓ 无影响

### vllm-ascend
- **[6e784075](https://github.com/vllm-project/vllm-ascend/commit/6e784075dcc36b603296f03a50cdc005cffe5c61)** ([#11727](https://github.com/vllm-project/vllm-ascend/pull/11727)) [BugFix] 修复：修复 quant DP full graph mode mrv2
  - 标签: `bugfix`, `mrv2`, `low-risk`, `model-runner`, `spec-decode`, `quantization`, `ci`, `tests`
  - 变更文件:
  - 修改 `.github/workflows/scripts/test_config.yaml` (+2/-0)
  - 新增 `tests/e2e/pull_request/two_card/model_runner_v2/test_data_parallel.py` (+95/-0)
  - 修改 `tests/ut/worker/a2/test_worker_v1.py` (+2/-4)
  - 修改 `vllm_ascend/quantization/method_adapters.py` (+9/-0)
  - 修改 `vllm_ascend/worker/v2/aclgraph_utils.py` (+1/-1)
  - 修改 `vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py` (+2/-2)
  - 修改 `vllm_ascend/worker/worker.py` (+2/-1)
  - Ascend 影响: ✓ 无影响

---

## 2026-07-14
### vllm
- **[26587f95](https://github.com/vllm-project/vllm/commit/26587f9519e22a5c4549ead7595ad9ca3229c4fd)** ([#48261](https://github.com/vllm-project/vllm/pull/48261)) [BugFix] [ModelRunner V2] 修复：修复 stale attn metadata speculator prefill cudagraph capture
  - 标签: `bugfix`, `mrv2`, `high-risk`, `model-runner`, `spec-decode`
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu/cudagraph_utils.py` (+17/-35)
  - 修改 `vllm/v1/worker/gpu/model_runner.py` (+7/-3)
  - 修改 `vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py` (+11/-45)
  - 修改 `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` (+14/-12)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py` (+2/-3)
  - 修改 `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (+11/-2)
  - 修改 `vllm/v1/worker/gpu/spec_decode/speculator.py` (+10/-8)
  - Ascend 影响: ✓ 无影响

### vllm-ascend
- **[f6b33f49](https://github.com/vllm-project/vllm-ascend/commit/f6b33f49cd732733769374c19b68d26a5c210ec0)** ([#11899](https://github.com/vllm-project/vllm-ascend/pull/11899)) [Refactor] [Attention] 更新：移除 paged 注意力
  - 标签: `refactor`, `high-risk`, `model-runner`, `attention`, `tests`, `docs`
  - 变更文件（共 15 个）:
  - 修改 `csrc/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp` (+0/-1)
  - 修改 `docs/source/faqs.md` (+0/-4)
  - 修改 `docs/source/locale/zh_CN/LC_MESSAGES/faqs.po` (+0/-18)
  - 修改 `docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po` (+0/-9)
  - 修改 `docs/source/tutorials/features/suffix_speculative_decoding.md` (+1/-1)
  - 修改 `docs/source/tutorials/models/Qwen3-Dense.md` (+1/-1)
  - 修改 `docs/source/user_guide/configuration/additional_config.md` (+0/-1)
  - 修改 `tests/e2e/pull_request/two_card/test_qwen3_performance.py` (+1/-1)
  - 修改 `tests/e2e/weekly/single_node/configs/Qwen3-32B.yaml` (+0/-1)
  - 修改 `tests/ut/_310p/attention/test_attention_v1_310.py` (+1/-4)
  - ... 及其他 5 个文件
  - Ascend 影响: ✓ 无影响

- **[5083d884](https://github.com/vllm-project/vllm-ascend/commit/5083d8844310831258f085ea6dfcac4a2f76ef58)** ([#11709](https://github.com/vllm-project/vllm-ascend/pull/11709)) [CI] 更新：main2main 0710
  - 标签: `chore`, `mrv2`, `high-risk`, `model-runner`, `sample`, `distributed`, `spec-decode`, `kv-cache`, `patch`, `ci`, `tests`
  - 变更文件（共 19 个）:
  - 修改 `.github/vllm-main-verified.commit` (+1/-1)
  - 修改 `pyproject.toml` (+1/-1)
  - 修改 `requirements.txt` (+1/-1)
  - 修改 `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py` (+5/-25)
  - 修改 `tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py` (+11/-1)
  - 新增 `tests/ut/patch/test_hunyuan_vl_processor_compat.py` (+388/-0)
  - 修改 `vllm_ascend/__init__.py` (+5/-0)
  - 修改 `vllm_ascend/core/single_type_kv_cache_manager.py` (+16/-6)
  - 修改 `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py` (+5/-1)
  - 修改 `vllm_ascend/patch/__init__.py` (+24/-0)
  - ... 及其他 9 个文件
  - Ascend 影响: ✓ 无影响

---

## 2026-07-13
### vllm
- **[1be6e937](https://github.com/vllm-project/vllm/commit/1be6e937b2b49bae652370d80294f6171bd7b981)** ([#48483](https://github.com/vllm-project/vllm/pull/48483)) 更新：降低 内存 所需 捕获 CUDA图 大 cudagraph 尺寸
  - 标签: `mrv2`, `high-risk`, `model-runner`
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu_model_runner.py` (+5/-1)
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 修改了 vLLM 核心接口文件 vllm/v1/worker/gpu_model_runner.py，可能影响 vllm-ascend 的适配实现
    - 建议测试区域: `整体功能回归测试`

### vllm-ascend
- **[41ff81e1](https://github.com/vllm-project/vllm-ascend/commit/41ff81e1a7a92e6e3f546198d80702d78b7a50b7)** 新增功能，涉及 Model Runner 模块，修改 .github/workflows/scripts/test_config.yaml；新增 tests/e2e/pull_request/one_card/spec_decode/test_dspark.py；修改 tests/e2e/pull_request/one_card/spec_decode/utils.py；及其他 9 个文件，变更 410 行，删除 24 行。
  - 标签: `feature`, `low-risk`, `model-runner`, `spec-decode`, `ops`, `patch`, `ci`, `tests`
  - 变更文件（共 12 个）:
  - 修改 `.github/workflows/scripts/test_config.yaml` (+9/-0)
  - 新增 `tests/e2e/pull_request/one_card/spec_decode/test_dspark.py` (+87/-0)
  - 修改 `tests/e2e/pull_request/one_card/spec_decode/utils.py` (+8/-0)
  - 修改 `vllm_ascend/ops/triton/spec_decode/utils.py` (+8/-2)
  - 修改 `vllm_ascend/patch/__init__.py` (+18/-0)
  - 修改 `vllm_ascend/patch/worker/__init__.py` (+1/-0)
  - 新增 `vllm_ascend/patch/worker/patch_qwen3_dspark.py` (+15/-0)
  - 修改 `vllm_ascend/spec_decode/__init__.py` (+3/-0)
  - 修改 `vllm_ascend/spec_decode/dflash_proposer.py` (+2/-2)
  - 新增 `vllm_ascend/spec_decode/dspark_proposer.py` (+196/-0)
  - ... 及其他 2 个文件
  - Ascend 影响: ✓ 无影响

---

## 2026-07-12
### vllm
- **[a02984ed](https://github.com/vllm-project/vllm/commit/a02984ed471488c0f0e8f73cab21be4325992d4c)** ([#47006](https://github.com/vllm-project/vllm/pull/47006)) [Perf] 更新：Qwen Replace MoE all-reduce reduce-scatter
  - 标签: `performance`, `model-runner`, `medium-risk`, `attention`
  - 变更文件:
  - 修改 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (+2/-0)
  - 修改 `vllm/model_executor/models/qwen3_5.py` (+9/-0)
  - 修改 `vllm/model_executor/models/qwen3_next.py` (+102/-11)
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 涉及 vLLM 核心代码变更（vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/model_executor/models/qwen3_5.py, vllm/model_executor/models/qwen3_next.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
    - 建议测试区域: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py`

- **[fc1c5480](https://github.com/vllm-project/vllm/commit/fc1c548093029f6487bbdc9c612995dfe7621a75)** ([#46725](https://github.com/vllm-project/vllm/pull/46725)) 更新：Runtime Draft Weight 更新 Speculative Decoding
  - 标签: `docs`, `worker`, `engine`, `model-runner`, `low-risk`, `distributed`
  - 变更文件（共 12 个）:
  - 修改 `docs/training/weight_transfer/base.md` (+4/-0)
  - 修改 `tests/entrypoints/openai/test_openai_schema.py` (+1/-0)
  - 修改 `tests/v1/worker/test_gpu_worker_weight_transfer.py` (+6/-0)
  - 修改 `vllm/distributed/weight_transfer/base.py` (+18/-0)
  - 修改 `vllm/distributed/weight_transfer/sparse_nccl_engine.py` (+1/-0)
  - 修改 `vllm/engine/protocol.py` (+4/-0)
  - 修改 `vllm/entrypoints/llm.py` (+4/-0)
  - 修改 `vllm/entrypoints/serve/dev/rlhf/api_router.py` (+6/-0)
  - 修改 `vllm/v1/engine/async_llm.py` (+4/-0)
  - 修改 `vllm/v1/worker/gpu/model_runner.py` (+6/-0)
  - ... 及其他 2 个文件
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 涉及 vLLM 核心代码变更（vllm/distributed/weight_transfer/base.py, vllm/distributed/weight_transfer/sparse_nccl_engine.py, vllm/engine/protocol.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
    - 建议测试区域: `vllm/distributed/weight_transfer/base.py`, `vllm/distributed/weight_transfer/sparse_nccl_engine.py`, `vllm/engine/protocol.py`, `vllm/entrypoints/llm.py`, `vllm/entrypoints/serve/dev/rlhf/api_router.py`

- **[481e481b](https://github.com/vllm-project/vllm/commit/481e481be786c1ca3229e26aa34c15ffd22375af)** ([#46384](https://github.com/vllm-project/vllm/pull/46384)) [2/N] 修复：核心 支持 partial prefix 缓存 hit hybrid model
  - 标签: `worker`, `engine`, `model-runner`, `config`, `scheduler`, `prefix-caching`, `bugfix`, `distributed`, `medium-risk`
  - 变更文件（共 21 个）:
  - 新增 `tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py` (+816/-0)
  - 修改 `tests/v1/core/test_deferred_block_free.py` (+62/-0)
  - 修改 `tests/v1/core/test_kv_cache_utils.py` (+60/-0)
  - 修改 `tests/v1/core/test_prefix_caching.py` (+4/-1)
  - 修改 `tests/v1/core/test_single_type_kv_cache_manager.py` (+8/-6)
  - 修改 `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` (+15/-15)
  - 修改 `tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py` (+1/-0)
  - 修改 `vllm/config/cache.py` (+10/-10)
  - 修改 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` (+17/-9)
  - 修改 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` (+3/-1)
  - ... 及其他 11 个文件
  - Ascend 影响: ⚠️ 影响 Ascend
    - 影响描述: 涉及 vLLM 核心代码变更（vllm/config/cache.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
    - 建议测试区域: `vllm/config/cache.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py`, `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`, `vllm/engine/arg_utils.py`, `vllm/v1/core/block_pool.py`

### vllm-ascend
- **[d19628a1](https://github.com/vllm-project/vllm-ascend/commit/d19628a1b292cba1ef33593a6446d3777f28574e)** 变更：[BugFix]Added the store_kv_block_metadata ascendC operator (#11865)
  - 标签: `kernels`, `spec-decode`, `bugfix`, `model-runner`, `medium-risk`, `ascend`, `attention`, `worker`
  - 变更文件（共 27 个）:
  - 修改 `csrc/attention/store_kv_block/op_host/CMakeLists.txt` (+0/-37)
  - 修改 `csrc/attention/store_kv_block/op_host/store_kv_block_infershape.cpp` (+0/-7)
  - 修改 `csrc/attention/store_kv_block/op_host/store_kv_block_tiling.cpp` (+11/-7)
  - 修改 `csrc/attention/store_kv_block/op_kernel/store_kv_block.h` (+11/-12)
  - 修改 `csrc/attention/store_kv_block/store_kv_block_torch_adpt.h` (+0/-92)
  - 新增 `csrc/attention/store_kv_block_metadata/CMakeLists.txt` (+10/-0)
  - 新增 `csrc/attention/store_kv_block_metadata/op_api/aclnn_store_kv_block_metadata.cpp` (+79/-0)
  - 新增 `csrc/attention/store_kv_block_metadata/op_api/aclnn_store_kv_block_metadata.h` (+36/-0)
  - 新增 `csrc/attention/store_kv_block_metadata/op_api/l0_store_kv_block_metadata.cpp` (+53/-0)
  - 新增 `csrc/attention/store_kv_block_metadata/op_api/l0_store_kv_block_metadata.h` (+26/-0)
  - ... 及其他 17 个文件
  - Ascend 影响: ✓ 无影响

---

## 2026-07-11
### vllm
- **[04d553f3](https://github.com/vllm-project/vllm/commit/04d553f390fd37e09ab111936ef1592881299957)** [Misc] Use meta tensor for KV cache stride calculation (#47316)
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu/attn_utils.py` (+1/-1)
  - Ascend 影响: ✓ 无影响

- **[9c18e90f](https://github.com/vllm-project/vllm/commit/9c18e90f6c94b90ecdaa99b2230389ba40e0fc69)** [BugFix] Fix packed HND KV cache reshape for FlashAttention (#47314)
  - 变更文件:
  - 修改 `vllm/v1/worker/gpu/attn_utils.py` (+1/-1)
  - Ascend 影响: ✓ 无影响

- **[5314665b](https://github.com/vllm-project/vllm/commit/5314665badcb93f798e117aacad8ce02f148cd73)** 为 DFlash 启用了注意力后端选择。在 load_dflash_model 中，将 speculative_config.attention_backend 传递给 draft 模型的 attention_config.backend，使得 draft 模型可以使用与目标模型不同的注意力后端。
  - 标签: `feature, low-risk, spec-decode, attention`
  - Ascend 影响: ✓ 无影响

- **[3daea7ce](https://github.com/vllm-project/vllm/commit/3daea7ceb990bff87e925b2f4b77325af052282f)** 修复了 Mamba 混合模型中 seq_lens_cpu_upper_bound 的传递问题。在 MambaHybridModelState.prepare_attn 方法中，将 seq_lens_cpu_upper_bound 参数传递给 prepare_attn 调用。
  - 标签: `bugfix, low-risk, model-runner`
  - Ascend 影响: ✓ 无影响

- **[32bb3195](https://github.com/vllm-project/vllm/commit/32bb3195f0b93f6971781479591f7a6ee666e7dc)** 为大型 logprobs 请求限制了内存使用。在 compute_token_logprobs 函数中，将 _topk_log_softmax_kernel 的 PADDED_TOPK 参数替换为 TOPK_BLOCK_SIZE，并设置上限为 1024。当 num_logprobs 很大时，kernel 会分块处理，避免一次性分配过大的内存。
  - 标签: `performance, low-risk, sampler`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1. compute_token_logprobs 函数中 _topk_log_softmax_kernel 的 PADDED_TOPK 参数被替换为 TOPK_BLOCK_SIZE，且新增了 _MAX_TOPK_BLOCK = 1024 上限。AscendSampler 如果实现了自己的 compute_token_logprobs 或使用了相同的 kernel，需要同步适配。2. _topk_log_softmax_kernel 的 kernel 参数从 PADDED_TOPK 改为 TOPK_BLOCK_SIZE，且内部逻辑改为循环分块处理。如果 Ascend 有自定义的 Triton kernel 实现，需要同步修改。
  - 建议测试区域: vllm_ascend/sample/sampler.py

- **[c53994e1](https://github.com/vllm-project/vllm/commit/c53994e1348bac3496aafb88e9e731124a00a8a7)** 在拒绝采样中使用 log1p 提高数值稳定性。将 rejection_sampler_utils.py 中的 tl.log(1 - ratio) 替换为 tldevice.log1p(-ratio)，当 ratio 接近 1 时，log1p 比 log(1 - x) 具有更高的数值精度。
  - 标签: `performance, low-risk, spec-decode`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-24
### vllm
- **[84c62e1c](https://github.com/vllm-project/vllm/commit/84c62e1cbdef4250fbfda83782fd250e07ad0256)** 该 commit 为多模态模型（特别是 Qwen2.5-VL、Qwen3-VL 等支持 EVS 的模型）添加了 Efficient Video Sampling（EVS）支持。核心变更包括：1) 新增 vllm/v1/worker/gpu/model_states/mm_pruning.py 文件，实现 MultiModalPruner 类，用于处理 M-RoPE 位置重计算和嵌入修剪；2) 修改 ModelState 接口（interface.py），新增 gather_mm_embeddings 默认方法，并将 get_mm_embeddings 方法签名增加 req_states 参数；3) 修改 DefaultModelState（default.py），集成 MultiModalPruner 的 recompute 和 strip 逻辑；4) 修改 GPUModelRunner（model_runner.py），简化 gather_mm_embeddings 调用，将参数传递改为直接传递 input_batch；5) 修改 Qwen2.5-VL 和 Qwen3-VL 模型的 recompute_mrope_positions 方法，支持 input_ids 为 torch.Tensor 类型；6) 修改 RopeState（rope.py），新增 read_prefill_positions 和 update_prefill_positions 方法用于读写分阶段 prefill 位置。该变更对多模态模型的视频处理性能有显著提升，但涉及多个接口变更，需要确保所有 ModelState 子类同步更新。
  - 标签: `feature, medium-risk, model-runner, multimodal`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 ModelState 接口（vllm/v1/worker/gpu/model_states/interface.py）中的 get_mm_embeddings 方法签名，新增 req_states 参数。vllm-ascend 的 NPUModelRunner 继承自 GPUModelRunner，其 get_mm_embeddings 方法在 default.py 中被重写，但 vllm-ascend 的 NPUModelRunner 也重写了 get_mm_embeddings 方法（在 vllm_ascend/worker/model_runner_v1.py 中），需要检查是否适配了新的 req_states 参数。同时，ModelState 接口新增了 gather_mm_embeddings 默认方法，vllm-ascend 的 DefaultModelState 重写了该方法，需要确认是否兼容。此外，RopeState 新增了 read_prefill_positions 和 update_prefill_positions 方法，vllm-ascend 如果使用了 RopeState 则需要同步更新。

- **[7ee4d220](https://github.com/vllm-project/vllm/commit/7ee4d220097db4b397e55fd4ad58caf6a7977c5b)** 修复 rejection sampler 中 placeholder draft token (-1) 的处理。在 rejection_greedy_sample_kernel 和 rejection_random_sample_kernel 中，添加了对 draft_token_id < 0 的检查，确保 placeholder token 被拒绝而不是被采样。同时修复了 _rejection_kernel 中可能的 OOB 指针访问。这是一个 Bug 修复，风险较低。
  - 标签: `bugfix, low-risk, spec-decode, sampler`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 RejectionSampler 的 Triton kernel 和 rejection_sampler_utils 中的 _rejection_kernel。vllm-ascend 的 AscendSampler 需要同步更新 placeholder draft token 的处理逻辑。
  - 建议测试区域: vllm_ascend/sample/, vllm_ascend/spec_decode/

- **[e2bdc246](https://github.com/vllm-project/vllm/commit/e2bdc24612ab0b7bf7a1bc67c955fd244e8660c4)** 修复 ROCm 平台在 Ray driver 线程中使用 use_v2_model_runner 的问题。将 CUDA_VISIBLE_DEVICES 的环境变量检查改为根据平台选择 HIP_VISIBLE_DEVICES 或 CUDA_VISIBLE_DEVICES。这是一个跨平台兼容性修复，风险较低。
  - 标签: `bugfix, low-risk, rocm`
  - Ascend 影响: ✓ 无影响

- **[0a3e2dbc](https://github.com/vllm-project/vllm/commit/0a3e2dbc09c8a70dfd18f728bb829bf29ffa7da6)** 在 MoE 中跳过 DP padding tokens。新增 VLLM_MOE_SKIP_PADDING 环境变量（默认关闭），当启用时，在 modular_kernel._prepare 和 DeepSeek V4 的 forward 中，将 padding tokens 的 expert id 设为 -1，使其被 MoE 内核跳过。同时修改了 InputBatch 和 CUDA Graph 捕获逻辑，添加 is_padding 标记。这是一个性能优化，风险中等。
  - 标签: `performance, medium-risk, moe`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-23
### vllm
- **[f3410b3b](https://github.com/vllm-project/vllm/commit/f3410b3bb16b1b0f33468a65f260148565c9948c)** 修复moe_wna16.py中tp_size的访问路径问题。原代码直接使用layer.moe_config.moe_parallel_config.tp_size，但RoutedExperts兼容性要求使用layer.moe_config.tp_size。变更将两处tp_size引用从moe_parallel_config.tp_size改为moe_config.tp_size，确保与RoutedExperts的接口兼容。这是一个低风险的bug修复。
  - 标签: `bugfix, low-risk, model-runner`
  - Ascend 影响: ✓ 无影响

- **[901a3b09](https://github.com/vllm-project/vllm/commit/901a3b091cf1c952ab582aefa6597e98f22055e5)** 修复GPT-OSS PP>1与EP组合问题。将get_ep_group().rank改为get_ep_group().rank_in_group，确保在流水线并行（PP）和专家并行（EP）同时启用时，正确获取EP组内的rank而非全局rank。
  - 标签: `bugfix, low-risk, model-runner`
  - Ascend 影响: ✓ 无影响

- **[a46f3eb2](https://github.com/vllm-project/vllm/commit/a46f3eb232b8a74bab0aab02b6a070ebf337125f)** 修复Model Runner V2中allowed_token_ids在logit bias kernel中的保留问题。在_bias_kernel中添加tl.debug_barrier()同步点，确保在读取原始logits后、写入-inf之前完成所有读取操作，以及在写入-inf完成后、恢复allowed_token_ids之前完成所有写入操作。这修复了Triton kernel中由于内存操作顺序问题导致部分allowed_token_ids被错误覆盖的bug。
  - 标签: `bugfix, medium-risk, sampler`
  - Ascend 影响: ✓ 无影响

- **[04c2a8de](https://github.com/vllm-project/vllm/commit/04c2a8deac44fdb1ca3e2b5ec3e6bf16f3f6a914)** 修复DeepEP V2中无效recv_topk_idx填充。新增_globalize_recv_topk_idx_kernel Triton kernel和_globalize_recv_topk_idx函数，用于将本地expert ID转换为全局expert ID，并将无效行（非本地expert、越界expert、未接收token的行）的recv_topk_idx填充为-1。这防止了未初始化缓冲区中的陈旧数据被错误地视为有效路由token。
  - 标签: `bugfix, medium-risk, model-runner`
  - Ascend 影响: ✓ 无影响

- **[8207ce08](https://github.com/vllm-project/vllm/commit/8207ce085069c1b6cf448b3172653eecfa8478ae)** 修复humming lm_head崩溃和FusedMoE weight_shape类型强制。主要变更：1) 在humming_utils.py中，修复input_size_per_partition的获取逻辑，使用hasattr避免在缺少input_size属性的层（如ParallelLMHead）上崩溃；2) 在routed_experts.py中，修复_load_single_value方法，直接赋值loaded_weight而非通过_to_scalar转换，以支持size-2的weight_shape参数。
  - 标签: `bugfix, medium-risk, quantization, model-runner`
  - Ascend 影响: ✓ 无影响

- **[e4859206](https://github.com/vllm-project/vllm/commit/e48592066ee4a435c9ac5316edbecb887596de02)** 为DeepEP V2绑定num_max_tokens_per_rank。在do_expand=False模式下，将num_max_tokens_per_rank从默认的max_num_batched_tokens（可能很大）绑定到实际DP-padded batch size的2的幂次上取整。这减少了DeepEP JIT编译的dispatch kernel数量，避免高并发时的cicc风暴。
  - 标签: `performance, medium-risk, model-runner`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-22
### vllm
- **[cec2ec11](https://github.com/vllm-project/vllm/commit/cec2ec11760f9f3beabd4c90451936078bf91533)** 修复异步推测解码中的竞态条件。在`gpu_model_runner.py`的`_prepare_inputs`方法中，当使用异步调度且mamba缓存模式不是"align"时，跳过CPU accepted counts的同步，避免与正在进行的D2H拷贝和输入批处理行移动产生竞态。同时修复了`gdn_attn.py`中CUDA Graph padding使用`m.num_actual_tokens`而非`m.num_reqs`的问题。这是一个中等风险的bug修复，影响混合模型的异步推测解码。
  - 标签: `bugfix, medium-risk, spec_decode, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响vllm-ascend的NPUModelRunner：1) `GPUModelRunner._prepare_inputs`方法中关于`num_accepted_tokens`同步的逻辑变更，vllm-ascend的`NPUModelRunner`如果重写了此方法，需要评估是否需要应用相同的竞态修复；2) 如果NPUModelRunner继承自GPUModelRunner且未重写`_prepare_inputs`，则自动受益于此修复。

---

## 2026-06-21
### vllm
- **[2cac89f9](https://github.com/vllm-project/vllm/commit/2cac89f9da865dfaceb6d337d97aaff5c9195e48)** 该commit为DFlash推测解码场景支持混合KV页面大小。核心变更：1) 在AttentionSpec中新增indexes_kv_by_block_stride字段，标识后端是否支持按块步长索引KV页面；2) AttentionBackend新增indexes_kv_by_block_stride类方法，通过get_kv_cache_stride_order判断；3) unify_kv_cache_spec_page_size逻辑扩展：当页面大小不可整除时，若后端支持则使用padding对齐，否则抛出NotImplementedError；4) _reshape_kv_cache重构为_reshape_attention_kv_cache，支持padded页面的strided view；5) use_uniform_kv_cache简化为直接检查kv_cache_spec.indexes_kv_by_block_stride。潜在风险：新增抽象方法需要所有AttentionBackend子类实现或继承默认行为。
  - 标签: `feature, medium-risk, attention, spec-decode, kv-cache`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) AttentionBackend新增indexes_kv_by_block_stride类方法，vllm-ascend的AscendAttentionBackend需实现此方法，否则默认返回False；2) AttentionSpec新增indexes_kv_by_block_stride字段，所有AttentionSpec子类（包括vllm-ascend使用的）需处理此字段；3) _reshape_kv_cache逻辑重构为_reshape_attention_kv_cache，NPUModelRunner中重写的_reshape_kv_cache_tensors方法需同步更新；4) use_uniform_kv_cache方法签名变更（移除cache_dtype参数），vllm-ascend的patch中若重写了此方法需同步。
  - 建议测试区域: vllm_ascend/attention/, vllm_ascend/worker/model_runner_v1.py

- **[183a430c](https://github.com/vllm-project/vllm/commit/183a430c137db3d5cd0b9025b816f26ee87328e7)** 该commit修复了V2 GPU采样器中min_tokens的off-by-one错误。在_bias_kernel中，将条件从pos < min_len改为pos + 1 < min_len，因为pos是0-based索引，而min_len是1-based计数。这是一个典型的off-by-one bug修复。
  - 标签: `bugfix, low-risk, sampler, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 该变更修改了vllm/v1/worker/gpu/sample/logit_bias.py中的_bias_kernel Triton内核。vllm-ascend的AscendSampler如果使用了类似的Triton内核实现min_tokens逻辑，需要同步修复此off-by-one错误。如果vllm-ascend使用独立的采样器实现，则不受影响。

- **[7df3d7da](https://github.com/vllm-project/vllm/commit/7df3d7dada840c68b85b26b79de7f59f676d58e3)** 该commit是一次大规模重构，确保所有异步H2D拷贝前内存已pinned。核心变更：1) 将PIN_MEMORY常量的定义从platform_utils迁移到torch_utils，统一使用is_pin_memory_available()；2) 新增async_tensor_h2d和np_to_pinned_tensor工具函数，封装pinning和异步拷贝逻辑；3) 所有模块中的pin_memory引用从is_pin_memory_available()迁移到PIN_MEMORY常量；4) 移除InputBatch的pin_memory参数，统一使用PIN_MEMORY；5) 多处手动pin_memory()调用改为使用async_tensor_h2d。潜在风险：大规模重构可能引入回归，但逻辑等价。
  - 标签: `refactor, performance, high-risk, core, memory`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) PIN_MEMORY常量从vllm.utils.platform_utils迁移到vllm.utils.torch_utils，vllm-ascend中所有引用is_pin_memory_available()的代码需改为引用PIN_MEMORY；2) async_tensor_h2d函数签名变更（参数顺序调整，新增np.ndarray和torch.Tensor支持），vllm-ascend中重写的async_tensor_h2d（如vllm/v1/worker/cpu/shm.py）需同步更新；3) InputBatch构造函数移除pin_memory参数，NPUModelRunner中创建InputBatch的代码需移除该参数；4) Sampler的pin_memory初始化从is_pin_memory_available()改为PIN_MEMORY，AscendSampler需同步；5) ThinkingBudgetStateHolder的maybe_create_thinking_budget_state_holder函数移除is_pin_memory参数，vllm-ascend中相关调用需更新。
  - 建议测试区域: vllm_ascend/worker/, vllm_ascend/sample/, vllm_ascend/patch/

- **[6e919960](https://github.com/vllm-project/vllm/commit/6e919960af42f79d6811d84b2d4316212fcf59cb)** 该commit优化了调度器中all_token_ids的拷贝逻辑。对于V2模型运行器，跳过all_token_ids的拷贝，因为V2运行器不使用此数据。变更包括：1) _make_cached_request_data中仅在非V2运行器时拷贝all_token_ids；2) schedule()中合并scheduled_new_reqs和scheduled_resumed_reqs时使用extend/clear替代+操作；3) 仅在非V2运行器时更新prev_step_scheduled_req_ids。
  - 标签: `performance, low-risk, scheduler, model-runner`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-18
### vllm
- **[ebbb2d55](https://github.com/vllm-project/vllm/commit/ebbb2d55ace74b7066bca0ff8f333012bb8c4299)** 修复Eagle推测解码中LoRA嵌入层共享问题。当目标模型的embed_tokens是LoRA包装层时，共享给draft模型时应该使用base_layer而非包装层，否则draft模型会使用目标模型的punica metadata（sized for target token count），导致多步draft解码时GPU越界访问。
  - 标签: `bugfix, medium-risk, speculative-decoding, lora`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-17
### vllm
- **[14b438a9](https://github.com/vllm-project/vllm/commit/14b438a98b0beb148c4bbe0ff126f2a32696aa92)** 修复ModelRunnerV2的多个兼容性问题。主要变更：1) 在VllmConfig中添加external_launcher + PP > 1的检查，标记为不支持；2) 在model_runner.py中，当inputs_embeds不为空且模型不需要原始input tokens时，将input_ids设为None；3) 在execute_model的model_inputs中添加intermediate_tensors=None；4) 修复whisper.py中_get_encoder_seq_lens的参数传递。
  - 标签: `bugfix, medium-risk, model-runner`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-16
### vllm
- **[2addbb9c](https://github.com/vllm-project/vllm/commit/2addbb9cc97e2f75165ab3b81c4287a1dd8a5b0c)** 修复多模态模型使用提示嵌入时的异步调度支持。移除了 `VllmConfig.__post_init__` 中禁止异步调度与提示嵌入同时使用的检查。在 `GPUModelRunner._prepare_input_ids` 中，当启用提示嵌入时，确保 `is_token_ids` 在异步路径中正确复制到 GPU。
  - 标签: `bugfix, medium-risk, scheduler, multimodal`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-15
### vllm
- **[b8336c3c](https://github.com/vllm-project/vllm/commit/b8336c3c7c298e0878f22a7bf70f4e295b2f4e01)** 该 commit 修复了 V2 model-runner 中 attention 分组的问题。在 init_attn_backend 函数中，attention 分组的 key 从 (attn_backend.full_cls_name(), layer_kv_cache_spec) 扩展为 (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)。这样具有不同 Q-head 数量的层（例如推测解码中的 draft head 和 target head）会获得独立的 metadata builder，避免了共享 metadata builder 导致的冲突。这是一个重要的 bugfix，影响使用推测解码的模型。
  - 标签: `bugfix, medium-risk, attention, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了 init_attn_backend 函数中 attention 分组的 key，增加了 num_heads_q 维度。vllm-ascend 的 NPUModelRunner 继承自 GPUModelRunner，如果 vllm-ascend 有自定义的 attention 初始化逻辑或 patch 了 init_attn_backend，需要同步更新分组 key 的逻辑。

---
