# MRV2 每日报告
生成时间: 2026-07-14 11:04:38
统计范围: 最近 3 天
MRV2 相关 commits 总数: 6

## 2026-07-13
### vllm
- **[1be6e937](https://github.com/vllm-project/vllm/commit/1be6e937b2b49bae652370d80294f6171bd7b981)** 涉及 Model Runner 模块，修改 vllm/v1/worker/gpu_model_runner.py，变更 5 行，删除 1 行。
  - 标签: `mrv2, high-risk, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了 vLLM 核心接口文件 vllm/v1/worker/gpu_model_runner.py，可能影响 vllm-ascend 的适配实现
  - 建议测试区域: 整体功能回归测试

### vllm-ascend
- **[41ff81e1](https://github.com/vllm-project/vllm-ascend/commit/41ff81e1a7a92e6e3f546198d80702d78b7a50b7)** 新增功能，涉及 Model Runner 模块，修改 .github/workflows/scripts/test_config.yaml；新增 tests/e2e/pull_request/one_card/spec_decode/test_dspark.py；修改 tests/e2e/pull_request/one_card/spec_decode/utils.py；及其他 9 个文件，变更 410 行，删除 24 行。
  - 标签: `feature, low-risk, model-runner, spec-decode, ops, patch, ci, tests`

---

## 2026-07-12
### vllm
- **[a02984ed](https://github.com/vllm-project/vllm/commit/a02984ed471488c0f0e8f73cab21be4325992d4c)** 变更：[Perf][Qwen] Replace MOE all-reduce with reduce-scatter (#47006)
  - 标签: `performance, model-runner, medium-risk, attention`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/model_executor/models/qwen3_5.py, vllm/model_executor/models/qwen3_next.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/model_executor/models/qwen3_5.py, vllm/model_executor/models/qwen3_next.py

- **[fc1c5480](https://github.com/vllm-project/vllm/commit/fc1c548093029f6487bbdc9c612995dfe7621a75)** 变更：Runtime Draft Weight Update for Speculative Decoding (#46725)
  - 标签: `docs, worker, engine, model-runner, low-risk, distributed`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/distributed/weight_transfer/base.py, vllm/distributed/weight_transfer/sparse_nccl_engine.py, vllm/engine/protocol.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/distributed/weight_transfer/base.py, vllm/distributed/weight_transfer/sparse_nccl_engine.py, vllm/engine/protocol.py, vllm/entrypoints/llm.py, vllm/entrypoints/serve/dev/rlhf/api_router.py

- **[481e481b](https://github.com/vllm-project/vllm/commit/481e481be786c1ca3229e26aa34c15ffd22375af)** 变更：[2/N][Core] support partial prefix cache hit for hybrid model (#46384)
  - 标签: `worker, engine, model-runner, config, scheduler, prefix-caching, bugfix, distributed, medium-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/config/cache.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/config/cache.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py, vllm/engine/arg_utils.py, vllm/v1/core/block_pool.py

### vllm-ascend
- **[d19628a1](https://github.com/vllm-project/vllm-ascend/commit/d19628a1b292cba1ef33593a6446d3777f28574e)** 变更：[BugFix]Added the store_kv_block_metadata ascendC operator (#11865)
  - 标签: `kernels, spec-decode, bugfix, model-runner, medium-risk, ascend, attention, worker`

---
