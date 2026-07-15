# MRV2 每日报告
生成时间: 2026-07-15 14:27:17
统计范围: 最近 7 天

**MRV2 定义**: `vllm/v1/worker/gpu/model_runner.py` 及其依赖的所有组件

MRV2 相关 commits 总数: 24

## 2026-07-15
### vllm-ascend
- **[6e784075](https://github.com/vllm-project/vllm-ascend/commit/6e784075dcc36b603296f03a50cdc005cffe5c61)** 【变更概述】【MRV2】[BugFix] 【MRV2】[MRV2] 【MRV2】修复：修复 quant DP full graph mode mrv2 【PR号】#11727 【变更类型】此提交为问题修复类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 .github/workflows/scripts/test_config.yaml (+2/-0); 新增 tests/e2e/pull_request/two_card/model_runner_v2/test_data_parallel.py (+95/-0); 修改 tests/ut/worker/a2/test_worker_v1.py (+2/-4); 修改 vllm_ascend/quantization/method_adapters.py (+9/-0); 修改 vllm_ascend/worker/v2/aclgraph_utils.py (+1/-1); ... 及其他 2 个文件 【统计数据】共修改 7 个文件，新增 113 行，删除 8 行 【风险评估】低风险 - 变更范围较小，影响可控 【涉及模块】model-runner, spec-decode, quantization, tests
  - 标签: `bugfix, mrv2, mrv2, low-risk, model-runner, spec-decode, quantization, ci, tests`

---

## 2026-07-14
### vllm
- **[26587f95](https://github.com/vllm-project/vllm/commit/26587f9519e22a5c4549ead7595ad9ca3229c4fd)** 【变更概述】【MRV2】[BugFix] 【MRV2】[ModelRunner V2] 【MRV2】修复：修复 stale attn metadata speculator prefill cudagraph capture 【PR号】#48261 【变更类型】此提交为问题修复类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 vllm/v1/worker/gpu/cudagraph_utils.py (+17/-35); 修改 vllm/v1/worker/gpu/model_runner.py (+7/-3); 修改 vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py (+11/-45); 修改 vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+14/-12); 修改 vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py (+2/-3); ... 及其他 2 个文件 【统计数据】共修改 7 个文件，新增 72 行，删除 108 行 【风险评估】高风险 - 涉及核心接口或架构变更，可能影响系统稳定性 【涉及模块】model-runner, spec-decode
  - 标签: `bugfix, mrv2, high-risk, model-runner, spec-decode`
  - Ascend 影响: ✓ 无影响

### vllm-ascend
- **[f6b33f49](https://github.com/vllm-project/vllm-ascend/commit/f6b33f49cd732733769374c19b68d26a5c210ec0)** 【变更概述】【MRV2】[Refactor] 【MRV2】[Attention] 【MRV2】更新：移除 paged 注意力 【PR号】#11899 【变更类型】此提交为代码重构类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 csrc/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp (+0/-1); 修改 docs/source/faqs.md (+0/-4); 修改 docs/source/locale/zh_CN/LC_MESSAGES/faqs.po (+0/-18); 修改 docs/source/locale/zh_CN/LC_MESSAGES/user_guide/configuration/additional_config.po (+0/-9); 修改 docs/source/tutorials/features/suffix_speculative_decoding.md (+1/-1); ... 及其他 10 个文件 【统计数据】共修改 15 个文件，新增 19 行，删除 412 行 【风险评估】高风险 - 涉及核心接口或架构变更，可能影响系统稳定性 【涉及模块】model-runner, attention, tests
  - 标签: `refactor, high-risk, model-runner, attention, tests, docs`

- **[5083d884](https://github.com/vllm-project/vllm-ascend/commit/5083d8844310831258f085ea6dfcac4a2f76ef58)** 【变更概述】【MRV2】[CI] 【MRV2】更新：main2main 0710 【PR号】#11709 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 .github/vllm-main-verified.commit (+1/-1); 修改 pyproject.toml (+1/-1); 修改 requirements.txt (+1/-1); 修改 tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py (+5/-25); 修改 tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py (+11/-1); ... 及其他 14 个文件 【统计数据】共修改 19 个文件，新增 780 行，删除 92 行 【风险评估】高风险 - 涉及核心接口或架构变更，可能影响系统稳定性 【涉及模块】model-runner, sample, distributed, spec-decode, kv-cache, patch, tests
  - 标签: `chore, mrv2, high-risk, model-runner, sample, distributed, spec-decode, kv-cache, patch, ci, tests`

---

## 2026-07-13
### vllm
- **[1be6e937](https://github.com/vllm-project/vllm/commit/1be6e937b2b49bae652370d80294f6171bd7b981)** 【变更概述】【MRV2】更新：降低 内存 所需 捕获 CUDA图 大 cudagraph 尺寸 【PR号】#48483 【变更类型】此提交为常规更新类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 vllm/v1/worker/gpu_model_runner.py (+5/-1) 【统计数据】共修改 1 个文件，新增 5 行，删除 1 行 【风险评估】高风险 - 涉及核心接口或架构变更，可能影响系统稳定性 【涉及模块】model-runner
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
- **[a02984ed](https://github.com/vllm-project/vllm/commit/a02984ed471488c0f0e8f73cab21be4325992d4c)** 【变更概述】[Perf] 更新：Qwen Replace MoE all-reduce reduce-scatter 【PR号】#47006 【变更类型】此提交为性能优化类型 【变更文件】修改 vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+2/-0); 修改 vllm/model_executor/models/qwen3_5.py (+9/-0); 修改 vllm/model_executor/models/qwen3_next.py (+102/-11) 【统计数据】共修改 3 个文件，新增 113 行，删除 11 行 【风险评估】中风险 - 涉及一定范围的功能修改，需要验证
  - 标签: `performance, model-runner, medium-risk, attention`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/model_executor/models/qwen3_5.py, vllm/model_executor/models/qwen3_next.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/model_executor/models/qwen3_5.py, vllm/model_executor/models/qwen3_next.py

- **[fc1c5480](https://github.com/vllm-project/vllm/commit/fc1c548093029f6487bbdc9c612995dfe7621a75)** 【变更概述】【MRV2】更新：Runtime Draft Weight 更新 Speculative Decoding 【PR号】#46725 【变更类型】此提交为文档更新类型 【变更文件】修改 docs/training/weight_transfer/base.md (+4/-0); 修改 tests/entrypoints/openai/test_openai_schema.py (+1/-0); 修改 tests/v1/worker/test_gpu_worker_weight_transfer.py (+6/-0); 修改 vllm/distributed/weight_transfer/base.py (+18/-0); 修改 vllm/distributed/weight_transfer/sparse_nccl_engine.py (+1/-0); ... 及其他 7 个文件 【统计数据】共修改 12 个文件，新增 129 行，删除 1 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `docs, worker, engine, model-runner, low-risk, distributed`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/distributed/weight_transfer/base.py, vllm/distributed/weight_transfer/sparse_nccl_engine.py, vllm/engine/protocol.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/distributed/weight_transfer/base.py, vllm/distributed/weight_transfer/sparse_nccl_engine.py, vllm/engine/protocol.py, vllm/entrypoints/llm.py, vllm/entrypoints/serve/dev/rlhf/api_router.py

- **[481e481b](https://github.com/vllm-project/vllm/commit/481e481be786c1ca3229e26aa34c15ffd22375af)** 【变更概述】【MRV2】[2/N] 修复：核心 支持 partial prefix 缓存 hit hybrid model 【PR号】#46384 【变更类型】此提交为问题修复类型 【变更文件】新增 tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+816/-0); 修改 tests/v1/core/test_deferred_block_free.py (+62/-0); 修改 tests/v1/core/test_kv_cache_utils.py (+60/-0); 修改 tests/v1/core/test_prefix_caching.py (+4/-1); 修改 tests/v1/core/test_single_type_kv_cache_manager.py (+8/-6); ... 及其他 16 个文件 【统计数据】共修改 21 个文件，新增 1651 行，删除 175 行 【风险评估】中风险 - 涉及一定范围的功能修改，需要验证
  - 标签: `worker, engine, model-runner, config, scheduler, prefix-caching, bugfix, distributed, medium-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 涉及 vLLM 核心代码变更（vllm/config/cache.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py），可能影响 vllm-ascend 的相应模块实现。建议 Ascend 侧关注接口兼容性。
  - 建议测试区域: vllm/config/cache.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py, vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py, vllm/engine/arg_utils.py, vllm/v1/core/block_pool.py

### vllm-ascend
- **[d19628a1](https://github.com/vllm-project/vllm-ascend/commit/d19628a1b292cba1ef33593a6446d3777f28574e)** 变更：[BugFix]Added the store_kv_block_metadata ascendC operator (#11865)
  - 标签: `kernels, spec-decode, bugfix, model-runner, medium-risk, ascend, attention, worker`

---

## 2026-07-11
### vllm
- **[04d553f3](https://github.com/vllm-project/vllm/commit/04d553f390fd37e09ab111936ef1592881299957)** [Misc] Use meta tensor for KV cache stride calculation (#47316)  Signed-off-by: Lucas Wilkinson <lwilkins@redhat.com> Co-authored-by: mergify[bot] <37929162+mergify[bot]@users.noreply.github.com> ---  vllm/v1/worker/gpu/attn_utils.py | 2 +-  1 file changed, 1 insertion(+), 1 deletion(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[9c18e90f](https://github.com/vllm-project/vllm/commit/9c18e90f6c94b90ecdaa99b2230389ba40e0fc69)** [BugFix] Fix packed HND KV cache reshape for FlashAttention (#47314)  Signed-off-by: Lucas Wilkinson <lwilkins@redhat.com> Co-authored-by: mergify[bot] <37929162+mergify[bot]@users.noreply.github.com> ---  vllm/v1/worker/gpu/attn_utils.py | 2 +-  1 file changed, 1 insertion(+), 1 deletion(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

---

## 2026-07-10
### vllm
- **[08dfd686](https://github.com/vllm-project/vllm/commit/08dfd68610d2e05a0d8ddc99c23488da6163df3f)** [Model] Add LongCat-Flash-Lite (n-gram embedding) (#47857)  Signed-off-by: mgoin <mgoin64@gmail.com> ---  CMakeLists.txt                                     |   1 +  csrc/libtorch_stable/ngram_embedding_kernels.cu    |  96 +++++  csrc/libtorch_stable/ops.h                         |   9 +  csrc/libtorch_stable/torch_bindings.cpp            |  13 +  tests/models/registry.py                           |   7 +  tests/models/utils.py                              |   9 +-  vllm/_custom_ops.py                                |  31 ++  vllm/config/speculative.py                         |   2 +-  vllm/config/vllm.py                                |   1 +  vllm/model_executor/models/config.py               |  20 +-  vllm/model_executor/models/longcat_flash.py        |  15 +-  vllm/model_executor/models/longcat_flash_mtp.py    |  18 +-  vllm/model_executor/models/longcat_flash_ngram.py  | 405 +++++++++++++++++++++  vllm/model_executor/models/registry.py             |   4 +  .../model_arch_config_convertor.py                 |   1 +  vllm/v1/worker/gpu/attn_utils.py                   |  10 +-  16 files changed, 630 insertions(+), 12 deletions(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[433f2911](https://github.com/vllm-project/vllm/commit/433f291195ded3ca8d278bc78da9280c5d4e5329)** 【变更概述】【MRV2】[CI] 维护：Right-size test-area timeouts from nightly durations 【PR号】#48186 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 .buildkite/test_areas/attention.yaml (+3/-3); 修改 .buildkite/test_areas/basic_correctness.yaml (+2/-2); 修改 .buildkite/test_areas/benchmarks.yaml (+2/-2); 修改 .buildkite/test_areas/compile.yaml (+11/-11); 修改 .buildkite/test_areas/cuda.yaml (+2/-2); ... 及其他 25 个文件 【统计数据】共修改 30 个文件，新增 183 行，删除 183 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, mrv2`
  - Ascend 影响: ✓ 无影响

- **[95ed0fea](https://github.com/vllm-project/vllm/commit/95ed0feaa5cd7fb16d72c53ce04950aaf07c4698)** 【变更概述】【MRV2】新增：DCP supports hybrid 注意力 【PR号】#40996 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 tests/distributed/test_context_parallel.py (+8/-0); 修改 tests/distributed/test_pynccl.py (+47/-0); 修改 tests/models/language/generation/test_hybrid.py (+13/-1); 修改 tests/models/multimodal/generation/test_vit_cudagraph.py (+1/-0); 修改 tests/test_config.py (+4/-4); ... 及其他 21 个文件 【统计数据】共修改 26 个文件，新增 785 行，删除 127 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, mrv2, distributed, attention, python, model-runner, low-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 可能影响 vllm-ascend 的接口或实现
  - 建议测试区域: vllm/config/model.py, vllm/v1/attention/backend.py, vllm/v1/attention/backends/flash_attn.py, vllm/v1/core/kv_cache_coordinator.py, vllm/v1/core/kv_cache_utils.py, vllm/v1/core/single_type_kv_cache_manager.py, vllm/v1/kv_cache_interface.py, vllm/v1/worker/block_table.py, vllm/v1/worker/cp_utils.py, vllm/v1/worker/gpu_input_batch.py, vllm/v1/worker/gpu_model_runner.py, vllm/v1/worker/tpu_input_batch.py

- **[766469a4](https://github.com/vllm-project/vllm/commit/766469a4c460043ae52cda19b1c52f0dc87e555c)** 【变更概述】[ROCm] 修复：Revert Part `[ROCm 修复 pooling startup workspace lock` 【PR号】#48154 【变更类型】此提交为问题修复类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 vllm/v1/worker/gpu_worker.py (+3/-22) 【统计数据】共修改 1 个文件，新增 3 行，删除 22 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `bugfix, low-risk, mrv2`
  - Ascend 影响: ✓ 无影响

- **[ff8d3488](https://github.com/vllm-project/vllm/commit/ff8d3488f248acc8b5c1d23243723eeb00c74914)** 【变更概述】[Bugfix] 新增：MRV2 Reset num_accepted_tokens add_request all modes 【PR号】#48132 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+2/-1) 【统计数据】共修改 1 个文件，新增 2 行，删除 1 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, mrv2, low-risk, python`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 可能影响 vllm-ascend 的接口或实现
  - 建议测试区域: vllm/v1/worker/gpu/model_states/mamba_hybrid.py

- **[e08a9151](https://github.com/vllm-project/vllm/commit/e08a9151468190575114de1c996275b993ec940a)** 【变更概述】[Bugfix] 更新：Preserve tensor causal metadata grouped 注意力 【PR号】#48135 【变更类型】此提交为日常维护类型 【变更文件】修改 vllm/v1/worker/gpu/attn_utils.py (+4/-2) 【统计数据】共修改 1 个文件，新增 4 行，删除 2 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, low-risk, python`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 可能影响 vllm-ascend 的接口或实现
  - 建议测试区域: vllm/v1/worker/gpu/attn_utils.py

- **[67e7ea89](https://github.com/vllm-project/vllm/commit/67e7ea8977bc4281d4d33dd3a81a5c5fab3df920)** 【变更概述】[ROCm] 维护：CI Set all timeout_in_minutes 180 【PR号】#48146 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 .buildkite/test-amd.yaml (+27/-27) 【统计数据】共修改 1 个文件，新增 27 行，删除 27 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, mrv2`
  - Ascend 影响: ✓ 无影响

---

## 2026-07-09
### vllm
- **[85b3a726](https://github.com/vllm-project/vllm/commit/85b3a7264b6c4e4e89a1c45a2c4ccfd1b8c342dc)** [Bugfix][Model Runner V2] Order uniform decodes first so spec decodes aren't misclassified as prefills (#47381)  Signed-off-by: Woosuk Kwon <woosuk@inferact.ai> Signed-off-by: Nick Hill <nickhill123@gmail.com> Co-authored-by: Claude Fable 5 <noreply@anthropic.com> Co-authored-by: Nick Hill <nickhill123@gmail.com> ---  tests/v1/worker/test_gpu_batch_ordering.py | 70 ++++++++++++++++++++++++++++++  vllm/model_executor/models/deepseek_v2.py  |  7 ++-  vllm/v1/worker/gpu/model_runner.py         | 12 ++++-  3 files changed, 86 insertions(+), 3 deletions(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[1cd75b3d](https://github.com/vllm-project/vllm/commit/1cd75b3dd4b3bf90e4ef81831b6f0dd91fde2fe1)** [Bugfix] Fix race condition in KVBlockZeroer (#48085)  Signed-off-by: Benjamin Chislett <bchislett@nvidia.com> Signed-off-by: Benjamin Chislett <chislett.ben@gmail.com> Co-authored-by: Wentao Ye <44945378+yewentao256@users.noreply.github.com> ---  tests/v1/worker/test_kv_block_zeroer.py | 44 +++++++++++++++++++++++++  vllm/v1/worker/gpu/model_runner.py      |  1 +  vllm/v1/worker/gpu_model_runner.py      |  1 +  vllm/v1/worker/utils.py                 | 57 +++++++++++++++++++++------------  4 files changed, 83 insertions(+), 20 deletions(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[95d6d6f4](https://github.com/vllm-project/vllm/commit/95d6d6f4bba8234088e62124ff20a482acd98714)** 【变更概述】[Bugfix] 更新：Use int8 workspace FlashInfer MLA decode 【PR号】#48046 【变更类型】此提交为问题修复类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 tests/kernels/attention/test_flashinfer_mla_decode.py (+87/-45); 修改 vllm/v1/attention/backends/mla/flashinfer_mla.py (+3/-1); 修改 vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+3/-1) 【统计数据】共修改 3 个文件，新增 93 行，删除 47 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `bugfix, low-risk, mrv2`
  - Ascend 影响: ✓ 无影响

- **[26831949](https://github.com/vllm-project/vllm/commit/26831949b48a0d81fba379dcaf7e378206fd9087)** 【变更概述】[ROCm] 修复：修复 pooling startup workspace lock 【PR号】#47912 【变更类型】此提交为问题修复类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 vllm/v1/attention/backends/rocm_attn.py (+3/-0); 修改 vllm/v1/attention/backends/triton_attn.py (+1/-0); 修改 vllm/v1/attention/ops/triton_prefill_attention.py (+14/-2); 修改 vllm/v1/worker/gpu_worker.py (+22/-3) 【统计数据】共修改 4 个文件，新增 40 行，删除 5 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `bugfix, low-risk, mrv2`
  - Ascend 影响: ✓ 无影响

- **[a5d19cbb](https://github.com/vllm-project/vllm/commit/a5d19cbb95872c4b426c06735733568542fa33db)** 【变更概述】【MRV2】[Core] 更新：Move MRV1 `late_interaction_runner.py` out MRV2 subtree 【PR号】#48014 【变更类型】此提交为日常维护类型 【MRV2关联】此变更涉及 Model Runner V2 核心模块，需重点关注 【变更文件】修改 tests/v1/worker/test_late_interaction_runner.py (+1/-1); 重命名 vllm/v1/pool/late_interaction_runner.py (+0/-0); 修改 vllm/v1/worker/gpu_model_runner.py (+1/-1) 【统计数据】共修改 3 个文件，新增 2 行，删除 2 行 【风险评估】低风险 - 变更范围较小，影响可控
  - 标签: `chore, mrv2`
  - Ascend 影响: ✓ 无影响

---

## 2026-07-08
### vllm
- **[0d12618e](https://github.com/vllm-project/vllm/commit/0d12618e98ff2d21d36081e0e9b4eb23573b6d38)** [Spec Decode] Support hybrid (SWA + full attention) DFlash drafters (#47914)  Signed-off-by: mgoin <mgoin64@gmail.com> Co-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com> ---  vllm/config/vllm.py                                | 17 ++++++++++++++  vllm/model_executor/models/qwen3_dflash.py         | 26 ++++++++++++++--------  vllm/v1/worker/gpu/attn_utils.py                   |  8 ++++---  vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py | 11 ++++-----  .../v1/worker/gpu/spec_decode/dflash/speculator.py | 20 ++++++++++++-----  vllm/v1/worker/gpu/spec_decode/dflash/utils.py     |  7 ++++--  vllm/v1/worker/gpu/spec_decode/dspark/utils.py     |  7 ++++--  vllm/v1/worker/gpu/spec_decode/eagle/utils.py      | 10 ++++++++-  vllm/v1/worker/gpu/spec_decode/speculator.py       |  3 ++-  9 files changed, 79 insertions(+), 30 deletions(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[2afa3f7e](https://github.com/vllm-project/vllm/commit/2afa3f7e950264bb179d030c23a1ed1f46558fd9)** 为 MiniMax-M3 模型支持跨层 allreduce-norm fusion 优化。通过将 MoE 的 cross-rank all-reduce 延迟到下一层的 input_layernorm 中融合执行，减少通信开销。涉及 FusedMoE、DeepseekV2 和 MiniMaxM3 模型。
  - 标签: `performance, medium-risk, model-runner, MoE`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: MiniMax-M3 和 DeepseekV2 模型的 MoE 层 all-reduce 逻辑发生变化，Ascend 侧的模型实现可能需要同步调整以支持 reduce_results 参数和 fused_allreduce_gemma_rms_norm 融合路径。
  - 建议测试区域: vllm/models/minimax_m3/, vllm/models/deepseek_v32/, tests/e2e/

- **[7bd15437](https://github.com/vllm-project/vllm/commit/7bd154375dc505046a6e59e6d8c884a9c6b8fc0f)** [Bugfix] Fix mamba+dflash for MRV2 (#47698) ---  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py | 5 +----  1 file changed, 1 insertion(+), 4 deletions(-)
  - 标签: ``
  - Ascend 影响: ✓ 无影响

- **[dd127d82](https://github.com/vllm-project/vllm/commit/dd127d82ed29c40b7daf6e751add49ff371b1d9d)** [Core][Engine] only materialize tokens when thinking budget is in req (#47053)  Signed-off-by: walterbm <walter.beller.morales@gmail.com>
  - 标签: ``
  - Ascend 影响: ✓ 无影响

---

## 2026-07-07
### vllm
- **[65dcde16](https://github.com/vllm-project/vllm/commit/65dcde16957c26cfd65f581b67787c871cea3206)** 修复PD分离+MTP在Qwen3.5(GDN)上的正确性问题。主要修改包括：1) 在combine_sampled_and_draft_tokens kernel中正确处理prompt-tail slots，只重写generated-token slots；2) 修复mamba_hybrid中spec-decode行的检测逻辑，使用num_scheduled_tokens == num_draft_tokens + 1判断decode行；3) 移除states.py中add_request对last_sampled_tokens的冗余写入。
  - 标签: `bugfix, pd-disagg, mtp, high-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu/input_batch.py中_combine_sampled_and_draft_tokens_kernel的token写入逻辑，vllm/worker/gpu/model_states/mamba_hybrid.py中spec-decode行检测逻辑，以及vllm/v1/worker/gpu/states.py中add_request的last_sampled_tokens写入。vllm-ascend的NPUModelRunner如果重写了这些方法或依赖这些行为，需要同步更新。
  - 建议测试区域: vllm_ascend/worker/model_runner_v1.py, vllm_ascend/worker/worker.py

- **[d3e69fd6](https://github.com/vllm-project/vllm/commit/d3e69fd6714e9d1bb6e8e4f03157090dc32e7960)** 使用blocking CUDA events避免忙等CUDA driver lock。在async_utils.py、spec_decode/utils.py、gpu_model_runner.py中，将torch.cuda.Event()改为torch.cuda.Event(blocking=True)，使CPU线程在等待GPU事件时进入睡眠状态而不是忙等，减少CUDA driver锁竞争。
  - 标签: `performance, cuda, medium-risk`
  - Ascend 影响: ✓ 无影响

- **[2f71b2bd](https://github.com/vllm-project/vllm/commit/2f71b2bd9f693f06973a8abf3823f095bd46ef69)** 在V2 runner中对齐混合encoder-decoder KV cache视图。当decoder self-attention（K/V-first布局）和cross-attention（blocks-first布局）共享同一raw allocation时，通过_align_mixed_attention_kv_cache_views函数重新调整blocks-first视图的stride，使其与K/V-first存储布局兼容。
  - 标签: `bugfix, rocm, attention, medium-risk`
  - Ascend 影响: ✓ 无影响

- **[69f31509](https://github.com/vllm-project/vllm/commit/69f3150981e4bc8a09439eb7cae0095605b964b4)** 修复XPU上PP（流水线并行）的精度问题。在broadcast方法中，在切换到broadcast stream之前同步main stream，确保所有操作在broadcast前完成。
  - 标签: `bugfix, xpu, pipeline-parallel, low-risk`
  - Ascend 影响: ✓ 无影响

- **[567a7843](https://github.com/vllm-project/vllm/commit/567a78432d8e00ac17e4288a4c15ca795c6a1bb4)** 修复DP+MTP hang问题。当DP各rank对input_fits_in_drafter判断不一致时，通过dummy_run防止hang。重构了draft proposal逻辑，将GPU token和CPU token的draft proposal路径分离，确保DP rank在drafter执行上保持一致。
  - 标签: `bugfix, data-parallel, mtp, speculative-decoding, high-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu_model_runner.py中propose_draft_token_ids逻辑，新增了dummy_run调用和draft_after_bookkeeping路径。vllm-ascend的NPUModelRunner如果重写了draft proposal逻辑，需要同步更新以处理DP rank不一致的情况。
  - 建议测试区域: vllm_ascend/worker/model_runner_v1.py, vllm_ascend/patch/worker/spec_decode_patch.py

- **[04adc884](https://github.com/vllm-project/vllm/commit/04adc8843bbe0711fed8edf50d6d4cd4fca400e7)** 修复DeepSeek-V4 fp8_ds_mla KV cache reshape问题。在get_kv_cache_spec中传递kv_quant_mode参数，确保在_reshape_kv_cache_tensors中能正确获取cache_dtype_str。修复了当kv_cache_spec有cache_dtype_str属性时优先使用该值而不是self.cache_config.cache_dtype。
  - 标签: `bugfix, deepseek, kv-cache, mla, medium-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu_model_runner.py中_reshape_kv_cache_tensors方法，优先使用kv_cache_spec的cache_dtype_str属性。vllm-ascend的NPUModelRunner如果重写了KV cache reshape逻辑，需要同步更新。同时修改了vllm/models/deepseek_v4/attention.py和vllm/v1/attention/backends/mla/sparse_swa.py中get_kv_cache_spec方法，新增kv_quant_mode参数。
  - 建议测试区域: vllm_ascend/worker/model_runner_v1.py, vllm_ascend/attention/attention_v1.py

---

## 2026-07-06
### vllm
- **[5bce653e](https://github.com/vllm-project/vllm/commit/5bce653e09ca62c870ea18d01a4180dc48d3bacb)** 该commit对Transformers建模后端进行了重大性能优化，通过引入图融合（fuser）机制，将HF模型中的GLU（gate+up投影）、QKV投影和RMSNorm自动检测并融合为vLLM的原生算子（MergedColumnParallelLinear、QKVParallelLinear、RMSNorm/GemmaRMSNorm），使Transformers后端的性能与原生vLLM模型实现一致。核心实现包括：1) 新增`fuser.py`和`fusers/`子模块，包含`GLUFuser`、`QKVFuser`、`RMSNormFuser`等具体融合器；2) 新增`fx_utils.py`提供FX图追踪和AST源码重写引擎；3) 修改`base.py`中的`recursive_replace`方法，使用融合器替换子模块；4) MoE模块也支持通过`MoEBlockFuser`进行融合；5) 移除了旧的`replace_rms_norm_class`函数。该变更大幅提升了Transformers后端的推理速度，同时保持了与原生vLLM相同的精度。
  - 标签: `feature, performance, model-runner, transformers, high-risk`
  - Ascend 影响: ✓ 无影响

- **[07f9baf7](https://github.com/vllm-project/vllm/commit/07f9baf7564b42ba7218ce9167bfcc4128028473)** 该commit回退了之前将`torch.cuda.Event`替换为`torch.Event`的变更（PR #47140）。原因是`torch.Event`在某些平台上（如XPU）可能不兼容，导致`RuntimeError: dummy base class`。回退后，所有使用`torch.Event`的地方恢复为`torch.cuda.Event`，同时在XPU平台上通过`torch.cuda.Event = torch.xpu.Event`进行适配。涉及多个模块，包括benchmarks、分布式通信、KV transfer、LoRA、MoE、DeepSeek V4 attention等。
  - 标签: `bugfix, refactor, distributed, attention, lora, high-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响。`torch.cuda.Event`的回退影响多个模块。vllm-ascend的`NPUWorker`、`NPUModelRunner`、`AscendAttentionBackend`等组件中如果使用了`torch.Event`（来自PR #47140的变更），需要确认是否已回退为`torch.cuda.Event`。vllm-ascend的`NPUPlatform`中可能已有对`torch.cuda.Event`的适配（如重定向到`torch.npu.Event`），需确保兼容。

- **[f2aaf591](https://github.com/vllm-project/vllm/commit/f2aaf5915102cd56b3a60f8e6e59c4a7f31268dd)** 该commit为Bailing混合模型添加了MTP（Multi-Token Prediction）推测解码支持。主要变更包括：1) 新增`bailing_moe_mtp.py`模型实现，包含`BailingMoeV25MTPModel`、`BailingMoeV25MultiTokenPredictor`和`BailingMoeV25MultiTokenPredictorLayer`；2) 在`linear_attn.py`中添加`BailingLinearAttentionBackend`和`BailingLinearAttentionMetadataBuilder`，支持MTP下的spec decode metadata构建；3) 在`bailing_linear_attn.py`中添加`bailing_linear_attention_decode_spec` Triton kernel，支持多步draft token的线性注意力解码；4) 在`speculative.py`中添加Bailing模型的hf_config_override；5) 在`model_arch_config_convertor.py`中添加`BailingHybridMTPModelArchConfigConvertor`。
  - 标签: `feature, spec-decode, bailing, attention, high-risk`
  - Ascend 影响: ✓ 无影响

---

## 2026-07-05
### vllm
- **[cc1d020d](https://github.com/vllm-project/vllm/commit/cc1d020d01949d11b7ef70dabb0eb196b3f39f53)** 该commit为MRV2（多模态推理V2）启用mm prefix双向注意力支持。核心变更包括：1) 在`vllm/v1/worker/gpu/attn_utils.py`中新增`compute_mm_prefix_ranges`函数，用于计算多模态token的PrefixLM双向注意力范围，并支持滑动窗口过滤；2) 修改`build_attn_metadata`函数，新增`mm_req_doc_ranges`参数，将多模态前缀范围传递给注意力元数据构建器；3) 在`vllm/v1/worker/gpu/model_states/default.py`的`prepare_attn`方法中，当模型支持多模态输入且为mm_prefix_lm时，调用`compute_mm_prefix_ranges`计算范围并传递给`build_attn_metadata`；4) 将`vllm/config/model.py`中的`is_mm_prefix_lm`属性从`@property`改为`@cached_property`，避免重复计算；5) 新增测试文件`tests/models/multimodal/generation/test_mm_prefix_lm.py`，验证Gemma3的prefix-LM mask正确性。实现方式是通过在注意力元数据构建时传入多模态token的文档范围，使注意力后端能够为这些token应用双向注意力mask。潜在风险：新增的`mm_req_doc_ranges`参数需要所有注意力后端的元数据构建器支持，否则可能导致兼容性问题。
  - 标签: `feature, medium-risk, attention, model-runner, multimodal`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) `vllm/v1/worker/gpu/attn_utils.py`中`build_attn_metadata`函数新增`mm_req_doc_ranges`参数，AscendAttentionBackend的元数据构建器（`AscendAttentionMetadataBuilder`）需要支持该参数以正确处理多模态前缀双向注意力。2) `vllm/v1/worker/gpu/attn_utils.py`中新增`compute_mm_prefix_ranges`函数，NPUModelRunner在`prepare_attn`方法中需要调用此函数计算多模态前缀范围。3) `vllm/v1/worker/gpu/model_states/default.py`中`prepare_attn`方法新增对`compute_mm_prefix_ranges`的调用逻辑，NPUModelRunner需要同步更新其注意力准备逻辑。4) `vllm/config/model.py`中`is_mm_prefix_lm`从`@property`改为`@cached_property`，NPUPlatform或相关配置补丁需要确保缓存行为一致。
  - 建议测试区域: tests/models/multimodal/generation/test_mm_prefix_lm.py

- **[b6cc46ec](https://github.com/vllm-project/vllm/commit/b6cc46ec3b903c71405f4355c1e9ecb47ae54bb2)** 该commit支持无需数据并行（DP）的序列并行，实现1.9%~5.0%的端到端吞吐量提升。核心变更包括：1) 在`vllm/config/parallel.py`中移除`use_sequence_parallel_moe`对`data_parallel_size > 1`的依赖，使序列并行可以在无DP时独立启用；2) 在`vllm/distributed/device_communicators/base_device_communicator.py`中，all2all管理器的初始化条件扩展为包含`use_sequence_parallel_moe`，确保序列并行EP时也初始化all2all通信；3) 在`vllm/forward_context.py`中，DPMetadata的创建逻辑扩展为支持序列并行场景，当无DP时直接使用本地token数量；4) 在`vllm/model_executor/layers/fused_moe/config.py`和`runner/moe_runner.py`中，all2all kernel和naive dispatch/combine的条件扩展为包含序列并行；5) 在`vllm/model_executor/models/deepseek_v2.py`中优化了序列并行的padding逻辑，使用更简洁的`(-hidden_states.shape[0]) % tp_world_size`计算padding大小；6) 在`vllm/model_executor/models/gpt_oss.py`中，当使用LoRA时禁用序列并行。这是一个性能优化变更，涉及分布式通信、MoE层和模型执行等多个模块。潜在风险：序列并行与LoRA的兼容性需要额外关注。
  - 标签: `performance, medium-risk, distributed, moe, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) `vllm/distributed/device_communicators/base_device_communicator.py`中all2all管理器初始化条件变更，NPUCommunicator的`__init__`方法需要同步更新，确保在`use_sequence_parallel_moe`为True时也初始化all2all通信。2) `vllm/forward_context.py`中DPMetadata创建逻辑变更，Ascend的分布式上下文设置（如`set_forward_context`的patch）需要支持无DP时的序列并行场景。3) `vllm/config/parallel.py`中`use_sequence_parallel_moe`属性变更，NPUPlatform或相关配置补丁需要确保序列并行可以在无DP时独立启用。4) `vllm/model_executor/layers/fused_moe/config.py`和`runner/moe_runner.py`中all2all和dispatch/combine条件变更，Ascend的MoE实现需要同步更新这些条件判断。
  - 建议测试区域: tests/distributed/test_sequence_parallel.py, tests/models/test_moe.py

- **[fa4321de](https://github.com/vllm-project/vllm/commit/fa4321de3d894c50c5ca0766dffa352d3fb07423)** 该commit修复TurboQuant注意力后端中KV cache dtype丢失的问题。变更内容：在`vllm/v1/worker/gpu/attn_utils.py`的`_reshape_kv_cache`和`_update_hybrid_attention_layout`函数中，当KV cache spec是`TQFullAttentionSpec`类型时，即使`kv_quant_mode`为`NONE`，也强制使用`cache_dtype`而非`"auto"`。这是因为TurboQuant后端需要明确的dtype信息来正确计算KV cache形状。这是一个低风险的bug修复，仅影响TurboQuant注意力后端。
  - 标签: `bugfix, low-risk, attention, quantization`
  - Ascend 影响: ✓ 无影响

---

## 2026-07-04
### vllm
- **[07516fda](https://github.com/vllm-project/vllm/commit/07516fda67d2133e26c0fd7386c0b0c8641e2a6e)** 该 commit 使 Dynamic Speculative Decoding (DSD) 兼容 Full CUDA Graphs（MRv2）。主要变更：1) 在 CudaGraphManager._init_candidates 中，当使用 DSD 时，会为每个可能的 decode query length（来自 num_speculative_tokens_per_batch_size 调度表）捕获 FULL decode graph；2) 在 CompilationConfig.resolve_cudagraph_mode_and_sizes 中新增 use_v2_model_runner 参数，MRv2 不再调整 cudagraph capture sizes；3) 移除了 VllmConfig 中 DSD 对 MRv2 的限制；4) 更新了文档，说明 Full Cudagraph 仅支持 MRv2；5) 新增了全面的测试用例。潜在风险：变更涉及 CUDA Graph 捕获逻辑的核心部分，但测试覆盖了各种边界情况。
  - 标签: `feature, high-risk, spec-decode, cuda-graph, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 vllm-ascend 的编译和模型运行模块。1) CompilationConfig.resolve_cudagraph_mode_and_sizes 新增了 use_v2_model_runner 参数，NPUModelRunner 在调用此方法时需要传递此参数（已在 MRv2 路径中传递 True）。2) CudaGraphManager._init_candidates 中 Dynamic SD 的 FULL graph 捕获逻辑发生变更，vllm-ascend 的 ACLGraphWrapper 需要同步更新其候选图生成逻辑以支持 DSD 的多个 decode query length。3) VllmConfig._maybe_override_dynamic_sd_cudagraph_mode 中新增了 use_v2_model_runner 检查，NPUPlatform 需要确认其编译配置是否受影响。
  - 建议测试区域: vllm_ascend/compilation/acl_graph.py, vllm_ascend/worker/model_runner_v1.py

- **[67ff0ae3](https://github.com/vllm-project/vllm/commit/67ff0ae30fe6b1ab1a912e10977d99ddb169c4b2)** 该 commit 支持了 nvfp4 KV cache 与 kv-cache-dtype-skip-layers 和 sliding_window 的组合使用。主要变更：1) 在 CacheConfig 中新增 skip_page_size_padded 字段；2) 在 AttentionLayer.get_kv_cache_spec 中，为 sliding window 层选择最大的 kernel block size 以适配 padded page；3) 在 Platform 基类中新增 _align_heterogeneous_kv_block_size 方法，用于对齐不同 KV dtype 的 block size；4) 在 GPUModelRunner 的 KV cache reshape 逻辑中，为 skip layers 使用 'auto' cache dtype；5) 更新了文档。潜在风险：变更涉及 KV cache 分配的核心逻辑，但通过 _align_heterogeneous_kv_block_size 方法进行了统一处理。
  - 标签: `feature, medium-risk, kv-cache, quantization, attention`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 vllm-ascend 的多个模块。1) Platform 基类新增了 _align_heterogeneous_kv_block_size 方法，NPUPlatform 需要检查是否需要覆盖此方法以支持 Ascend NPU 上的异构 KV cache block size 对齐。2) AttentionLayer.get_kv_cache_spec 中 sliding window 的 block_size 选择逻辑变更，AscendAttentionBackend 需要确认其 get_supported_kernel_block_sizes 返回的值是否兼容新的选择逻辑。3) GPUModelRunner._reshape_kv_cache_tensors 中新增了 per-layer cache_dtype 逻辑，NPUModelRunner 需要同步更新其 KV cache reshape 逻辑。4) attn_utils.py 中的 _reshape_kv_cache 和 _update_hybrid_attention_layout 也新增了 per-layer cache_dtype 逻辑，vllm-ascend 的 attention 相关 patch 需要检查。
  - 建议测试区域: vllm_ascend/platform.py, vllm_ascend/worker/model_runner_v1.py, vllm_ascend/attention/attention_v1.py

---

## 2026-07-03
### vllm
- **[3775d5fc](https://github.com/vllm-project/vllm/commit/3775d5fcabf7bc5d4d92768485d860d132c6e1b6)** 为ROCm平台添加新的CI测试分组，包括MRCR评估、vLLM IR测试、KDA kernel测试、Model Runner V2系列测试（core/examples/distributed/PP/spec decode）、GGUF插件测试等。同时修复了Triton kernel在AMD后端上的num_stages兼容性问题（chunk_delta_h kernel在ROCm上不支持num_stages=4）。这是一个纯CI和ROCm平台适配的变更。
  - 标签: `ci, test, rocm`
  - Ascend 影响: ✓ 无影响

- **[979f5511](https://github.com/vllm-project/vllm/commit/979f5511d78b317760d45df9290233c27793a0af)** 修复Gemma4模型中图像双向注意力超出滑动窗口的问题。主要变更：1) 在Attention类中添加mm_prefix_clamp_sliding_window属性；2) Gemma4模型在滑动层上设置此属性为True；3) Gemma4ForConditionalGeneration类设置mm_prefix_clamp_sliding_window=True；4) 在FlashAttention和TritonAttention后端中实现滑动窗口钳制逻辑；5) GPUModelRunner中根据此属性决定是否跳过超出滑动窗口的mm_prefix范围。
  - 标签: `bugfix, attention, model`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: Attention类新增mm_prefix_clamp_sliding_window属性，AscendAttentionBackend需要处理此属性。GPUModelRunner中mm_prefix逻辑变更（新增_clamps_in_kernel判断），NPUModelRunner需要同步更新。FlashAttention和TritonAttention后端中的mm_prefix_clamp_sliding_window实现逻辑需要AscendAttentionBackend参考实现。
  - 建议测试区域: vllm_ascend/tests/models/test_gemma4.py, vllm_ascend/tests/attention/test_attention_backend.py

- **[276b837d](https://github.com/vllm-project/vllm/commit/276b837dc4d6a15ec7a82099dccd4c997eec916b)** 修复ModelRunner V2在shutdown时未释放所有模型引用的问题。在shutdown方法中添加了删除model_state和speculator引用的逻辑，确保模型权重被正确释放。
  - 标签: `bugfix, model-runner, memory`
  - Ascend 影响: ✓ 无影响

---

## 2026-07-02
### vllm
- **[a47f38f8](https://github.com/vllm-project/vllm/commit/a47f38f82569c236d7d23b7ad0c8792ac6d62247)** 修复推测解码中block verification kernel的int32偏移溢出问题。当词表较大（如GLM约155k）时，logit_idx * vocab_stride会超过int32范围。变更将Triton kernel中的索引变量（req_state_idx, start_idx, logit_idx）显式转换为int64，避免乘法溢出。同时添加了测试用例验证高索引位置下的正确性。
  - 标签: `bugfix, low-risk, spec-decode, kernel`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py中的Triton kernel。vllm-ascend的推测解码补丁（vllm_ascend.patch.worker.spec_decode_patch）如果覆盖了此文件，需要同步更新kernel中的int64转换逻辑。

- **[3e158ae6](https://github.com/vllm-project/vllm/commit/3e158ae62d1c227004fa9f702a51126e58ebbcb2)** 修复Mamba2模型在非推测解码模式下崩溃的问题。在prepare_attn方法中，num_accepted_tokens的创建逻辑仅在推测解码启用时（num_speculative_tokens > 0）才执行，避免了非推测解码场景下访问未初始化变量的错误。
  - 标签: `bugfix, low-risk, model-runner, mamba`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu/model_states/mamba_hybrid.py。vllm-ascend的NPUModelRunner如果使用了MambaHybridModelState或相关逻辑，需确保此修复已同步。

- **[a2f71300](https://github.com/vllm-project/vllm/commit/a2f713002df9fd08c0fe13272c76547421721f2d)** 默认启用V2 Model Runner（对所有dense模型）。变更修改了DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES，移除了显式列出的架构（如LlamaForCausalLM），改为通过_is_default_v2_model_runner_model属性判断：非MoE、非hybrid、非attention-free的generate类型模型默认启用V2。同时更新了相关测试用例，处理V2下async scheduling的max_concurrent_batches计算差异，以及V2下resumed request作为NewRequestData而非cached request的调度行为变化。
  - 标签: `feature, medium-risk, model-runner, config`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: V2 Model Runner默认启用影响所有模型执行路径。vllm-ascend的NPUModelRunner继承自GPUModelRunner，需确保：1) V2下的输入准备、模型执行、采样流程兼容；2) async scheduling的max_concurrent_batches计算逻辑（V2下+1）已适配；3) scheduler中resumed request的处理方式（NewRequestData vs cached request）已同步。vllm-ascend的补丁（vllm_ascend.patch.worker.engine_core_patch, scheduler_patch等）需检查是否覆盖了这些变更。
  - 建议测试区域: vllm_ascend/tests/test_model_runner.py, vllm_ascend/tests/test_scheduler.py

- **[2b753ad2](https://github.com/vllm-project/vllm/commit/2b753ad200d52a2dc16e61ff3c92a45711e2750c)** 为DSpark推测器添加checkpoint支持。支持speculators格式的checkpoint，其中draft_vocab_size可以小于target vocab_size，并包含d2t/t2d remap表。修改了DSparkMarkovHead的markov_w2输出维度为draft_vocab_size，添加了compute_draft_logits和map_draft_to_target方法，以及d2t scatter逻辑用于probabilistic rejection sampling。同时添加了update_dspark配置转换函数。
  - 标签: `feature, medium-risk, spec-decode, model`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 修改了vllm/v1/worker/gpu/spec_decode/dspark/speculator.py和多个模型文件。vllm-ascend的推测解码补丁（vllm_ascend.patch.worker.spec_decode_patch）需同步更新DSparkSpeculator的实现，特别是：1) load_draft_model中的d2t scatter逻辑；2) _sample_sequential中的compute_draft_logits和map_draft_to_target调用；3) dspark_bonus_anchor配置处理。

---

## 2026-07-01
### vllm
- **[f5a8d733](https://github.com/vllm-project/vllm/commit/f5a8d73377d0f0a4e00cba172f9fbd0d50471b07)** 新增 DSpark 推测解码支持，这是一种半自回归并行草稿生成方法。DSpark 在单个并行前向中生成整个 token 块（类似 DFlash），然后通过轻量级顺序 Markov 头注入块内依赖。变更涉及多个模块：1) 配置层新增 'dspark' 方法类型和 use_dspark() 方法；2) 新增 Qwen3DSparkModel 和 DSparkDeepseekV4ForCausalLM 模型实现；3) Scheduler 中 num_lookahead_tokens 计算逻辑调整；4) GPUModelRunner 中 speculative_config.method 检查新增 'dspark'；5) 新增 DSparkSpeculator 类，继承自 DFlashSpeculator；6) 稀疏 SWA 注意力构建器支持非因果索引；7) 模型注册表新增 DSparkDraftModel 和 Qwen3DSparkModel 条目。这是一个高风险的大规模 feature 变更，涉及推测解码核心流程。
  - 标签: `feature, high-risk, spec_decode, scheduler, attention, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) Scheduler 中 num_lookahead_tokens 计算逻辑新增 DSpark 分支（self.num_lookahead_tokens = self.num_spec_tokens），vllm-ascend 的 scheduler_patch 需要同步更新。2) GPUModelRunner 中 speculative_config.method 检查新增 'dspark'，vllm-ascend 的 block_table_patch 需要更新。3) SpeculativeConfig 新增 use_dspark() 方法，vllm-ascend 的 spec_decode_patch 需要评估。4) SparseSWAMetadataBuilder 新增非因果索引构建逻辑和 is_dspark 标志，vllm-ascend 的 attention_backend_patch 需要评估。5) DFlashSpeculator 的 _prepare_dflash_inputs_kernel 新增 SAMPLE_FROM_ANCHOR 参数和 max_model_len 参数，vllm-ascend 的 spec_decode_patch 需要同步更新。
  - 建议测试区域: vllm_ascend/patch/platform/scheduler_patch, vllm_ascend/patch/worker/block_table_patch, vllm_ascend/patch/worker/spec_decode_patch, vllm_ascend/patch/worker/attention_backend_patch

- **[e7d0fcbc](https://github.com/vllm-project/vllm/commit/e7d0fcbc0954382f10fb4c9cee1df6f3a16113e8)** 修复 main 分支上的多个 CI 失败问题。包括：1) 权重传输测试中移除平台条件判断，始终设置 Ray 环境变量；2) Mamba prefix cache 测试中添加 load_format='dummy'；3) 修复 fused_moe 中 weight_loader 调用错误；4) DeepSeek-V2 模型中添加 residual 连续性保证；5) Gemma3 多模态编码器 CUDA Graph 捕获接口添加 path 参数。这是一个中等风险的 bugfix 集合。
  - 标签: `bugfix, medium-risk, ci, distributed, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) DeepSeek-V2 模型中新增 residual.contiguous() 调用，vllm-ascend 的 model_registry_patch 需要评估是否需要在 Ascend 实现中做同样处理。2) fused_moe 中 weight_loader 调用修复（self.weight_loader -> param.weight_loader），vllm-ascend 的 MoE 相关补丁需要评估。

- **[77a9c5ae](https://github.com/vllm-project/vllm/commit/77a9c5ae28a3d054e6caf60c7e14082453b3ae47)** 权重同步系统重构。主要变更包括：1) 将稀疏 NCCL 引擎从密集 NCCL 引擎中分离为独立的 SparseNCCLWeightTransferEngine；2) 简化 WeightTransferEngine 基类，移除 receive_sparse_weights 和 trainer_send_sparse_weights 方法，将 layerwise reload 生命周期移到 start_weight_update/finish_weight_update 中；3) Worker 端的权重更新流程简化，移除稀疏补丁应用逻辑；4) 新增 nccl_common.py 共享 NCCL 初始化逻辑；5) 更新所有示例和文档。这是一个高风险的大规模 refactor。
  - 标签: `refactor, high-risk, distributed, weight-transfer`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) WeightTransferEngine 基类构造函数签名变更：parallel_config 参数替换为 vllm_config + device，vllm-ascend 的 executor_patch 和 worker_base_patch 需要同步更新。2) WeightTransferEngine 新增 start_weight_update/finish_weight_update 抽象方法，vllm-ascend 的 worker_base_patch 需要实现这些方法。3) GPUModelRunner 中移除了 apply_sparse_weight_patches 方法，vllm-ascend 的 block_table_patch 需要移除相关引用。4) GPUWorker 中 update_weights/finish_weight_update 逻辑简化，vllm-ascend 的 worker_base_patch 需要同步更新。5) WeightTransferEngineFactory.create_engine 签名变更，vllm-ascend 的 executor_patch 需要更新。
  - 建议测试区域: vllm_ascend/patch/worker/worker_base_patch, vllm_ascend/patch/worker/executor_patch, vllm_ascend/patch/worker/block_table_patch

- **[9969466a](https://github.com/vllm-project/vllm/commit/9969466a597810db6e06b4942dd6cc2086885ee2)** 为 MiMo 模型添加 SWA（滑动窗口注意力）+ DFlash 推测解码支持。主要变更包括：1) qwen3_dflash.py 中新增 _resolve_layer_attention 函数，支持从配置中解析每层的滑动窗口和因果性；2) DFlashQwen3Attention 支持滑动窗口和 attention_sink_bias；3) DFlashQwen3Model 支持独立的 mask_embedding；4) MiMoV2Model 添加 EagleModelMixin 支持；5) FlashAttentionMetadata 新增 sliding_window 字段，支持非因果滑动窗口的对称化。这是一个中等风险的 feature。
  - 标签: `feature, medium-risk, spec_decode, attention, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) FlashAttentionMetadata 新增 sliding_window 字段，vllm-ascend 的 attention_backend_patch 需要评估是否需要同步更新。2) FlashAttentionMetadataBuilder 中新增 _maybe_symmetrize_window 函数，vllm-ascend 的 attention_backend_patch 需要评估。3) DFlashQwen3ForCausalLM 中新增 _read_mask_embedding 方法，vllm-ascend 的 spec_decode_patch 需要评估。

---

## 2026-06-30
### vllm
- **[e840f0d3](https://github.com/vllm-project/vllm/commit/e840f0d3f5d26803e907d64a84be521d9568900a)** 将项目中所有 `torch.cuda.Event` 替换为 `torch.Event`，并添加 pre-commit 检查禁止新的 `torch.cuda.Event` 使用。这是平台抽象化工作的一部分，使代码能在非 CUDA 设备上运行。变更涉及 130 个文件，包括 benchmarks、测试、分布式 KV 传输、LoRA、模型层、采样器、spec decode 等多个模块。
  - 标签: `refactor, platform, low-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 直接影响 Ascend 平台。vllm-ascend 的 `vllm/v1/worker/xpu_model_runner.py` 中原本有 `torch.cuda.Event = torch.Event` 的 patch，该 patch 需要移除。同时 vllm-ascend 的 worker 和 attention 模块中如果使用了 `torch.cuda.Event`，需要替换为 `torch.Event`。
  - 建议测试区域: vllm_ascend/worker/, vllm_ascend/attention/, vllm_ascend/patch/

- **[db808b39](https://github.com/vllm-project/vllm/commit/db808b39614384a0349378268a46a1a0feabcec3)** 实现 block verification 拒绝采样方法（Sun et al., 2024）。新增 `use_block_verification` 参数，在 `rejection_sample` 中实现 block verification 逻辑：1) 计算累积联合比率 `cumulative_log_p`；2) 计算残差质量 `residual_mass`；3) 使用 block verification 阈值 `h` 决定接受长度；4) 在 resample 阶段根据 block verification 调整目标分布。新增多个 Triton kernel 支持这些计算。同时更新 `SpeculativeConfig` 支持 `"block"` 作为 `rejection_sample_method`。
  - 标签: `feature, spec-decode, sampler, high-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 Ascend 的投机解码模块。vllm-ascend 的 `vllm_ascend/spec_decode/` 模块实现了投机解码的 Ascend 适配，新增的 block verification 方法需要 Ascend 实现对应的 Triton kernel 或使用 CPU fallback。`RejectionSampleMethod` 类型新增 `"block"` 值，影响 `SpeculativeConfig` 的解析。
  - 建议测试区域: vllm_ascend/spec_decode/, vllm_ascend/sample/

- **[8cc24233](https://github.com/vllm-project/vllm/commit/8cc242335de805cac390580f0dcd9e69b6ed86c0)** 优化XPU Worker的关闭逻辑，防止资源泄漏。主要变更包括：1) 在XPUPlatform.check_and_update_config中，当shutdown_timeout为0时自动设置为5秒，确保oneCCL/Level Zero资源有足够时间释放；2) 在GPUModelRunner.shutdown()中，将ROCm特定的内存清理逻辑扩展为同时适用于ROCm和XPU；3) 在GPUWorker.shutdown()中，将CuMemAllocator的release_pools调用限制在cuda_alike平台，避免XPU平台错误调用；4) 新增XPUWorker.shutdown()方法，调用父类shutdown后释放XpuMemAllocator的内存池；5) 更新测试工具以支持XPU平台的内存查询；6) 调整CI配置以包含新的测试。这是一个低风险的优化，主要影响XPU平台的资源管理。
  - 标签: `performance, low-risk, worker, xpu, resource-management`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响GPUModelRunner.shutdown()和GPUWorker.shutdown()方法。GPUModelRunner.shutdown()中增加了对XPU平台的内存清理逻辑（gc.collect + empty_cache + synchronize），NPUModelRunner继承自GPUModelRunner，需要确认Ascend平台是否也需要类似的清理。GPUWorker.shutdown()中将CuMemAllocator.release_pools()限制在cuda_alike平台，NPUWorker继承自WorkerBase而非GPUWorker，但NPUWorker有自己的shutdown逻辑，需要确认是否也需要类似的平台条件判断。

- **[fb42e521](https://github.com/vllm-project/vllm/commit/fb42e5219edcbce66fb1e758c004e610e618f70a)** 将 `torch.cuda.mem_get_info` 替换为 `torch.accelerator.get_memory_info`。这是平台抽象化工作的一部分，使内存查询代码能在非 CUDA 设备上运行。变更涉及测试、模型层、worker、内存工具等多个模块。同时更新 pre-commit 检查，禁止新的 `torch.cuda.mem_get_info` 和 `current_platform.mem_get_info` 使用。从 CPU 平台类中移除 `mem_get_info` 方法。
  - 标签: `refactor, platform, medium-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 直接影响 Ascend 平台。vllm-ascend 的 `vllm/v1/worker/xpu_model_runner.py` 中原本有 `torch.cuda.mem_get_info = torch.xpu.mem_get_info` 的 patch，该 patch 需要移除。同时 vllm-ascend 中如果使用了 `current_platform.mem_get_info`，需要替换为 `torch.accelerator.get_memory_info`。
  - 建议测试区域: vllm_ascend/worker/, vllm_ascend/patch/

- **[61ab70ec](https://github.com/vllm-project/vllm/commit/61ab70ec3bd13dd422b86f3b80207d322994a5e7)** V2 Model Runner 支持 mamba hybrid 模型的 align prefix cache。主要变更：1) 移除 V2 runner 对 align mamba cache mode 的限制；2) 在 `MambaHybridModelState` 中添加 `preprocess_state` 和 `postprocess_state` 方法，实现 GPU 上的 align 状态迁移；3) 新增 `preprocess_mamba_align_fused_kernel` 和 `precopy_mamba_align_fused_kernel` Triton kernel；4) 重构 `postprocess_mamba_fused_kernel` 支持 V2 的 idx_mapping 和预计算的新 computed tokens；5) 更新 `MambaSpecDecodeGPUContext` 添加 `run_fused_precopy` 和 `run_fused_postprocess_align` 方法；6) 更新 warmup 逻辑为 align 模式预留额外 block。
  - 标签: `feature, model-runner, mamba, spec-decode, high-risk`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 影响 Ascend 的 model runner 和 mamba 模块。`MambaHybridModelState` 新增 `preprocess_state` 和 `postprocess_state` 方法，`ModelSpecificState` 接口新增 `preprocess_state` 方法。`postprocess_state` 的签名已更改，新增 `num_computed_tokens` 参数。`MambaSpecDecodeGPUContext` 新增 `run_fused_precopy` 和 `run_fused_postprocess_align` 方法。vllm-ascend 的 `NPUModelRunner` 如果使用了 mamba 模型，需要同步更新。
  - 建议测试区域: vllm_ascend/worker/, vllm_ascend/ops/mamba/

---

## 2026-06-29
### vllm
- **[a2abce64](https://github.com/vllm-project/vllm/commit/a2abce646f7db07f2169dfc59433d4128bc404de)** 修复EPLB负载记录中的padding token问题。核心变更：1) 在`EplbState`中新增`num_unpadded_tokens_tensors`列表，记录每个ubatch中真实（非padding）token的数量；2) 在`base_router.py`的Triton kernel中新增`HAS_NUM_UNPADDED`常量，当提供时跳过padding token的负载记录；3) 新增`EplbState.prepare_forward`方法，在每次前向传播前更新unpadded token计数；4) 在`GPUModelRunner`和spec decode的多个speculator中调用`prepare_forward`；5) 新增`compute_hash_cached`工具函数缓存config hash；6) 更新测试用例。
  - 标签: `bugfix, medium-risk, distributed`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) `EplbLayerState`新增`num_unpadded_tokens_tensors`字段，影响所有使用EPLB的MoE层。vllm-ascend中如果有自定义EPLB实现，需要同步更新。2) `eplb_map_to_physical_and_record`函数新增`num_unpadded_tokens`参数，影响路由器的EPLB映射调用。3) `EplbState`新增`prepare_forward`方法，影响EPLB状态管理。4) `GPUModelRunner`新增`eplb.prepare_forward`调用，影响模型执行流程。5) spec decode的多个speculator新增`_prepare_eplb_forward`调用，影响推测解码流程。
  - 建议测试区域: vllm_ascend/tests/test_eplb.py, vllm_ascend/tests/test_routing.py

- **[04724365](https://github.com/vllm-project/vllm/commit/0472436541c842ecda6d249411f1d35649291a79)** 优化推测解码中draft prefill的hidden states收集逻辑。当`last_hidden_states is hidden_states`时（即模型返回了与输入相同的张量对象），直接使用`sample_hidden_states`而非通过索引`hidden_states[last_token_indices]`收集，避免冗余的gather操作。
  - 标签: `performance, low-risk, spec-decode`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-28
### vllm
- **[5c91039c](https://github.com/vllm-project/vllm/commit/5c91039c41bc0b6a4a4ab2dc5f62115946e38a30)** 将 DeepSeek-V2 模型的 MoE all-reduce 替换为 reduce-scatter，以提升 3.1%~3.2% 的端到端吞吐。实现方式：在 `DeepseekV2DecoderLayer` 中新增 `use_sequence_parallel_moe` 标志，当启用时，在 attention 输出后执行 reduce-scatter 将 hidden states 分散到各 TP rank，MoE 在分散后的数据上计算，最后通过 all-gather 恢复完整结果。这减少了 MoE 计算前的 all-reduce 通信开销。同时修改了 `DeepseekV2MoE` 的 `forward` 方法以支持 `already_sequence_parallel` 参数。
  - 标签: `performance, high-risk, distributed, model-runner`
  - Ascend 影响: ✓ 无影响

- **[6eb63a1d](https://github.com/vllm-project/vllm/commit/6eb63a1da6996abad00323dc7e845dc868996524)** 修复 DeepSeek-V3.2 模型中索引器权重加载的问题。当 `index_topk_freq>1` 时，只有部分层构建了 indexer，但 checkpoint 中所有层都包含 indexer 权重。变更在 `load_weights` 中检查当前层是否实际构建了 indexer，如果没有则跳过加载对应的 checkpoint 权重，避免 KeyError 或错误加载。
  - 标签: `bugfix, low-risk, model-runner`
  - Ascend 影响: ✓ 无影响

- **[c7ca0bcc](https://github.com/vllm-project/vllm/commit/c7ca0bccae667934c29c654544131cdab046adfd)** 为 GLM-4.5/6/7 模型添加 Fused Shared Expert (FSE) 支持。当 AITER 的 fused MoE 和 fusion shared experts 都启用时，将 shared experts 的权重合并到 routed experts 中，通过 FusedMoE 统一处理。变更涉及 `glm4_moe.py` 和 `glm4_moe_mtp.py` 中的权重加载逻辑，将 `mlp.shared_experts` 的权重按 `n_shared_experts` 切分后映射为额外的 routed expert 权重。
  - 标签: `feature, high-risk, model-runner, rocm`
  - Ascend 影响: ✓ 无影响

- **[c6741b2a](https://github.com/vllm-project/vllm/commit/c6741b2ad48a46e87d2cce35d113c4ae0950af91)** 新增 Unlimited-OCR 模型支持。该模型基于 DeepSeek-OCR 架构，但使用 DeepSeek-V2 MoE 语言骨干（64 routed + 2 shared experts）和 plain MHA（非 MLA）。核心特性：1) 使用 Reference Sliding Window Attention (R-SWA) 进行注意力计算，通过 FA4 的 mask_mod 或 FlexAttention 实现；2) 支持最多 32 个 local crops（vs DeepSeek-OCR 的 6 个）；3) 多图像请求时禁用 crop 模式。变更涉及模型定义、配置、处理器、tokenizer、KV cache 管理（新增 `RSWASpec` 和 `RSWAManager`）、注意力后端（FA4 和 FlexAttention 的 R-SWA mask_mod）等多个模块。
  - 标签: `feature, high-risk, model-runner, attention, multimodal`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) `vllm/v1/attention/backend.py` 中 `CommonAttentionMetadata` 新增 `rswa_prefix_lens` 字段，所有注意力后端（包括 AscendAttentionBackend）的元数据结构需要适配。2) `vllm/v1/core/single_type_kv_cache_manager.py` 中 `remove_skipped_blocks` 方法签名变更（新增 `num_prompt_tokens` 参数），影响 Ascend 的 KV cache manager patch。3) `vllm/v1/worker/gpu/input_batch.py` 中 `InputBatch` 新增 `rswa_prefix_lens` 字段，影响 Ascend 的 input batch 处理。4) `vllm/v1/kv_cache_interface.py` 新增 `RSWASpec`，如果 Ascend 需要支持 R-SWA 则需要实现对应的 spec 和 manager。

- **[11a12305](https://github.com/vllm-project/vllm/commit/11a12305c0522c5c1ed273d7d3dc2304ac0cd495)** 修复 Model Runner V2 中 MTP draft 模型的 hidden states 处理。将 `AutoRegressiveSpeculator` 的 `model_returns_tuple` 属性移除，改为在 `_run_model` 中通过 `isinstance(ret_hidden_states, tuple)` 动态判断返回值类型。这样 MTP 模型（如 DeepSeek）可以返回 `(logits_hidden, feedback_hidden)` 元组，而无需声明 `model_returns_tuple=True`。同时移除了 `Gemma4Speculator` 和 `MTPSpeculator` 中的 `model_returns_tuple` 属性。
  - 标签: `bugfix, low-risk, spec-decode`
  - Ascend 影响: ✓ 无影响

- **[b6caeb5a](https://github.com/vllm-project/vllm/commit/b6caeb5a0966103c6df22f019270d66233e1b687)** 修复 Spec Decode 中 rejection sampling 的随机数精度问题。将 `tl_rand64`（64-bit 随机数）替换为 `tl_rand32`（32-bit 随机数），并使用 fp32 均匀分布阈值进行 acceptance 判断。这避免了 fp64 随机数生成的开销，同时保持足够的精度。
  - 标签: `performance, low-risk, spec-decode, kernels`
  - Ascend 影响: ✓ 无影响

---

## 2026-06-27
### vllm
- **[c6dd32a8](https://github.com/vllm-project/vllm/commit/c6dd32a810aa8c4eda5696722c807e53d9f595a5)** 为 ModelRunner V2 支持实时 embeddings。主要变更包括：1) 在 `vllm/v1/worker/gpu/mm/encoder_runner.py` 中，`gather_mm_embeddings` 方法增加了对实时模型的支持，当模型支持实时推理时，不再跳过 decode 请求的 embeddings 收集；2) 在 `vllm/v1/worker/gpu/model_runner.py` 中，`execute_model` 方法在 dummy run 时使用预分配的 dummy inputs_embeds，避免调用 encoder；3) 在 `vllm/v1/worker/gpu/model_states/interface.py` 中，`gather_mm_embeddings` 的参数名从 `num_computed_prefill_tokens_np` 改为 `num_computed_tokens_np`；4) 在 `vllm/v1/worker/gpu/model_states/default.py` 中新增 `dummy_inputs_embeds` 方法。风险中等，因为修改了 ModelRunner 的核心逻辑和接口参数名，可能影响继承 `GPUModelRunner` 的外部实现。
  - 标签: `feature, medium-risk, model-runner, multimodal`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1) `vllm/v1/worker/gpu/mm/encoder_runner.py` 中 `gather_mm_embeddings` 方法的参数名从 `computed_prefill_lens` 改为 `num_computed_tokens`，vllm-ascend 的 `EncoderRunner` 实现若直接调用此方法需同步更新参数名。2) `vllm/v1/worker/gpu/model_states/interface.py` 中 `ModelState` 基类新增了 `dummy_inputs_embeds` 方法，vllm-ascend 的 `NPUModelRunner` 若继承 `GPUModelRunner` 需确认是否需要 override 此方法。3) `vllm/v1/worker/gpu/model_states/interface.py` 中 `gather_mm_embeddings` 的参数名从 `num_computed_prefill_tokens_np` 改为 `num_computed_tokens_np`，vllm-ascend 的 `NPUModelRunner` 若调用此方法需同步更新。
  - 建议测试区域: vllm_ascend/worker/model_runner_v1.py, vllm_ascend/worker/test_encoder_runner.py

- **[1d41009e](https://github.com/vllm-project/vllm/commit/1d41009e81eb6493f2c19e9d2a0d472564764e62)** 修复 ModelRunner V2 中 cross-attention block table 的尺寸计算问题。在 `vllm/v1/worker/gpu/model_runner.py` 中，`initialize_kv_cache` 方法在计算 `block_table_max_model_len` 时，除了考虑 `max_source_positions`，还加入了 `self.scheduler_config.max_num_encoder_input_tokens`，确保 cross-attention block table 能够索引 encoder tokens（如 Whisper 的 ~1500 tokens），这些 tokens 可能超过 decoder 的 `max_model_len`。风险较低，修复了明确的 bug。
  - 标签: `bugfix, low-risk, model-runner, attention`
  - Ascend 影响: ✓ 无影响

- **[b94f212e](https://github.com/vllm-project/vllm/commit/b94f212e37f4ddf4b5e1cc96cd87217f36e3ec0c)** 重构 ModelState 初始化逻辑，消除代码重复。将 `DefaultModelState`、`EncoderDecoderModelState` 和 `DiffusionGemmaModelState` 中重复的初始化代码（设置 `vllm_config`、`model_config`、`scheduler_config`、`model`、`device`、`max_model_len`、`max_num_reqs`、`max_num_tokens`、`inputs_embeds_size`、`dtype`、`supports_mm_inputs`、`encoder_cache`、`encoder_runner` 等）提取到基类 `ModelState.__init__` 中。子类现在只需调用 `super().__init__()` 即可。风险较低，纯重构，不改变行为。
  - 标签: `refactor, low-risk, model-runner`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: `vllm/v1/worker/gpu/model_states/interface.py` 中 `ModelState` 基类的 `__init__` 方法从抽象方法变为具体实现，初始化了 `encoder_runner` 等属性。vllm-ascend 的 `NPUModelRunner` 若直接访问 `self.encoder_runner` 或 `self.supports_mm_inputs` 等属性，行为不变。但若 vllm-ascend 有自定义的 `ModelState` 子类，需确保调用 `super().__init__()`。

---

## 2026-06-26
### vllm
- **[37ce3492](https://github.com/vllm-project/vllm/commit/37ce34922f7f5e58241369511130cd99c1c50bfe)** 修复了 Triton MoE 中 NVFP4 模拟的 CUDA Graph 捕获失败问题。在 Nvfp4QuantizationEmulationTritonExperts 类中新增了 a1_scale 属性，返回 self.a1_gscale，并在 triton_moe.py 中将 moe_kernel_quantize_input 的调用从 self.a1_scale or self.a1_gscale 改为 self.a1_scale。
  - 标签: `bugfix, low-risk, model-runner, quantization`
  - Ascend 影响: ✓ 无影响

- **[c2507fb2](https://github.com/vllm-project/vllm/commit/c2507fb2937aa8c8e74bea15719d04fb6090befe)** 为 ROCm 平台的 bias-routed MoE 实现了共享专家融合（shared-expert fusion），并启用了 MiniMax-M3 模型的 mxfp8 支持。主要变更：1) FusedMoE 层新增 shared_expert_weight 参数，用于在融合共享专家时调整权重；2) FusedTopKBiasRouter 新增 num_fused_shared_experts 和 shared_expert_weight 参数，在路由计算后将共享专家作为额外的 routed-expert slot 追加；3) MiniMax-M3 模型在 ROCm 平台上通过 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS 环境变量启用共享专家融合；4) mxfp8_native_moe 中修复了 binning 逻辑，使用 w13.shape[0] 而非 global_num_experts 来正确处理融合后的权重张量。
  - 标签: `feature, performance, medium-risk, model-runner, moe`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1. FusedMoE 层新增了 shared_expert_weight 参数，NPUModelRunner 在调用 FusedMoE 时需要传递此参数。2. FusedTopKBiasRouter 新增了 num_fused_shared_experts 和 shared_expert_weight 参数，Ascend 的 MoE 路由实现（如果有自定义路由）需要同步适配。3. create_fused_moe_router 函数新增了 shared_expert_weight 参数，所有调用该函数的地方需要适配。4. determine_expert_counts 函数中移除了对 rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() 的独占依赖，改为同时支持 envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS，Ascend 的 MoE 配置逻辑不受影响。

- **[8e394244](https://github.com/vllm-project/vllm/commit/8e394244a59afc67a37bf47dab0ab76bf5ce5885)** 为 MiniMax-M3-MXFP4 模型启用了 AITER MoE 后端。主要变更：1) FusedMoEConfig 新增 intermediate_pad 字段；2) rocm_aiter_moe.py 支持 SWIGLUOAI_UNINTERLEAVE 激活函数，并新增 activation_interleave 参数控制 gate_mode；3) MiniMax-M3 模型在调用 FusedMoE 时传递 intermediate_pad=0。
  - 标签: `feature, low-risk, model-runner, moe`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1. FusedMoEConfig 新增了 intermediate_pad 字段，NPUModelRunner 在创建 FusedMoEConfig 时需要适配。2. FusedMoE 函数新增了 intermediate_pad 参数，所有调用 FusedMoE 的地方需要传递此参数。3. MoEActivation 新增了 SWIGLUOAI_UNINTERLEAVE 枚举值，Ascend 的 MoE 激活函数处理逻辑需要检查是否需要支持此枚举。

- **[5b330417](https://github.com/vllm-project/vllm/commit/5b33041746b9b9ab45bdbd9b42cdd5d19357879a)** 修复了 whisper 测试中的两个问题：1) EncoderCache 新增 __len__ 方法；2) maybe_create_mm_pruner 中的空值检查从 not rope_state 改为 rope_state is None，从 not encoder_cache 改为 encoder_cache is None，从 not model_config.multimodal_config 改为 model_config.multimodal_config is None，以避免在空张量或空列表等 falsy 值上误判。
  - 标签: `bugfix, low-risk, model-runner, multimodal`
  - Ascend 影响: ⚠️ 影响 Ascend
  - 影响描述: 1. EncoderCache 新增了 __len__ 方法，所有使用 EncoderCache 的代码（包括 Ascend 的 encoder 实现）可以调用 len() 获取缓存大小。2. maybe_create_mm_pruner 中的空值检查从 not 改为 is None，这改变了行为：之前空张量（falsy）会触发提前返回，现在只有 None 才会触发。Ascend 的 MM pruner 实现如果依赖旧行为需要适配。

- **[02a1f237](https://github.com/vllm-project/vllm/commit/02a1f23711c5bdbff81eb8a610dde39e1141d036)** 为 DFlash 实现了逐层 K-norm 的融合 RMSNorm。修改了 csrc 中的 rms_norm kernel，支持 2D 权重（[num_groups, hidden_size]），使得可以一次性对所有层的 K 进行 RMSNorm，而不是逐层循环。在 Qwen3DFlash 模型中，将 K-norm 权重堆叠为 [num_layers, head_dim] 的连续张量，并调用一次 ops.rms_norm 完成所有层的归一化。
  - 标签: `performance, low-risk, model-runner, spec-decode`
  - Ascend 影响: ✓ 无影响

- **[652d962b](https://github.com/vllm-project/vllm/commit/652d962bc9df7e04959e84ce478c3a8d26fe52a7)** 为推测解码减少了 draft token 生成时的 TP 通信。新增 use_local_argmax_reduction 配置选项，当启用时，draft 模型的 greedy sample 通过调用 model.get_top_tokens() 在本地获取 argmax，而不是先通过 compute_logits 计算完整 logits 再 argmax，从而将通信量从 O(vocab_size) 降低到 O(2*tp_size)。同时增加了验证逻辑，确保该模式与 probabilistic 采样不兼容，且 draft 模型实现了 get_top_tokens 方法。
  - 标签: `performance, low-risk, spec-decode`
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
