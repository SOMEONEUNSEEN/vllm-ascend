#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos

DSPARK_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DSPARK_MODELS = ["deepseek-ai/dspark_qwen3_8b_block7"]


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("dspark_model", DSPARK_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    ("compilation_config", "enable_adaptive_verification"),
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            False,
            id="full_decode_only",
        ),
        pytest.param({}, False, id="default_full_and_piecewise"),
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            True,
            id="full_decode_only-adaptive",
        ),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_spec_decoding(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
    enable_adaptive_verification: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    num_speculative_tokens = 7
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            **({"enable_adaptive_verification": True} if enable_adaptive_verification else {}),
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    if enable_adaptive_verification:
        return

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.84, 0.48, 0.32, 0.20, 0.09, 0.09, 0.02]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("dspark_model", DSPARK_MODELS)
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_probabilistic_spec_decoding(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    num_speculative_tokens = 7
    seed = 42
    other_seed = 1234

    def sampling_params(request_seed: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_k=20,
            top_p=0.95,
            seed=request_seed,
            ignore_eos=True,
        )

    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        enable_prefix_caching=False,
        speculative_config={
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            "draft_sample_method": "probabilistic",
        },
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params(seed))
        outputs_other_seed = runner.model.generate(prompts, sampling_params(other_seed))
        metrics = runner.model.get_metrics()

    def token_ids(outputs: list) -> list[list[int]]:
        return [out.outputs[0].token_ids for out in outputs]

    ids = token_ids(outputs)
    ids_other_seed = token_ids(outputs_other_seed)

    for request_ids in ids + ids_other_seed:
        assert len(request_ids) == max_tokens
        assert len(set(request_ids)) > 1, "Degenerate output: single repeated token"
    assert any(a != o for a, o in zip(ids, ids_other_seed)), (
        "Outputs identical across different seeds: probabilistic sampling likely fell back to greedy"
    )
    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    print(f"probabilistic acceptance_per_pos: {acceptance_per_pos}")
    golden = [0.74, 0.48, 0.36, 0.27, 0.17, 0.10, 0.04]
    match = all(abs(a - b) < 0.2 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


def _extract_acceptance_stats(metrics: list) -> tuple[int, list[int]]:
    """Return cumulative (num_drafts, accepted_tokens_per_pos) from metrics."""
    num_drafts = 0
    accepted_per_pos: list[int] = []
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            if not accepted_per_pos:
                accepted_per_pos = [0] * len(metric.values)
            for pos in range(len(metric.values)):
                accepted_per_pos[pos] += metric.values[pos]
    return num_drafts, accepted_per_pos


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("dspark_model", DSPARK_MODELS)
@pytest.mark.parametrize("max_tokens", [48])
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dspark_synthetic_rejection_sampling(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    """DSpark spec decoding with synthetic rejection sampling.

    Guards the SYNTHETIC_MODE branch of the NPU rejection sampling kernel
    end to end. Acceptance is driven purely by u ~ U(0, 1) < conditional
    rate, decoupled from draft quality, so the measured per-position
    acceptance must match the configured synthetic_acceptance_rates — a
    stronger assertion than a draft-quality golden. Both verify paths are
    covered with one engine: temperature=0 exercises the greedy SYNTHETIC
    branch (the NPU-specific scalar-random adaptation) and temperature=0.7
    the non-greedy one; per-run rates are recovered from cumulative
    metrics deltas.
    """
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    num_speculative_tokens = 7
    # Unconditional per-position acceptance rates (must be non-increasing,
    # one entry per draft position). The kernel loads the conditional rates
    # (c_i = p_i / p_{i-1}), so the measured per-position acceptance must
    # reproduce these values in distribution.
    rates = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]
    # Tolerance: 4 * 48 = 192 verify steps per run; the worst-case binomial
    # sigma (rate 0.4) is ~0.035, so 0.12 leaves >3.4 sigma of margin at
    # every position.
    tolerance = 0.12

    def sampling_params(temperature: float, seed: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            ignore_eos=True,
        )

    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_rates": rates,
        },
    ) as runner:
        outputs_greedy = runner.model.generate(prompts, sampling_params(0.0, 42))
        greedy_stats = _extract_acceptance_stats(runner.model.get_metrics())
        outputs_sampled = runner.model.generate(prompts, sampling_params(0.7, 1234))
        sampled_stats = _extract_acceptance_stats(runner.model.get_metrics())

    # Correctness: full-length, non-degenerate outputs.
    for outputs in (outputs_greedy, outputs_sampled):
        for output in outputs:
            request_ids = output.outputs[0].token_ids
            assert len(request_ids) == max_tokens
            assert len(set(request_ids)) > 1, "Degenerate output: single repeated token"

    # Acceptance health: both verify paths must reproduce the configured
    # rates. A broken rate index, u generation, or SYNTHETIC branch wiring
    # shifts the measured rates away from the configured ones.
    for name, (metrics_before, metrics_after) in (
        ("greedy", greedy_stats),
        ("sampled", sampled_stats),
    ):
        drafts_before, accepted_before = metrics_before
        drafts_after, accepted_after = metrics_after
        drafts = drafts_after - drafts_before
        assert drafts > 0, f"No verify steps recorded for the {name} run"
        acceptance_per_pos = [(a - b) / drafts for a, b in zip(accepted_after, accepted_before)]
        print(f"synthetic {name} acceptance_per_pos: {acceptance_per_pos}")
        match = all(abs(a - r) < tolerance for a, r in zip(acceptance_per_pos, rates))
        assert match, (
            f"synthetic {name} acceptance_per_pos {acceptance_per_pos} does "
            f"not match configured rates {rates} (tolerance {tolerance})"
        )
