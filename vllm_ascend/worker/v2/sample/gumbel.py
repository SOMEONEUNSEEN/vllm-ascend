# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

# CANN libdevice is the Ascend equivalent of triton.language.extra.libdevice
# (used by upstream for log1p). Unavailable on non-Ascend Triton builds.
try:
    from triton.language.extra.cann import libdevice as tldevice
except ImportError:
    tldevice = None

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10


@triton.jit(do_not_specialize=["logits_stride", "vocab_size"])
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """
    Args:
        logits: Tensor of shape (num_tokens, vocab_size) containing the logits.
        expanded_idx_mapping: Tensor containing the mapping from token index
            to request index of tensor temperature.
        temperature: Tensor containing the temperature value for each request.
    """
    num_tokens, vocab_size = logits.shape
    # BLOCK_SIZE fits the FP32 working vector within the Ascend A2/A3 UB limit
    # (192 KB). Upstream restores the FP32 logit upcast (vllm #49033), so on
    # main / v0.27.1 the kernel gets FP32 logits and 44032 * 32 bits fits.
    BLOCK_SIZE = 44032
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    # NPU: triton-ascend has no scalar masked load, so clamp the index to
    # avoid OOB access for invalid requests (e.g. -1 mappings) instead.
    safe_req_state_idx = tl.maximum(req_state_idx, 0)
    temp = tl.load(temp_ptr + safe_req_state_idx).to(tl.float32)

    if logits_cache_ptr is not None:
        # Store the logits *before* temperature. Dividing first would produce
        # a value that is generally not representable in the cache's dtype,
        # forcing it to be fp32. Consumers (the rejection sampler) divide by
        # the same temperature on load, which reproduces the value used below
        # bitwise.
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        tl.store(
            logits_cache_ptr + req_state_idx * logits_cache_stride + col * vocab_size + block,
            logits,
            mask=mask & is_valid_req,
        )

    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + safe_req_state_idx)
        # NOTE(Ronald1995): change pos's dtype to tl.int32, because triton-ascend's
        # compiler doesn't support uint64 of pos arg.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)

        # NOTE(Ronald1995): r is tl.float64 in vllm, change it to tl.float32,
        # because triton-ascend's compiler does not support float64.
        # Clamp away from 0 so the flipped Gumbel transform stays finite
        # (NPU equivalent of upstream tl_rand32(includes_zero=False)).
        u = tl.maximum(tl.rand(gumbel_seed, block).to(tl.float32), _TL_RAND_MIN)
        # Draw the large-noise tail (which decides the argmax winner) from
        # u -> 0, where fp32 has fine resolution, instead of u -> 1
        # (same flipped transform as upstream `-log(-log1p(-u))`).
        gumbel_noise = -tl.log(-tldevice.log1p(-u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit(
    do_not_specialize=[
        "local_argmax_stride",
        "local_max_stride",
        "logits_cache_stride",
        "logits_stride",
        "vocab_size",
        "num_blocks",
    ]
)
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0)

    for block_idx in range(num_blocks):
        block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        value, idx = gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            logits_cache_ptr,
            logits_cache_stride,
            logits_cache_col_ptr,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
            PER_TOKEN_COL=PER_TOKEN_COL,
        )
        token_id = block_idx * BLOCK_SIZE + idx
        tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
        tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    if use_fp64:
        raise NotImplementedError("FP64 Gumbel sampling is not supported on NPU.")
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    # NPU does not support float64; use float32 for local_max.
    local_max = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens,)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
