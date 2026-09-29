# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    """Pad the token axis (dim 0) to the TP multiple, then take this rank's chunk."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    # Upstream counterpart: vllm/models/common/ops/sequence_parallel.py
    # sp_shard L45-48 (introduced in 38a466e7b6, #46789).
    if sp_pad > 0:
        x = F.pad(x, (0, 0) * (x.ndim - 1) + (0, sp_pad))
    chunk = x.shape[0] // tp_size
    out = x[tp_rank * chunk : (tp_rank + 1) * chunk]
    return out


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    """Pad rows to the TP multiple, then reduce-scatter across TP ranks."""
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    # Avoid copying the full input when its token count is already aligned.
    if sp_pad > 0:
        pad_shape = [sp_pad, x.shape[1]]
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Pad with True rows up to the TP multiple, then take this rank's chunk.

    The output row layout matches ``sp_shard`` so the mask stays aligned with
    the sharded hidden states.
    """
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-num_tokens) % tp_size
    # Upstream counterpart: vllm/models/common/ops/sequence_parallel.py
    # sp_padding_mask L63-65.
    if sp_pad > 0:
        is_padding = F.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    out = is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
    return out
