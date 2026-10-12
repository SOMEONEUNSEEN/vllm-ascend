# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.common.ops import sequence_parallel as sp


@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (10, 4), (1, 4), (3, 1), (0, 4)])
@pytest.mark.parametrize("trailing_shape", [(8,), (2, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_shard_padding_matches_pad(monkeypatch, num_tokens, tp_size, trailing_shape, dtype, noncontiguous):
    shape = (num_tokens, *trailing_shape)
    storage_shape = (*shape[:-1], shape[-1] * 2) if noncontiguous else shape
    x = torch.randn(storage_shape, dtype=dtype)
    if noncontiguous:
        x = x[..., ::2]
    original = x.clone()
    padding = (-num_tokens) % tp_size
    expected = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, padding))
    chunk = expected.shape[0] // tp_size
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)

    for rank in range(tp_size):
        monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        result = sp._ascend_sp_shard_impl(x)
        torch.testing.assert_close(result, expected[rank * chunk : (rank + 1) * chunk], rtol=0, atol=0)
        assert result.dtype == x.dtype and result.device == x.device
        # Mutation detects input aliasing on both padded and aligned paths.
        result.fill_(-1)
        torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (10, 4), (1, 4), (3, 1), (0, 4)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_padding_mask_matches_pad(monkeypatch, num_tokens, tp_size, noncontiguous):
    storage_size = num_tokens * 2 if noncontiguous else num_tokens
    mask = torch.arange(storage_size) % 3 == 0
    if noncontiguous:
        mask = mask[::2]
    original = mask.clone()
    padding = (-num_tokens) % tp_size
    expected = torch.nn.functional.pad(mask, (0, padding), value=True)
    chunk = expected.shape[0] // tp_size
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)

    for rank in range(tp_size):
        monkeypatch.setattr(sp, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        result = sp._ascend_sp_padding_mask_impl(mask)
        torch.testing.assert_close(result, expected[rank * chunk : (rank + 1) * chunk], rtol=0, atol=0)
        assert result.dtype == mask.dtype and result.device == mask.device
        result.logical_not_()
        torch.testing.assert_close(mask, original, rtol=0, atol=0)


@pytest.mark.parametrize("custom_collective", [False, True])
@pytest.mark.parametrize("num_tokens,tp_size", [(8, 4), (10, 4), (1, 4), (3, 1), (0, 4)])
def test_reduce_scatter_padding(monkeypatch, custom_collective, num_tokens, tp_size):
    x = torch.arange(num_tokens * 8, dtype=torch.float32).reshape(num_tokens, 8)
    padding = (-num_tokens) % tp_size
    expected = torch.nn.functional.pad(x, (0, 0, 0, padding))
    inputs = []

    def reduce_scatter(value):
        inputs.append(value)
        torch.testing.assert_close(value, expected)
        if padding == 0:
            # The aligned path must preserve the input buffer, avoiding Cat.
            assert value is x
        return value[: value.shape[0] // tp_size]

    custom = Mock(side_effect=reduce_scatter) if custom_collective else Mock(return_value=None)
    fallback = Mock(side_effect=lambda value, dim: reduce_scatter(value))
    communicator = SimpleNamespace(custom_reduce_scatter=custom)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(sp, "get_tp_group", lambda: SimpleNamespace(device_communicator=communicator))
    monkeypatch.setattr(sp, "tensor_model_parallel_reduce_scatter", fallback)

    # This CPU-only test exercises the custom-op implementation directly;
    # the public wrapper intentionally registers only the PrivateUse1 kernel.
    result = sp._ascend_sp_reduce_scatter_impl(x)

    assert len(inputs) == 1
    torch.testing.assert_close(result, expected[: expected.shape[0] // tp_size])
    custom.assert_called_once()
    if custom_collective:
        fallback.assert_not_called()
    else:
        fallback.assert_called_once_with(inputs[0], 0)
