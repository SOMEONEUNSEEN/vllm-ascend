from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu import attn_utils as upstream_attn_utils
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.utils import AttentionGroup

from tests.deepseek_v41_cache_utils import allocate_cache_views, make_cache_config
from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAC4Backend,
    AscendDSAC4StateBackend,
    AscendDSAC128Backend,
    AscendDSAC128StateBackend,
    AscendDSAMetadataBuilder,
    AscendDSASWABackend,
)
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
    DeepseekV41MetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4 import compressor as deepseek_v4_compressor
from vllm_ascend.models.deepseek_v4 import indexer as deepseek_v4_indexer
from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_model
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.mark.parametrize(
    ("replicated_indexer", "expected_size"),
    [(False, 1), (True, 4)],
)
def test_sfa_indexer_cache_spec_uses_dcp_replication(monkeypatch, replicated_indexer, expected_size):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
    torch.nn.Module.__init__(indexer_module)
    monkeypatch.setattr(
        indexer_module,
        "get_kv_cache_spec",
        lambda _config: object(),
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(index_head_dim=128),
        ),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {layer_name: indexer_module},
    )
    monkeypatch.setattr(
        attn_utils,
        "enable_sfa_dcp_replicated_indexer",
        lambda _config: replicated_indexer,
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(is_sparse_li_c8_layer=lambda _layer_name: False),
    )

    spec = attn_utils.get_kv_cache_spec(vllm_config)[layer_name]

    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.sfa_dcp_replicated_indexer_size == expected_size


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "scale_dtype", "component_dims"),
    [
        (
            AscendDeviceType.A2,
            torch.int8,
            torch.float16,
            (128, 1),
        ),
        (
            AscendDeviceType.A5,
            torch.float8_e4m3fn,
            torch.float32,
            (128, 1, 132),
        ),
    ],
)
def test_mrv2_initializes_dsv4_cache_only_layer(
    monkeypatch,
    device_type,
    cache_dtype,
    scale_dtype,
    component_dims,
):
    """Exercise DSV4 discovery, allocation, reshape, and binding as one flow."""
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    cache_config = SimpleNamespace(block_size=32, cache_dtype="auto")
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                compress_ratios=[4],
                model_type="deepseek_v4",
            ),
        ),
        cache_config=cache_config,
        kv_transfer_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )

    cache_layer = deepseek_v4_indexer.AscendDeepseekV4IndexerCache.__new__(
        deepseek_v4_indexer.AscendDeepseekV4IndexerCache
    )
    torch.nn.Module.__init__(cache_layer)
    cache_layer.head_dim = 128
    cache_layer.dtype = torch.int8
    cache_layer.cache_config = cache_config
    cache_layer.compress_ratio = 4
    cache_layer.kv_cache = torch.tensor([])

    monkeypatch.setattr(
        deepseek_v4_indexer,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {layer_name: cache_layer},
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_vllm_config",
        lambda: vllm_config,
    )
    monkeypatch.setattr(
        upstream_attn_utils,
        "get_shared_kv_cache_layers",
        lambda _config: {},
    )

    discovered_specs = attn_utils.get_kv_cache_spec(vllm_config)
    assert set(discovered_specs) == {layer_name}
    spec = discovered_specs[layer_name]
    assert isinstance(spec, AscendMLAAttentionSpec)
    assert spec.block_size == cache_config.block_size * cache_layer.compress_ratio
    assert spec.storage_block_size == cache_config.block_size
    merged_spec = spec.merge([spec])
    assert merged_spec.compress_ratio == cache_layer.compress_ratio
    assert merged_spec.storage_block_size == cache_config.block_size

    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                shared_by=[layer_name],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )
    attn_group = AttentionGroup(
        backend=AscendDSAC4Backend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    runner_kv_caches: list[Any] = []

    kv_caches = upstream_attn_utils.init_kv_cache(
        runner_kv_caches=runner_kv_caches,
        forward_context={layer_name: cache_layer},
        kv_cache_config=kv_cache_config,
        attn_groups=[[attn_group]],
        device=torch.device("cpu"),
        cache_dtype=cache_config.cache_dtype,
        kernel_block_sizes=[spec.block_size],
        vllm_config=vllm_config,
    )

    cache_components = kv_caches[layer_name]
    assert cache_layer.kv_cache is cache_components
    assert len(runner_kv_caches) == 1
    assert runner_kv_caches[0] is cache_components
    assert [component.shape for component in cache_components] == [
        (num_blocks, spec.storage_block_size, 1, dim) for dim in component_dims
    ]
    assert [component.dtype for component in cache_components] == [
        cache_dtype,
        scale_dtype,
        *([cache_dtype] if device_type == AscendDeviceType.A5 else []),
    ]
    backing_storage = cache_components[0].untyped_storage().data_ptr()
    assert all(component.untyped_storage().data_ptr() == backing_storage for component in cache_components)


class _RecordingDSAMetadataBuilder(AscendDSAMetadataBuilder):
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls
        self.for_cudagraph_capture = False

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ):
        self.for_cudagraph_capture = True
        return super().build_for_cudagraph_capture(
            common_attn_metadata,
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
        **kwargs,
    ):
        del common_prefix_len, fast_build
        self.common_ratio_to_sas_metadata = kwargs["common_ratio_to_sas_metadata"]
        call = {
            "common_attn_metadata": common_attn_metadata,
            "common_ratio_to_sas_metadata": self.common_ratio_to_sas_metadata,
            "for_cudagraph_capture": self.for_cudagraph_capture,
            "num_actual_reqs": kwargs["num_actual_reqs"],
            "pcp_context": kwargs.get("pcp_context"),
            "pcp_cache_group_idx": kwargs.get("pcp_cache_group_idx"),
        }
        assert "block_size" not in kwargs
        self.calls.append(call)
        call["common_ratio_to_sas_metadata"].setdefault("first_group", len(self.calls) == 1)
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


def _make_dsa_metadata_groups():
    layer_names = [
        "model.layers.0.self_attn.compressor",
        "model.layers.0.self_attn.indexer",
    ]
    specs = [
        AscendMLAAttentionSpec(
            block_size=storage_block_size * compress_ratio,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
            model_version="deepseek_v4",
        )
        for storage_block_size, compress_ratio in ((32, 4), (64, 128))
    ]
    calls: list[dict[str, Any]] = []
    attn_groups = [
        [
            AttentionGroup(
                backend=(AscendDSAC4Backend if spec.compress_ratio == 4 else AscendDSAC128Backend),
                layer_names=[layer_name],
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
                metadata_builders=[_RecordingDSAMetadataBuilder(calls)],
            )
        ]
        for group_id, (layer_name, spec) in enumerate(zip(layer_names, specs))
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
            for layer_name, spec in zip(layer_names, specs)
        ],
    )
    return layer_names, specs, calls, attn_groups, kv_cache_config


def test_prepare_kernel_block_sizes_uses_logical_size_for_dsv4():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    attn_groups = [
        [
            AttentionGroup(
                backend=AscendDSAC4Backend,
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=spec,
                kv_cache_group_id=0,
            )
        ]
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"],
                kv_cache_spec=spec,
            )
        ],
    )

    assert spec.storage_block_size == 32
    assert upstream_attn_utils.prepare_kernel_block_sizes(kv_cache_config, attn_groups) == [spec.block_size]


@pytest.mark.parametrize(
    ("device_type", "expected_c128_state_sizes"),
    [
        (AscendDeviceType.A2, [8, 16, 32]),
        (AscendDeviceType.A5, [4, 8, 16]),
    ],
)
def test_dsv4_backends_declare_role_specific_logical_sizes(
    monkeypatch,
    device_type,
    expected_c128_state_sizes,
):
    monkeypatch.setattr(
        dsa_v1,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )

    assert AscendDSAC4Backend.get_supported_kernel_block_sizes() == [128, 256, 512]
    assert AscendDSAC128Backend.get_supported_kernel_block_sizes() == [4096, 8192, 16384]
    assert AscendDSASWABackend.get_supported_kernel_block_sizes() == [32, 64, 128]
    assert AscendDSAC4StateBackend.get_supported_kernel_block_sizes() == [2, 4, 8]
    assert AscendDSAC128StateBackend.get_supported_kernel_block_sizes() == expected_c128_state_sizes

    c4_cache = SimpleNamespace(compress_ratio=4)
    c128_cache = SimpleNamespace(compress_ratio=128)
    c4_indexer = cast(deepseek_v4_indexer.AscendDeepseekV4IndexerCache, c4_cache)
    c128_indexer = cast(deepseek_v4_indexer.AscendDeepseekV4IndexerCache, c128_cache)
    swa_cache = cast(deepseek_v4_model.AscendDeepseekV4SWACache, SimpleNamespace())
    c4_state = cast(deepseek_v4_compressor.AscendCompressorStateCache, c4_cache)
    c128_state = cast(deepseek_v4_compressor.AscendCompressorStateCache, c128_cache)
    assert deepseek_v4_indexer.AscendDeepseekV4IndexerCache.get_attn_backend(c4_indexer) is AscendDSAC4Backend
    assert deepseek_v4_indexer.AscendDeepseekV4IndexerCache.get_attn_backend(c128_indexer) is AscendDSAC128Backend
    assert deepseek_v4_model.AscendDeepseekV4SWACache.get_attn_backend(swa_cache) is AscendDSASWABackend
    assert deepseek_v4_compressor.AscendCompressorStateCache.get_attn_backend(c4_state) is AscendDSAC4StateBackend
    assert deepseek_v4_compressor.AscendCompressorStateCache.get_attn_backend(c128_state) is AscendDSAC128StateBackend


@pytest.mark.parametrize(
    (
        "caller",
        "cudagraph_mode",
        "for_capture",
        "pcp_size",
        "expected_input_tokens",
    ),
    [
        ("default", None, False, 1, 5),
        ("model_state", CUDAGraphMode.NONE, False, 1, 5),
        ("model_state", CUDAGraphMode.FULL, False, 1, 8),
        ("pcp_capture", CUDAGraphMode.NONE, True, 2, 8),
    ],
)
def test_mrv2_builds_shared_dsa_metadata_for_each_execution_mode(
    caller,
    cudagraph_mode,
    for_capture,
    pcp_size,
    expected_input_tokens,
):
    layer_names, specs, calls, attn_groups, kv_cache_config = _make_dsa_metadata_groups()
    block_tables = (
        torch.zeros((4, 1), dtype=torch.int32),
        torch.zeros((4, 1), dtype=torch.int32),
    )
    slot_mappings = torch.zeros((2, 8), dtype=torch.int32)
    dcp_local_seq_lens = torch.tensor(
        [2, 1, 0, 0],
        dtype=torch.int32,
    )
    pcp_context = object() if caller == "pcp_capture" else None
    pcp_manager = (
        SimpleNamespace(
            build_attention_context=MagicMock(return_value=pcp_context),
        )
        if pcp_context is not None
        else None
    )

    if caller == "default":
        metadata = attn_utils.build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=2,
            num_tokens=5,
            query_start_loc_gpu=torch.tensor([0, 2, 5], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
            max_query_len=3,
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            max_seq_len=8,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_np=np.array([2, 3], dtype=np.int32),
            positions=torch.arange(5, dtype=torch.int32),
            dcp_local_seq_lens=dcp_local_seq_lens[:2],
        )
    else:
        model_state = AscendModelState.__new__(AscendModelState)
        model_state.max_model_len = 8
        model_state.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=pcp_size,
            ),
        )
        model_state.pcp_manager = pcp_manager
        input_batch = SimpleNamespace(
            num_reqs=2,
            num_reqs_after_padding=4,
            num_tokens=5,
            num_tokens_after_padding=8,
            query_start_loc_np=np.array([0, 2, 5, 5, 5], dtype=np.int32),
            query_start_loc=torch.tensor([0, 2, 5, 5, 5], dtype=torch.int32),
            num_scheduled_tokens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
            seq_lens_np=np.array([2, 3, 0, 0], dtype=np.int32),
            is_prefilling_np=np.array([True, True, False, False]),
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=torch.arange(8, dtype=torch.int32),
            attn_state=None,
        )
        metadata = model_state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=cudagraph_mode,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
            for_capture=for_capture,
        )

    assert set(metadata) == set(layer_names)
    assert len(calls) == 2
    for call in calls:
        common_metadata = call["common_attn_metadata"]
        assert common_metadata.num_actual_tokens == 5
        assert common_metadata.num_input_tokens == expected_input_tokens
        assert call["for_cudagraph_capture"] is for_capture
        assert call["num_actual_reqs"] == 2
        assert call["pcp_context"] is pcp_context
        if caller != "default":
            assert torch.equal(
                common_metadata.is_prefilling,
                torch.tensor([True, True, False, False]),
            )
        expected_dcp_local_seq_lens = dcp_local_seq_lens[:2] if caller == "default" else dcp_local_seq_lens
        torch.testing.assert_close(common_metadata.dcp_local_seq_lens, expected_dcp_local_seq_lens)
    cache_name = "common_ratio_to_sas_metadata"
    assert calls[0][cache_name] is calls[1][cache_name]
    assert calls[1][cache_name]["first_group"] is True
    if pcp_context is not None:
        assert [call["pcp_cache_group_idx"] for call in calls] == [0, 1]
        assert pcp_manager is not None
        pcp_manager.build_attention_context.assert_called_once_with()
    else:
        assert all(call["pcp_cache_group_idx"] is None for call in calls)


class _PrefillStateBuilder:
    def build(self, common_prefix_len, common_attn_metadata):
        assert common_prefix_len == 0
        return common_attn_metadata.is_prefilling


def test_build_attn_metadata_propagates_prefill_state():
    attn_group = SimpleNamespace(
        layer_names=["layer.0"],
        get_metadata_builder=lambda _: _PrefillStateBuilder(),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
    )
    is_prefilling = torch.tensor([True])

    metadata = attn_utils.build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=(torch.zeros(1, dtype=torch.int64),),
        kv_cache_config=kv_cache_config,
        is_prefilling=is_prefilling,
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.tensor([0], dtype=torch.int64),
    )

    assert metadata["layer.0"] is is_prefilling


class _RecordingV41MetadataBuilder(DeepseekV41MetadataBuilder):
    def __init__(self, calls: list[dict[str, Any]]):
        self.calls = calls
        self.for_cudagraph_capture = False

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ):
        self.for_cudagraph_capture = True
        return super().build_for_cudagraph_capture(common_attn_metadata, **kwargs)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
        **kwargs,
    ):
        del common_prefix_len, fast_build
        self.calls.append(
            {
                "common_attn_metadata": common_attn_metadata,
                "for_cudagraph_capture": self.for_cudagraph_capture,
                **kwargs,
            }
        )
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


@pytest.mark.parametrize("for_capture", [False, True])
def test_build_attn_metadata_injects_v41_shared_dicts_across_groups(for_capture):
    calls: list[dict[str, Any]] = []
    attn_groups = [
        [
            AttentionGroup(
                backend=DeepseekV41CacheBackend,
                layer_names=[f"model.layers.{2 * gid}.self_attn.attn"],
                kv_cache_spec=SimpleNamespace(),
                kv_cache_group_id=gid,
                metadata_builders=[_RecordingV41MetadataBuilder(calls)],
            )
        ]
        for gid in range(2)
    ]
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(), SimpleNamespace()])

    attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=2,
        num_actual_reqs=1,
        num_tokens=4,
        query_start_loc_gpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([2, 2], dtype=torch.int32),
        max_seq_len=4,
        block_tables=(torch.zeros((2, 1), dtype=torch.int32),) * 2,
        slot_mappings=(torch.zeros(4, dtype=torch.int32),) * 2,
        kv_cache_config=kv_cache_config,
        full_graph_mode=True,
        for_cudagraph_capture=for_capture,
    )

    assert len(calls) == 2
    for call in calls:
        assert call["num_actual_reqs"] == 1
        assert call["skip_ring_state_update"] is False
        assert call["full_graph_mode"] is True
        assert call["for_cudagraph_capture"] is for_capture
    # Slot coordinates stay group-local while batch metadata is shared.
    assert calls[0]["common_v41_metadata"] is not calls[1]["common_v41_metadata"]
    assert calls[0]["common_v41_batch_metadata"] is calls[1]["common_v41_batch_metadata"]


def _make_v41_runtime(monkeypatch):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[2], model_type="deepseek_v4.1"),
        ),
        cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
        kv_transfer_config=None,
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        upstream_attn_utils,
        "get_shared_kv_cache_layers",
        lambda _config: {},
    )
    return vllm_config


def test_mrv2_initializes_v41_cache_layers_end_to_end(monkeypatch):
    """Exercise V4.1 allocation, reshape, and binding as one flow."""
    vllm_config = _make_v41_runtime(monkeypatch)
    cache_config = make_cache_config(num_blocks=2, block_size=64, head_size=128, index_size=64, draft_layers=3)

    forward_context: dict[str, Any] = {}
    layer_names = []
    for group in cache_config.kv_cache_groups:
        for name in group.layer_names:
            layer = DeepseekV41CacheLayer.__new__(DeepseekV41CacheLayer)
            torch.nn.Module.__init__(layer)
            layer.kv_cache = [torch.empty(0)]
            forward_context[name] = layer
            layer_names.append(name)

    attn_groups = []
    for gid, group in enumerate(cache_config.kv_cache_groups):
        by_spec: dict[int, tuple[Any, list[str]]] = {}
        for name in group.layer_names:
            spec = group.kv_cache_spec.kv_cache_specs[name]
            by_spec.setdefault(id(spec), (spec, []))[1].append(name)
        attn_groups.append(
            [
                AttentionGroup(
                    backend=DeepseekV41CacheBackend,
                    layer_names=names,
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                for spec, names in by_spec.values()
            ]
        )

    kernel_block_sizes = [group.kv_cache_spec.block_size for group in cache_config.kv_cache_groups]
    runner_kv_caches: list[Any] = []
    kv_caches = upstream_attn_utils.init_kv_cache(
        runner_kv_caches=runner_kv_caches,
        forward_context=forward_context,
        kv_cache_config=cache_config,
        attn_groups=attn_groups,
        device=torch.device("cpu"),
        cache_dtype="auto",
        kernel_block_sizes=kernel_block_sizes,
        vllm_config=vllm_config,
    )

    # V4.1 layers consume ``kv_cache[0]``: the bound value must be a list.
    for name in layer_names:
        assert forward_context[name].kv_cache == [kv_caches[name]]
    # The runner list is filled once per resource in sorted-name order.
    assert len(runner_kv_caches) == len(layer_names)
    assert runner_kv_caches == [kv_caches[name] for name in sorted(layer_names)]

    # Views match the reference slot layout (shape and dtype per plane).
    _, reference = allocate_cache_views(cache_config)
    assert set(kv_caches) == set(reference)
    for name, ref in reference.items():
        mine = kv_caches[name]
        mine_planes = mine if isinstance(mine, tuple) else (mine,)
        ref_planes = ref if isinstance(ref, tuple) else (ref,)
        for mine_plane, ref_plane in zip(mine_planes, ref_planes):
            assert mine_plane.shape == ref_plane.shape
            assert mine_plane.dtype == ref_plane.dtype


def test_allocate_kv_cache_v41_rejects_mixed_specs(monkeypatch):
    _make_v41_runtime(monkeypatch)
    cache_config = make_cache_config(num_blocks=2, block_size=64, head_size=128, index_size=64)
    cache_config.kv_cache_groups[-1].kv_cache_spec = AscendMLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="Mixed V4.1"):
        attn_utils._allocate_kv_cache(cache_config, {}, torch.device("cpu"))


def test_v41_eager_fallback_forces_skip_compiled_for_runtime_none():
    from vllm_ascend.worker.v2 import model_runner as v2_model_runner

    calls: list[dict[str, Any]] = []

    @contextmanager
    def fake_original(*args, **kwargs):
        calls.append(kwargs.copy())
        yield

    original_fn = vllm_model_runner.set_forward_context
    original_flag = v2_model_runner._V41_EAGER_FALLBACK_INSTALLED
    v2_model_runner._V41_EAGER_FALLBACK_INSTALLED = False
    vllm_model_runner.set_forward_context = fake_original
    try:
        v2_model_runner._install_v41_eager_fallback()

        v41_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v4.1"),
                hf_text_config=SimpleNamespace(model_type="deepseek_v4.1_text"),
            )
        )
        llama_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="llama"),
                hf_text_config=SimpleNamespace(model_type="llama"),
            )
        )

        with vllm_model_runner.set_forward_context(None, v41_config, cudagraph_runtime_mode=CUDAGraphMode.NONE):
            pass
        assert calls[-1]["skip_compiled"] is True

        # Uniform decode (FULL) still runs the compiled full graph.
        with vllm_model_runner.set_forward_context(None, v41_config, cudagraph_runtime_mode=CUDAGraphMode.FULL):
            pass
        assert "skip_compiled" not in calls[-1]

        # Non-V4.1 models keep upstream behavior untouched.
        with vllm_model_runner.set_forward_context(None, llama_config, cudagraph_runtime_mode=CUDAGraphMode.NONE):
            pass
        assert "skip_compiled" not in calls[-1]
    finally:
        vllm_model_runner.set_forward_context = original_fn
        v2_model_runner._V41_EAGER_FALLBACK_INSTALLED = original_flag
