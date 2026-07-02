import os
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler_dfx import (
    RejectionSamplerDFX,
    RejectionSamplerErrorCode,
    RejectionSamplerMetrics,
)


class TestRejectionSamplerDFX(TestBase):
    def setUp(self):
        self.original_env = os.environ.get("VLLM_ASCEND_REJECTION_SAMPLER_DFX", "0")

    def tearDown(self):
        os.environ["VLLM_ASCEND_REJECTION_SAMPLER_DFX"] = self.original_env
        RejectionSamplerDFX._instance = None

    def test_dfx_singleton(self):
        dfx1 = RejectionSamplerDFX()
        dfx2 = RejectionSamplerDFX()
        self.assertIs(dfx1, dfx2)

    def test_dfx_disabled_by_default(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "0"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            self.assertFalse(dfx.enabled)

    def test_dfx_enabled(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            self.assertTrue(dfx.enabled)

    def test_metrics_initialization(self):
        metrics = RejectionSamplerMetrics()
        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.acceptance_rate, 0.0)
        self.assertEqual(metrics.triton_fallback_ratio, 0.0)

    def test_acceptance_rate_calculation(self):
        metrics = RejectionSamplerMetrics()
        metrics.accepted_tokens = 80
        metrics.total_draft_tokens = 100
        self.assertEqual(metrics.acceptance_rate, 0.8)

        metrics.total_draft_tokens = 0
        self.assertEqual(metrics.acceptance_rate, 0.0)

    def test_triton_fallback_ratio(self):
        metrics = RejectionSamplerMetrics()
        metrics.triton_kernel_launches = 90
        metrics.pytorch_fallback_executions = 10
        self.assertAlmostEqual(metrics.triton_fallback_ratio, 0.1)

    def test_reduce_sample_hit_rate(self):
        metrics = RejectionSamplerMetrics()
        metrics.reduce_sample_path_hits = 80
        metrics.fallback_path_hits = 20
        self.assertEqual(metrics.reduce_sample_hit_rate, 0.8)

    def test_record_request(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_request(is_greedy=True)
            dfx.record_request(is_greedy=False)
            self.assertEqual(dfx.metrics.total_requests, 2)
            self.assertEqual(dfx.metrics.greedy_requests, 1)
            self.assertEqual(dfx.metrics.random_requests, 1)

    def test_record_draft_tokens(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_draft_tokens(10)
            dfx.record_draft_tokens(20)
            self.assertEqual(dfx.metrics.total_draft_tokens, 30)

    def test_validate_tensor_shape_mismatch(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            tensor = torch.randn(2, 3, 4)
            result = dfx.validate_tensor(tensor, "test", expected_ndim=2)
            self.assertEqual(result, RejectionSamplerErrorCode.ERR_TENSOR_SHAPE_MISMATCH)

    def test_validate_tensor_shape_ok(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            tensor = torch.randn(2, 3)
            result = dfx.validate_tensor(tensor, "test", expected_ndim=2)
            self.assertEqual(result, RejectionSamplerErrorCode.SUCCESS)

    def test_validate_tensor_not_contiguous(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            tensor = torch.randn(2, 3)[:, ::-1]
            result = dfx.validate_tensor(tensor, "test", check_contiguous=True)
            self.assertEqual(result, RejectionSamplerErrorCode.ERR_TENSOR_NOT_CONTIGUOUS)

    def test_validate_tensor_contiguous(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            tensor = torch.randn(2, 3).contiguous()
            result = dfx.validate_tensor(tensor, "test", check_contiguous=True)
            self.assertEqual(result, RejectionSamplerErrorCode.SUCCESS)

    def test_check_placeholder_leak(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            output = torch.tensor([[10, -1], [20, 21]])
            result = dfx.check_placeholder_leak(output)
            self.assertEqual(result, RejectionSamplerErrorCode.ERR_PLACEHOLDER_TOKEN_LEAK)

    def test_check_placeholder_no_leak(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            output = torch.tensor([[10, 11], [20, 21]])
            result = dfx.check_placeholder_leak(output)
            self.assertEqual(result, RejectionSamplerErrorCode.SUCCESS)

    def test_check_acceptance_rate_anomaly(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.metrics.total_draft_tokens = 200
            dfx.metrics.accepted_tokens = 0
            result = dfx.check_acceptance_rate_anomaly()
            self.assertEqual(result, RejectionSamplerErrorCode.ERR_ACCEPTANCE_RATE_ANOMALY)

    def test_check_acceptance_rate_normal(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.metrics.total_draft_tokens = 100
            dfx.metrics.accepted_tokens = 50
            result = dfx.check_acceptance_rate_anomaly()
            self.assertEqual(result, RejectionSamplerErrorCode.SUCCESS)

    def test_get_metrics_summary(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_request(is_greedy=True)
            dfx.record_draft_tokens(10)
            dfx.record_accepted_tokens(8)
            summary = dfx.get_metrics_summary()
            self.assertEqual(summary["total_requests"], 1)
            self.assertEqual(summary["total_draft_tokens"], 10)
            self.assertEqual(summary["accepted_tokens"], 8)
            self.assertAlmostEqual(summary["acceptance_rate"], 0.8)

    def test_error_code_to_string(self):
        self.assertEqual(RejectionSamplerErrorCode.to_string(0), "Success")
        self.assertEqual(RejectionSamplerErrorCode.to_string(1001), "Tensor shape mismatch")
        self.assertEqual(RejectionSamplerErrorCode.to_string(9999), "Unknown error code: 9999")

    def test_record_triton_kernel_launch(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_triton_kernel_launch()
            self.assertEqual(dfx.metrics.triton_kernel_launches, 1)

    def test_record_pytorch_fallback(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_pytorch_fallback()
            self.assertEqual(dfx.metrics.pytorch_fallback_executions, 1)

    def test_record_reduce_sample_path(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_reduce_sample_path(True)
            dfx.record_reduce_sample_path(False)
            self.assertEqual(dfx.metrics.reduce_sample_path_hits, 1)
            self.assertEqual(dfx.metrics.fallback_path_hits, 1)

    def test_record_verify_enabled(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_verify_enabled(True, False)
            dfx.record_verify_enabled(False, True)
            self.assertEqual(dfx.metrics.block_verify_enabled, 1)
            self.assertEqual(dfx.metrics.entropy_verify_enabled, 1)

    def test_reset_metrics(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_REJECTION_SAMPLER_DFX": "1"}):
            RejectionSamplerDFX._instance = None
            dfx = RejectionSamplerDFX()
            dfx.record_request(is_greedy=True)
            dfx.reset_metrics()
            self.assertEqual(dfx.metrics.total_requests, 0)
