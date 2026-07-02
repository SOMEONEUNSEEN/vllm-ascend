# SPDX-License-Identifier: Apache-2.0

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


class RejectionSamplerErrorCode:
    """Error codes for rejection sampling DFX diagnostics."""

    SUCCESS = 0
    ERR_TENSOR_SHAPE_MISMATCH = 1001
    ERR_TENSOR_NOT_CONTIGUOUS = 1002
    ERR_TENSOR_DEVICE_MISMATCH = 1003
    ERR_TRITON_KERNEL_LAUNCH_FAILED = 2001
    ERR_REDUCE_SAMPLE_CONFIG_MISMATCH = 3001
    ERR_PLACEHOLDER_TOKEN_LEAK = 4001
    ERR_ACCEPTANCE_RATE_ANOMALY = 4002
    ERR_BLOCK_VERIFY_CONFIG_INVALID = 5001
    ERR_ENTROPY_VERIFY_MISSING_LOGITS = 5002
    ERR_GREEDY_ARGMAX_MISMATCH = 6001

    @classmethod
    def to_string(cls, code: int) -> str:
        mapping = {
            cls.SUCCESS: "Success",
            cls.ERR_TENSOR_SHAPE_MISMATCH: "Tensor shape mismatch",
            cls.ERR_TENSOR_NOT_CONTIGUOUS: "Tensor is not contiguous",
            cls.ERR_TENSOR_DEVICE_MISMATCH: "Tensor device mismatch",
            cls.ERR_TRITON_KERNEL_LAUNCH_FAILED: "Triton kernel launch failed",
            cls.ERR_REDUCE_SAMPLE_CONFIG_MISMATCH: "Reduce sample config/data mismatch",
            cls.ERR_PLACEHOLDER_TOKEN_LEAK: "Placeholder token (-1) leaked in output",
            cls.ERR_ACCEPTANCE_RATE_ANOMALY: "Acceptance rate anomaly detected",
            cls.ERR_BLOCK_VERIFY_CONFIG_INVALID: "Block verify config invalid",
            cls.ERR_ENTROPY_VERIFY_MISSING_LOGITS: "Entropy verify missing original logits",
            cls.ERR_GREEDY_ARGMAX_MISMATCH: "Greedy argmax mismatch across TP ranks",
        }
        return mapping.get(code, f"Unknown error code: {code}")


@dataclass
class RejectionSamplerMetrics:
    """Metrics collected during rejection sampling."""

    total_requests: int = 0
    greedy_requests: int = 0
    random_requests: int = 0
    accepted_tokens: int = 0
    total_draft_tokens: int = 0
    recovered_tokens: int = 0
    bonus_tokens_added: int = 0
    triton_kernel_launches: int = 0
    pytorch_fallback_executions: int = 0
    reduce_sample_path_hits: int = 0
    fallback_path_hits: int = 0
    block_verify_enabled: int = 0
    entropy_verify_enabled: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @property
    def triton_fallback_ratio(self) -> float:
        total = self.triton_kernel_launches + self.pytorch_fallback_executions
        if total == 0:
            return 0.0
        return self.pytorch_fallback_executions / total

    @property
    def reduce_sample_hit_rate(self) -> float:
        total = self.reduce_sample_path_hits + self.fallback_path_hits
        if total == 0:
            return 0.0
        return self.reduce_sample_path_hits / total


class RejectionSamplerDFX:
    """DFX module for rejection sampling diagnostics and metrics.

    This module provides:
    - Metrics collection (acceptance rate, fallback ratio, etc.)
    - Error code system
    - Tensor validation
    - Diagnostic logging
    """

    _instance: Optional["RejectionSamplerDFX"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RejectionSamplerDFX":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self.enabled = bool(int(os.getenv("VLLM_ASCEND_REJECTION_SAMPLER_DFX", "0")))
        self.metrics = RejectionSamplerMetrics()
        self._metrics_lock = threading.Lock()
        self._last_log_time = 0

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self.metrics = RejectionSamplerMetrics()

    def record_request(self, is_greedy: bool) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.total_requests += 1
            if is_greedy:
                self.metrics.greedy_requests += 1
            else:
                self.metrics.random_requests += 1

    def record_draft_tokens(self, num_draft_tokens: int) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.total_draft_tokens += num_draft_tokens

    def record_accepted_tokens(self, num_accepted: int) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.accepted_tokens += num_accepted

    def record_recovered_tokens(self, num_recovered: int) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.recovered_tokens += num_recovered

    def record_bonus_token(self, added: bool) -> None:
        if not self.enabled or not added:
            return
        with self._metrics_lock:
            self.metrics.bonus_tokens_added += 1

    def record_triton_kernel_launch(self) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.triton_kernel_launches += 1

    def record_pytorch_fallback(self) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            self.metrics.pytorch_fallback_executions += 1

    def record_reduce_sample_path(self, is_reduce_sample: bool) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            if is_reduce_sample:
                self.metrics.reduce_sample_path_hits += 1
            else:
                self.metrics.fallback_path_hits += 1

    def record_verify_enabled(self, block_verify: bool, entropy_verify: bool) -> None:
        if not self.enabled:
            return
        with self._metrics_lock:
            if block_verify:
                self.metrics.block_verify_enabled += 1
            if entropy_verify:
                self.metrics.entropy_verify_enabled += 1

    def validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_ndim: Optional[int] = None,
        check_contiguous: bool = True,
        expected_device: Optional[torch.device] = None,
    ) -> int:
        if not self.enabled:
            return RejectionSamplerErrorCode.SUCCESS

        if expected_ndim is not None and tensor.ndim != expected_ndim:
            return RejectionSamplerErrorCode.ERR_TENSOR_SHAPE_MISMATCH

        if check_contiguous and not tensor.is_contiguous():
            return RejectionSamplerErrorCode.ERR_TENSOR_NOT_CONTIGUOUS

        if expected_device is not None and tensor.device != expected_device:
            return RejectionSamplerErrorCode.ERR_TENSOR_DEVICE_MISMATCH

        return RejectionSamplerErrorCode.SUCCESS

    def check_acceptance_rate_anomaly(self) -> int:
        if not self.enabled:
            return RejectionSamplerErrorCode.SUCCESS

        rate = self.metrics.acceptance_rate
        if self.metrics.total_draft_tokens > 100:
            if rate == 0.0:
                return RejectionSamplerErrorCode.ERR_ACCEPTANCE_RATE_ANOMALY
            if rate < 0.01:
                return RejectionSamplerErrorCode.ERR_ACCEPTANCE_RATE_ANOMALY
        return RejectionSamplerErrorCode.SUCCESS

    def check_placeholder_leak(self, output_token_ids: torch.Tensor) -> int:
        if not self.enabled:
            return RejectionSamplerErrorCode.SUCCESS

        if (output_token_ids == -1).any():
            return RejectionSamplerErrorCode.ERR_PLACEHOLDER_TOKEN_LEAK
        return RejectionSamplerErrorCode.SUCCESS

    def get_metrics_summary(self) -> Dict[str, Any]:
        with self._metrics_lock:
            return {
                "total_requests": self.metrics.total_requests,
                "greedy_requests": self.metrics.greedy_requests,
                "random_requests": self.metrics.random_requests,
                "total_draft_tokens": self.metrics.total_draft_tokens,
                "accepted_tokens": self.metrics.accepted_tokens,
                "recovered_tokens": self.metrics.recovered_tokens,
                "bonus_tokens_added": self.metrics.bonus_tokens_added,
                "acceptance_rate": self.metrics.acceptance_rate,
                "triton_kernel_launches": self.metrics.triton_kernel_launches,
                "pytorch_fallback_executions": self.metrics.pytorch_fallback_executions,
                "triton_fallback_ratio": self.metrics.triton_fallback_ratio,
                "reduce_sample_path_hits": self.metrics.reduce_sample_path_hits,
                "fallback_path_hits": self.metrics.fallback_path_hits,
                "reduce_sample_hit_rate": self.metrics.reduce_sample_hit_rate,
                "block_verify_enabled": self.metrics.block_verify_enabled,
                "entropy_verify_enabled": self.metrics.entropy_verify_enabled,
            }

    def log_metrics_summary(self) -> None:
        if not self.enabled:
            return

        import time
        current_time = time.time()
        if current_time - self._last_log_time < 60:
            return
        self._last_log_time = current_time

        from vllm.logger import logger
        summary = self.get_metrics_summary()
        logger.info(
            "[sample/rejection_sampler_dfx] Metrics summary: "
            "total_requests=%s, acceptance_rate=%.4f, "
            "triton_fallback_ratio=%.4f, reduce_sample_hit_rate=%.4f",
            summary["total_requests"],
            summary["acceptance_rate"],
            summary["triton_fallback_ratio"],
            summary["reduce_sample_hit_rate"],
        )
