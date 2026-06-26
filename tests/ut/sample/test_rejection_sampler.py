"""
Unit tests for rejection sampler functions including block verify and entropy verify.

These tests verify the correctness of:
1. Standard rejection sampling (rejection_random_sample_pytorch)
2. Block verify rejection sampling (rejection_random_sample_block_verify_pytorch)
3. Recovered token sampling (sample_recovered_tokens_pytorch, sample_recovered_tokens_blockwise_pytorch)
4. Greedy rejection sampling (rejection_greedy_sample_pytorch)
5. Entropy verify mode that modifies acceptance threshold based on distribution entropy
"""

from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.rejection_sampler import (
    expand_batch_to_tokens,
    expand_pytorch,
    rejection_greedy_sample_pytorch,
    rejection_random_sample_block_verify_pytorch,
    rejection_random_sample_pytorch,
    sample_recovered_tokens_blockwise_pytorch,
    sample_recovered_tokens_pytorch,
)

# Global constants (mirroring vLLM values)
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0
MAX_SPEC_LEN = 8


def mock_pin_memory(original_func):
    """Decorator to remove pin_memory=True from tensor operations for testing."""

    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


class TestRejectionRandomSamplePytorch(TestBase):
    """Tests for standard rejection_random_sample_pytorch function.

    Acceptance logic:
        - ratio = target_prob / draft_prob
        - If draft_prob > 0 and ratio >= uniform_prob: accept (use draft token)
        - Else: reject (use recovered token at first rejection)
        - If all accepted: bonus token at position = num_draft
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_all_accepted(self):
        """All draft tokens should be accepted when ratio >= uniform for all positions."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        # draft_probs[pos, token] = P(draft_token | pos)
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1 with P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0 with P=0.8
            ]
        )
        # target_probs[pos, token] = P(target_token | pos)
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: token 1 with P=0.9 (ratio = 0.9/0.6 = 1.5 >= 0.5)
                [0.9, 0.1, 0.0],  # pos 1: token 0 with P=0.9 (ratio = 0.9/0.8 = 1.125 >= 0.6)
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Both positions accepted: output = [draft_token_0, draft_token_1, bonus]
        assert output_token_ids[0, 0].item() == 1  # draft token at pos 0
        assert output_token_ids[0, 1].item() == 0  # draft token at pos 1
        assert output_token_ids[0, 2].item() == 100  # bonus token

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_first_rejected(self):
        """First token rejected should use recovered token at position 1."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1 with P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0 with P=0.8
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.3, 0.7],  # pos 0: token 1 with P=0.3 (ratio = 0.3/0.6 = 0.5 < 0.7 -> REJECT)
                [0.9, 0.1, 0.0],  # pos 1: won't be reached due to rejection at pos 0
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # First position rejected: output = [recovered_0, PLACEHOLDER, PLACEHOLDER]
        assert output_token_ids[0, 0].item() == 99  # recovered at pos 0
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_second_rejected(self):
        """Second token rejected should use recovered token at position 1."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1 with P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0 with P=0.8
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: token 1 with P=0.9 (ratio = 0.9/0.6 = 1.5 >= 0.5 -> ACCEPT)
                [0.1, 0.1, 0.8],  # pos 1: token 0 with P=0.1 (ratio = 0.1/0.8 = 0.125 < 0.6 -> REJECT)
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # First accepted, second rejected: output = [draft_0, recovered_1, PLACEHOLDER]
        assert output_token_ids[0, 0].item() == 1  # accepted draft at pos 0
        assert output_token_ids[0, 1].item() == 88  # recovered at pos 1
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_ngram_mode(self):
        """In NGRAM mode (draft_probs=None), all draft tokens have probability 1.0.

        Acceptance depends only on target probability since draft_prob=1.0.
        ratio = target_prob / 1.0 = target_prob
        """
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None  # NGRAM mode
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],  # pos 0: token 0 with P=0.6 (>= 0.5 -> ACCEPT)
                [0.1, 0.1, 0.8],  # pos 1: token 1 with P=0.1 (< 0.6 -> REJECT)
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
        )

        # First accepted (target_prob=0.6 >= 0.5), second rejected (target_prob=0.1 < 0.6)
        assert output_token_ids[0, 0].item() == 0  # accepted draft at pos 0
        assert output_token_ids[0, 1].item() == 88  # recovered at pos 1
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_multi_batch(self):
        """Test with multiple batches having different rejection points."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1, 2])
        # Batch 0: 1 token at position 0
        # Batch 1: 2 tokens at positions 1, 2
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0 (batch 0): token 1, P=0.6
                [0.8, 0.2, 0.0],  # pos 1 (batch 1): token 0, P=0.8
                [0.2, 0.3, 0.5],  # pos 2 (batch 1): token 2, P=0.5
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: token 1, P=0.9 (ratio=1.5 >= 0.5 -> ACCEPT)
                [0.1, 0.1, 0.8],  # pos 1: token 0, P=0.1 (ratio=0.125 < 0.6 -> REJECT)
                [0.2, 0.3, 0.5],  # pos 2: token 2, P=0.5 (ratio=1.0 >= 0.5 -> but pos 1 rejected)
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.5, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Batch 0: 1 token accepted -> [1, 100, -1]
        assert output_token_ids[0, 0].item() == 1  # accepted
        assert output_token_ids[0, 1].item() == 100  # bonus (only 1 draft token)
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

        # Batch 1: first accepted, second rejected -> [0, 77, -1]
        assert output_token_ids[1, 0].item() == 0  # accepted draft
        assert output_token_ids[1, 1].item() == 77  # recovered at rejection
        assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_random_sample_placeholder_tokens(self):
        """Placeholder tokens (PLACEHOLDER_TOKEN_ID) should always be rejected."""
        batch_size = 1
        max_spec_len = 1
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1])
        draft_token_ids = torch.tensor([PLACEHOLDER_TOKEN_ID])
        draft_probs = None  # NGRAM mode
        target_probs = torch.tensor([[0.0, 0.0, 1.0]])
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([2])
        uniform_probs = torch.tensor([0.0])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
        )

        # Placeholder rejected, recovered token used
        assert output_token_ids[0, 0].item() == 2  # recovered
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID


class TestRejectionRandomSampleBlockVerify(TestBase):
    """Tests for block verify rejection sampling.

    Block verify uses cumulative product acceptance:
        - pi[i] = min(pi[i-1] * (target_prob/draft_prob), 1.0)
        - cum_uniform[i] = cum_uniform[i-1] * uniform_prob[i]
        - Token i accepted if pi[i] >= cum_uniform[i]
        - Uses last accepted position for bonus token placement
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_block_verify_all_accepted(self):
        """All tokens accepted when cumulative product stays above threshold."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1, P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0, P=0.8
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: ratio = 0.9/0.6 = 1.5
                [0.9, 0.1, 0.0],  # pos 1: ratio = 0.9/0.8 = 1.125
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Both accepted with block verify:
        # pos 0: pi=1.0*1.5=1.5, cum_uniform=0.5, 1.5>=0.5 -> accept
        # pos 1: pi=1.0*1.125=1.125, cum_uniform=0.5*0.6=0.3, 1.125>=0.3 -> accept
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100  # bonus

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_block_verify_second_rejected(self):
        """Second token rejected due to cumulative threshold."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1, P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0, P=0.8
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: ratio = 0.9/0.6 = 1.5 (high acceptance)
                [0.1, 0.1, 0.8],  # pos 1: ratio = 0.1/0.8 = 0.125 (low)
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Block verify with cumulative threshold:
        # pos 0: pi=1.0, cum=0.5, 1.0*1.5=1.5>=0.5 -> accept, last=0
        # pos 1: pi=1.5*0.125=0.1875, cum=0.5*0.6=0.3, 0.1875<0.3 -> reject
        assert output_token_ids[0, 0].item() == 1  # accepted
        assert output_token_ids[0, 1].item() == 88  # recovered at rejection pos
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_block_verify_first_rejected(self):
        """First token rejected immediately."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0: token 1, P=0.6
                [0.8, 0.2, 0.0],  # pos 1: token 0, P=0.8
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.3, 0.7],  # pos 0: ratio = 0.3/0.6 = 0.5
                [0.9, 0.1, 0.0],  # pos 1: ratio = 0.9/0.8 = 1.125
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Block verify:
        # pos 0: pi=1.0, cum=0.7, 1.0*0.5=0.5<0.7 -> reject immediately
        assert output_token_ids[0, 0].item() == 99  # recovered at pos 0
        assert output_token_ids[0, 1].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_block_verify_ngram_mode(self):
        """Block verify with NGRAM mode (draft_probs=None)."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([0, 1])
        draft_probs = None  # NGRAM mode
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],  # pos 0: token 0, P=0.6
                [0.1, 0.8, 0.1],  # pos 1: token 1, P=0.8
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=True,
        )

        # Block verify with draft_prob=1.0 for all:
        # pos 0: pi=1.0*0.6=0.6, cum=0.5, 0.6>=0.5 -> accept, last=0
        # pos 1: pi=0.6*0.8=0.48, cum=0.5*0.6=0.3, 0.48>=0.3 -> accept, last=1
        assert output_token_ids[0, 0].item() == 0
        assert output_token_ids[0, 1].item() == 1
        assert output_token_ids[0, 2].item() == 100  # bonus

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_block_verify_multi_batch(self):
        """Block verify with multiple batches."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 0, 2])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],  # pos 0 (batch 0): token 1, P=0.6
                [0.8, 0.2, 0.0],  # pos 1 (batch 1): token 0, P=0.8
                [0.3, 0.3, 0.4],  # pos 2 (batch 1): token 2, P=0.4
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],  # pos 0: ratio = 0.9/0.6 = 1.5
                [0.9, 0.1, 0.0],  # pos 1: ratio = 0.9/0.8 = 1.125
                [0.2, 0.3, 0.5],  # pos 2: ratio = 0.5/0.4 = 1.25
            ]
        )
        bonus_token_ids = torch.tensor([[100], [200]])
        recovered_token_ids = torch.tensor([99, 88, 77])
        uniform_probs = torch.tensor([0.5, 0.6, 0.5])
        is_greedy = torch.tensor([False, False])
        vocab_size = 3

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
        )

        # Batch 0 (1 token): pos 0 -> pi=1.5, cum=0.5, accept, bonus at pos 1
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 100
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID

        # Batch 1 (2 tokens): pos 1 -> pi=1.125, cum=0.6, accept; pos 2 -> pi=1.125*1.25=1.406, cum=0.3, accept
        assert output_token_ids[1, 0].item() == 0
        assert output_token_ids[1, 1].item() == 2
        assert output_token_ids[1, 2].item() == 200  # bonus


class TestEntropyVerify(TestBase):
    """Tests for ENTROPY_VERIFY mode.

    Entropy verify modifies acceptance threshold based on target distribution entropy:
        - entropy = -sum(p * log(p)) for ori_target_probs
        - exp_neg_entropy = exp(-entropy * POSTERIOR_ALPHA)
        - threshold = min(exp_neg_entropy, POSTERIOR_THRESHOLD)
        - modified_uniform = threshold * uniform_prob

    High entropy (uniform-like distribution) -> lower threshold -> more accepting
    Low entropy (peaked distribution) -> higher threshold (near POSTERIOR_THRESHOLD) -> stricter
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_high_entropy_accepts_more(self):
        """High entropy (near-uniform) should lower threshold, making acceptance easier."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.2],
                [0.1, 0.9, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # ori_target_probs with HIGH entropy (uniform-like: [0.33, 0.33, 0.34])
        ori_target_probs = torch.tensor(
            [
                [0.33, 0.33, 0.34],
                [0.33, 0.34, 0.33],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # With high entropy, threshold is lowered:
        # pos 0: ratio=0.8/0.6=1.33, uniform=0.7, threshold=low->accept
        # pos 1: ratio=0.9/0.8=1.125, uniform=0.6, threshold=low->accept
        # Actually need to compute: entropy of [0.33, 0.33, 0.34] ~ 1.098, exp(-1.098*0.4) ~ 0.64
        # threshold = min(0.64, 0.95) = 0.64
        # modified_uniform = 0.64 * 0.7 = 0.448 for pos 0
        # ratio 1.33 >= 0.448 -> ACCEPT
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_low_entropy_stricter(self):
        """Low entropy (peaked) should keep threshold high, making acceptance stricter."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],
                [0.8, 0.2, 0.0],
            ]
        )
        # Borderline target probs where entropy affects acceptance
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.2],  # pos 0: ratio=1.33
                [0.1, 0.9, 0.0],  # pos 1: ratio=1.125
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # ori_target_probs with LOW entropy (peaked: [0.01, 0.98, 0.01])
        ori_target_probs = torch.tensor(
            [
                [0.01, 0.98, 0.01],
                [0.01, 0.98, 0.01],
            ]
        )

        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # With low entropy, threshold stays near 0.95:
        # entropy of [0.01, 0.98, 0.01] ~ 0.06, exp(-0.06*0.4) ~ 0.976
        # threshold = min(0.976, 0.95) = 0.95
        # modified_uniform = 0.95 * 0.7 = 0.665 for pos 0
        # ratio 1.33 >= 0.665 -> ACCEPT
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_no_ori_probs_uses_target(self):
        """When ori_target_probs is None, fallback to target_probs for entropy."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.8, 0.2],
                [0.1, 0.9, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.7, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3

        # Without ori_target_probs, use target_probs for entropy calculation
        rejection_random_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=None,
        )

        # Should not crash and should produce valid output
        # Uses target_probs for entropy when ori_target_probs is None
        assert output_token_ids[0, 0].item() in [1, 99]
        assert output_token_ids[0, 1].item() in [0, 88, PLACEHOLDER_TOKEN_ID]

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_entropy_verify_block_verify_combined(self):
        """Test entropy verify combined with block verify."""
        batch_size = 1
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2])
        draft_token_ids = torch.tensor([1, 0])
        draft_probs = torch.tensor(
            [
                [0.0, 0.6, 0.4],
                [0.8, 0.2, 0.0],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.1, 0.0],
            ]
        )
        bonus_token_ids = torch.tensor([[100]])
        recovered_token_ids = torch.tensor([99, 88])
        uniform_probs = torch.tensor([0.5, 0.6])
        is_greedy = torch.tensor([False])
        vocab_size = 3
        ori_target_probs = torch.tensor(
            [
                [0.33, 0.33, 0.34],
                [0.33, 0.34, 0.33],
            ]
        )

        rejection_random_sample_block_verify_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            IS_NGRAM=False,
            ENTROPY_VERIFY=True,
            POSTERIOR_THRESHOLD=0.95,
            POSTERIOR_ALPHA=0.4,
            EPSILON=1e-10,
            ori_target_probs=ori_target_probs,
        )

        # With block verify + entropy:
        # High entropy -> low threshold
        # pos 0: ratio=1.5, cum=0.5, threshold=low -> pi=1.5>=modified_cum -> accept
        # pos 1: ratio=1.125, cum=0.5*0.6=0.3, pi=1.5*1.125=1.6875>=modified_cum -> accept
        assert output_token_ids[0, 0].item() == 1
        assert output_token_ids[0, 1].item() == 0
        assert output_token_ids[0, 2].item() == 100


class TestSampleRecoveredTokens(TestBase):
    """Tests for recovered token sampling functions.

    When a draft token is rejected, we need to sample a recovered token from
    a modified distribution: max(0, target_prob - draft_prob)
    """

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_basic(self):
        """Test basic recovered token sampling."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.1, 0.3],  # token 0: P=0.6, residual for token 0: max(0, 0.8-0.6)=0.2
                [0.2, 0.7, 0.1],  # token 1: P=0.7, residual for token 1: max(0, 0.6-0.7)=0
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.1, 0.1],  # token 0
                [0.3, 0.3, 0.4],  # token 1
            ]
        )
        q = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # uniform normalization
                [1.0, 1.0, 1.0],
            ]
        )
        vocab_size = 3

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
        )

        # For token 0: residual = [0.8-0.6, 0.1-0.1, 0.1-0.3] = [0.2, 0, 0] -> argmax = 0
        # For token 1: residual = [0.3-0.2, 0.3-0.7, 0.4-0.1] = [0.1, 0, 0.3] -> argmax = 2
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 2

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_ngram(self):
        """Test recovered token sampling in NGRAM mode (draft_probs=None)."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = None  # NGRAM mode
        target_probs = torch.tensor(
            [
                [0.6, 0.2, 0.2],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        vocab_size = 3

        sample_recovered_tokens_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=True,
        )

        # NGRAM mode: residual = target_probs (draft contribution zeroed out)
        # For token 0: residual = [0.6, 0.2, 0.2] -> argmax = 0
        # For token 1: residual = [0.3, 0.3, 0.4] -> argmax = 2
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 2

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_sample_recovered_tokens_blockwise(self):
        """Test block-wise recovered token sampling with prefix product."""
        output_token_ids = torch.empty(2, dtype=torch.int32)
        cu_num_draft_tokens = torch.tensor([1, 2])
        draft_token_ids = torch.tensor([1, 2])
        draft_probs = torch.tensor(
            [
                [0.6, 0.1, 0.3],
                [0.2, 0.7, 0.1],
            ]
        )
        target_probs = torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        q = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        vocab_size = 3
        target_indices = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
            ]
        )

        sample_recovered_tokens_blockwise_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=False,
            target_indices=target_indices,
            enable_reduce_sampling=True,
        )

        # Blockwise uses prefix product p_i for residual calculation
        # Token 0: ratio = 0.8/0.6 = 1.33, p_0 = min(1.0*1.33, 1.0) = 1.0
        # Token 1: ratio = 0.3/0.7 = 0.43, p_1 = min(1.0*0.43, 1.0) = 0.43
        # residual = p_i * target_probs - draft_probs
        # For token 0: [1.0*0.8-0.6, 1.0*0.1-0.1, 1.0*0.1-0.3] = [0.2, 0, 0] -> argmax = 0
        # For token 1: [0.43*0.3-0.2, 0.43*0.3-0.7, 0.43*0.4-0.1] = [-0.071, -0.571, 0.072] -> argmax = 2
        assert output_token_ids[0].item() == 0
        assert output_token_ids[1].item() == 2


class TestExpandFunctions(TestBase):
    """Tests for batch-to-token expansion functions."""

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_pytorch(self):
        """Test expand_pytorch broadcasts batch values to tokens."""
        input_ptr = torch.tensor([10, 20, 30], dtype=torch.int32)
        cu_num_tokens_ptr = torch.tensor([2, 5, 7])
        output_ptr = torch.empty(7, dtype=torch.int32)

        expand_pytorch(
            output_ptr,
            input_ptr,
            cu_num_tokens_ptr,
            replace_from=0,
            replace_to=0,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,
        )

        expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
        assert torch.equal(output_ptr, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_batch_to_tokens(self):
        """Test expand_batch_to_tokens wrapper function."""
        x = torch.tensor([10, 20, 30])
        cu_num_tokens = torch.tensor([2, 5, 7])
        num_tokens = 7

        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            result = expand_batch_to_tokens(x, cu_num_tokens, num_tokens)
            expected = torch.tensor([10, 10, 20, 20, 20, 30, 30])
            assert torch.equal(result, expected)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_expand_batch_to_tokens_with_replacement(self):
        """Test expand_batch_to_tokens with value replacement."""
        x = torch.tensor([0, 20, 30])  # first value is 0
        cu_num_tokens = torch.tensor([2, 5, 7])
        num_tokens = 7

        with patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False):
            result = expand_batch_to_tokens(x, cu_num_tokens, num_tokens, replace_from=0, replace_to=100)
            expected = torch.tensor([100, 100, 20, 20, 20, 30, 30])
            assert torch.equal(result, expected)


class TestRejectionGreedySample(TestBase):
    """Tests for greedy rejection sampling."""

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_all_match(self):
        """All draft tokens match target when using greedy (argmax)."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 4])
        num_draft_tokens = [2, 2]
        draft_token_ids = torch.tensor([10, 11, 20, 21])
        target_argmax = torch.tensor([10, 11, 20, 21])  # All match
        bonus_token_ids = torch.tensor([[100], [200]])

        is_greedy = torch.tensor([True, True])

        rejection_greedy_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len,
            is_greedy,
        )

        # All draft tokens accepted, bonus tokens appended
        assert output_token_ids[0, 0].item() == 10
        assert output_token_ids[0, 1].item() == 11
        assert output_token_ids[0, 2].item() == 100
        assert output_token_ids[1, 0].item() == 20
        assert output_token_ids[1, 1].item() == 21
        assert output_token_ids[1, 2].item() == 200

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_mismatch(self):
        """First mismatch causes rejection at that position."""
        batch_size = 2
        max_spec_len = 2
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        cu_num_draft_tokens = torch.tensor([2, 4])
        num_draft_tokens = [2, 2]
        draft_token_ids = torch.tensor([10, 11, 20, 21])
        target_argmax = torch.tensor([10, 99, 20, 22])  # Mismatch at positions 1 and 3
        bonus_token_ids = torch.tensor([[100], [200]])

        is_greedy = torch.tensor([True, True])

        rejection_greedy_sample_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_draft_tokens,
            max_spec_len,
            is_greedy,
        )

        # Batch 0: pos 0 matches, pos 1 mismatches (use target_argmax[1]=99)
        # Batch 1: pos 0 matches, pos 1 mismatches (use target_argmax[3]=22)
        assert output_token_ids[0, 0].item() == 10
        assert output_token_ids[0, 1].item() == 99  # target at mismatch
        assert output_token_ids[0, 2].item() == PLACEHOLDER_TOKEN_ID
        assert output_token_ids[1, 0].item() == 20
        assert output_token_ids[1, 1].item() == 22  # target at mismatch
        assert output_token_ids[1, 2].item() == PLACEHOLDER_TOKEN_ID

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    def test_rejection_greedy_sample_spec_len_1(self):
        """Test special case with spec_len=1."""
        batch_size = 2
        max_spec_len = 1
        output_token_ids = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID)

        draft_token_ids = torch.tensor([10, 20])
        target_argmax = torch.tensor([10, 99])  # Batch 0 matches, batch 1 mismatches
        bonus_token_ids = torch.tensor([[100], [200]])

        # Use spec_len=1 version for efficiency
        from vllm_ascend.sample.rejection_sampler import rejection_greedy_sample_spec_len_1_pytorch

        rejection_greedy_sample_spec_len_1_pytorch(
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
        )

        assert output_token_ids[0, 0].item() == 10
        assert output_token_ids[0, 1].item() == 100  # bonus
        assert output_token_ids[1, 0].item() == 99  # target (mismatch)
        assert output_token_ids[1, 1].item() == 200  # bonus
