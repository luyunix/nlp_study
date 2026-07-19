"""Property tests for the teaching attention implementations."""

from __future__ import annotations

import unittest

import torch

from attention_from_scratch.model import DotProductAttention, GeneralAttention


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.query = torch.randn(2, 8)
        self.keys = torch.randn(2, 5, 8)
        self.values = torch.randn(2, 5, 8)

    def test_dot_attention_shapes_and_probability_sum(self) -> None:
        context, weights = DotProductAttention()(
            self.query,
            self.keys,
            self.values,
        )
        self.assertEqual(context.shape, (2, 8))
        self.assertEqual(weights.shape, (2, 5))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_masked_positions_receive_zero_probability(self) -> None:
        mask = torch.tensor(
            [[True, True, False, False, False], [True] * 5]
        )
        _context, weights = DotProductAttention()(
            self.query,
            self.keys,
            self.values,
            mask,
        )
        self.assertTrue(torch.equal(weights[0, 2:], torch.zeros(3)))

    def test_general_attention(self) -> None:
        context, weights = GeneralAttention(8)(
            self.query,
            self.keys,
            self.values,
        )
        self.assertEqual(context.shape, (2, 8))
        self.assertEqual(weights.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
