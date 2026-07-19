"""Component and end-to-end checks for the reconstructed Transformer."""

from __future__ import annotations

import unittest

import torch

from transformer_from_scratch.model import (
    LayerNorm,
    MultiHeadedAttention,
    PositionalEncoding,
    attention,
    make_model,
    subsequent_mask,
)


class TransformerComponentsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_subsequent_mask(self) -> None:
        mask = subsequent_mask(4)
        expected = torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, True],
                ]
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_positional_encoding_preserves_shape(self) -> None:
        layer = PositionalEncoding(8, dropout=0.0, max_len=16)
        x = torch.zeros(2, 5, 8)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y[0], y[1]))

    def test_attention_respects_mask(self) -> None:
        x = torch.randn(1, 4, 8)
        output, weights = attention(
            x,
            x,
            x,
            mask=subsequent_mask(4),
        )
        self.assertEqual(output.shape, (1, 4, 8))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(1, 4),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.equal(
                weights.triu(diagonal=1),
                torch.zeros_like(weights),
            )
        )

    def test_multi_head_shape_and_probabilities(self) -> None:
        layer = MultiHeadedAttention(4, 16, dropout=0.0)
        x = torch.randn(2, 5, 16)
        y = layer(x, x, x, subsequent_mask(5))
        self.assertEqual(y.shape, (2, 5, 16))
        self.assertIsNotNone(layer.attn)
        self.assertEqual(layer.attn.shape, (2, 4, 5, 5))
        self.assertTrue(
            torch.allclose(
                layer.attn.sum(dim=-1),
                torch.ones(2, 4, 5),
                atol=1e-6,
            )
        )

    def test_layer_norm_statistics(self) -> None:
        layer = LayerNorm(16)
        x = torch.randn(3, 5, 16)
        y = layer(x)
        self.assertTrue(
            torch.allclose(
                y.mean(dim=-1),
                torch.zeros(3, 5),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                y.var(dim=-1, unbiased=False),
                torch.ones(3, 5),
                atol=2e-5,
            )
        )

    def test_full_model(self) -> None:
        model = make_model(
            src_vocab=31,
            tgt_vocab=37,
            n=2,
            d_model=16,
            d_ff=32,
            h=4,
            dropout=0.0,
        )
        model.eval()

        src = torch.randint(0, 31, (2, 6))
        tgt = torch.randint(0, 37, (2, 5))
        src_mask = torch.ones(2, 1, 6, dtype=torch.bool)
        tgt_mask = subsequent_mask(5)

        with torch.no_grad():
            log_probs = model(src, tgt, src_mask, tgt_mask)

        self.assertEqual(log_probs.shape, (2, 5, 37))
        self.assertTrue(
            torch.allclose(
                log_probs.exp().sum(dim=-1),
                torch.ones(2, 5),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
