"""Shape and behavior tests for the teaching Seq2Seq implementation."""

from __future__ import annotations

import unittest

import torch

from seq2seq_from_scratch.model import (
    AttentionDecoderGRU,
    EncoderGRU,
    Seq2Seq,
)


class Seq2SeqTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.encoder = EncoderGRU(20, 6, 8)
        self.decoder = AttentionDecoderGRU(25, 7, 8)
        self.model = Seq2Seq(
            self.encoder,
            self.decoder,
            start_id=1,
            end_id=2,
        )
        self.source = torch.randint(3, 20, (3, 5))

    def test_encoder_shapes(self) -> None:
        output, hidden = self.encoder(self.source)
        self.assertEqual(output.shape, (3, 5, 8))
        self.assertEqual(hidden.shape, (1, 3, 8))

    def test_attention_probabilities(self) -> None:
        output, hidden = self.encoder(self.source)
        logits, new_hidden, weights = self.decoder.forward_step(
            torch.ones(3, dtype=torch.long),
            hidden,
            output,
        )
        self.assertEqual(logits.shape, (3, 25))
        self.assertEqual(new_hidden.shape, (1, 3, 8))
        self.assertTrue(
            torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-6)
        )

    def test_masked_source_positions_receive_zero_weight(self) -> None:
        output, hidden = self.encoder(self.source)
        mask = torch.tensor([[True, True, False, False, False]] * 3)
        _logits, _hidden, weights = self.decoder.forward_step(
            torch.ones(3, dtype=torch.long),
            hidden,
            output,
            mask,
        )
        self.assertTrue(torch.equal(weights[:, 2:], torch.zeros(3, 3)))

    def test_training_output_shape(self) -> None:
        target = torch.randint(3, 25, (3, 7))
        target[:, 0] = 1
        logits = self.model(
            self.source,
            target,
            teacher_forcing_ratio=1.0,
        )
        self.assertEqual(logits.shape, (3, 6, 25))

    def test_greedy_decode_shapes(self) -> None:
        ids, weights = self.model.greedy_decode(
            self.source,
            maximum_length=4,
        )
        self.assertEqual(ids.shape[0], 3)
        self.assertLessEqual(ids.shape[1], 4)
        self.assertEqual(weights.shape, (3, ids.shape[1], 5))


if __name__ == "__main__":
    unittest.main()
