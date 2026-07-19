"""Behavior checks for the recurrent-model teaching code."""

from __future__ import annotations

import unittest

import torch

from rnn_from_scratch.model import NameClassifier, SimpleRNNCell


class RecurrentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_simple_cell_shape(self) -> None:
        cell = SimpleRNNCell(input_size=6, hidden_size=8)
        x_t = torch.randn(4, 6)
        h_previous = cell.initial_hidden(4)
        self.assertEqual(cell(x_t, h_previous).shape, (4, 8))

    def test_all_classifier_types_return_log_probabilities(self) -> None:
        sequence = torch.randn(3, 7, 12)
        for kind in ("rnn", "lstm", "gru"):
            with self.subTest(kind=kind):
                model = NameClassifier(12, 10, 5, kind=kind)
                log_probs = model(sequence)
                self.assertEqual(log_probs.shape, (3, 5))
                self.assertTrue(
                    torch.allclose(
                        log_probs.exp().sum(dim=-1),
                        torch.ones(3),
                        atol=1e-6,
                    )
                )

    def test_bidirectional_classifier_uses_both_directions(self) -> None:
        model = NameClassifier(
            input_size=9,
            hidden_size=7,
            output_size=4,
            kind="lstm",
            bidirectional=True,
        )
        self.assertEqual(model.classifier.in_features, 14)
        self.assertEqual(model(torch.randn(2, 6, 9)).shape, (2, 4))

    def test_invalid_kind(self) -> None:
        with self.assertRaises(ValueError):
            NameClassifier(4, 5, 3, kind="transformer")


if __name__ == "__main__":
    unittest.main()
