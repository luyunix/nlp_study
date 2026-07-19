"""Small, explicit RNN/LSTM/GRU models for the course exercises."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleRNNCell(nn.Module):
    """One recurrent step written from the defining equation.

    ``h_t = tanh(W_x x_t + W_h h_(t-1))``
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x_t: Tensor, h_previous: Tensor) -> Tensor:
        return torch.tanh(
            self.input_to_hidden(x_t)
            + self.hidden_to_hidden(h_previous)
        )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)


class NameClassifier(nn.Module):
    """Classify a padded character sequence with RNN, LSTM, or GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        kind: str = "rnn",
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        recurrent_types = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        if kind not in recurrent_types:
            raise ValueError("kind must be 'rnn', 'lstm', or 'gru'")

        self.kind = kind
        self.recurrent = recurrent_types[kind](
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * directions, output_size)

    def forward(self, sequence: Tensor) -> Tensor:
        _output, state = self.recurrent(sequence)
        hidden = state[0] if self.kind == "lstm" else state
        if self.recurrent.bidirectional:
            final_state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            final_state = hidden[-1]
        return torch.log_softmax(self.classifier(final_state), dim=-1)
