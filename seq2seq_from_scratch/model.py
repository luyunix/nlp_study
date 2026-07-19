"""Readable GRU encoder/attention-decoder implementation.

All public sequence tensors use ``batch_first=True``:

* source token ids: ``[B, S]``
* target token ids: ``[B, T]``
* encoder outputs: ``[B, S, H]``
* decoder logits: ``[B, T-1, V_target]``
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn


class EncoderGRU(nn.Module):
    """Embed a source sentence and return every GRU state."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_size,
            padding_idx=padding_id,
        )
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            batch_first=True,
        )

    def forward(self, source_ids: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(source_ids)
        return self.gru(embedded)


class AttentionDecoderGRU(nn.Module):
    """Decode one token with dot-product attention over encoder outputs."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        *,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_size,
            padding_idx=padding_id,
        )
        self.gru = nn.GRU(
            embedding_size + hidden_size,
            hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocabulary_size)

    def forward_step(
        self,
        input_ids: Tensor,
        hidden: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one decoder time step.

        ``input_ids`` is ``[B]`` and ``hidden`` is ``[1,B,H]``.
        """

        query = hidden[-1]
        scores = torch.bmm(
            encoder_outputs,
            query.unsqueeze(-1),
        ).squeeze(-1)
        if source_mask is not None:
            scores = scores.masked_fill(
                ~source_mask,
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs,
        )
        embedded = self.embedding(input_ids).unsqueeze(1)
        recurrent_input = torch.cat((embedded, context), dim=-1)
        output, hidden = self.gru(recurrent_input, hidden)
        logits = self.classifier(output.squeeze(1))
        return logits, hidden, weights


class Seq2Seq(nn.Module):
    """Join an encoder and attention decoder for training or greedy use."""

    def __init__(
        self,
        encoder: EncoderGRU,
        decoder: AttentionDecoderGRU,
        *,
        start_id: int,
        end_id: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.start_id = start_id
        self.end_id = end_id

    def forward(
        self,
        source_ids: Tensor,
        target_ids: Tensor,
        *,
        teacher_forcing_ratio: float = 1.0,
        source_mask: Tensor | None = None,
    ) -> Tensor:
        """Return training logits for target positions 1..T-1."""

        encoder_outputs, hidden = self.encoder(source_ids)
        decoder_input = target_ids[:, 0]
        logits_by_step = []
        for step in range(1, target_ids.size(1)):
            logits, hidden, _weights = self.decoder.forward_step(
                decoder_input,
                hidden,
                encoder_outputs,
                source_mask,
            )
            logits_by_step.append(logits)
            use_teacher = random.random() < teacher_forcing_ratio
            decoder_input = (
                target_ids[:, step]
                if use_teacher
                else logits.argmax(dim=-1)
            )
        return torch.stack(logits_by_step, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: Tensor,
        *,
        maximum_length: int,
        source_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Greedily decode ids and return per-step attention weights."""

        encoder_outputs, hidden = self.encoder(source_ids)
        batch_size = source_ids.size(0)
        current = torch.full(
            (batch_size,),
            self.start_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        ids = []
        attention = []
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=source_ids.device,
        )
        for _ in range(maximum_length):
            logits, hidden, weights = self.decoder.forward_step(
                current,
                hidden,
                encoder_outputs,
                source_mask,
            )
            current = logits.argmax(dim=-1)
            ids.append(current)
            attention.append(weights)
            finished |= current.eq(self.end_id)
            if bool(finished.all()):
                break
        return torch.stack(ids, dim=1), torch.stack(attention, dim=1)
