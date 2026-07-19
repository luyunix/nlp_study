"""Readable single-query attention implementations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _masked_softmax(scores: Tensor, mask: Tensor | None) -> Tensor:
    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
    return torch.softmax(scores, dim=-1)


class DotProductAttention(nn.Module):
    """Attend from one query ``[B,H]`` to values ``[B,L,H]``."""

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        weights = _masked_softmax(scores, mask)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights


class GeneralAttention(nn.Module):
    """Luong general attention: project the query before matching keys."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected_query = self.query_projection(query)
        scores = torch.bmm(
            keys,
            projected_query.unsqueeze(-1),
        ).squeeze(-1)
        weights = _masked_softmax(scores, mask)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights
