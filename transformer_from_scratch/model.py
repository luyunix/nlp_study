"""Transformer building blocks, implemented incrementally from lessons 111–143."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import Tensor, nn


class Embeddings(nn.Module):
    """Map token ids to vectors and scale them by ``sqrt(d_model)``."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.lut(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position information without changing the shape."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        # The slice keeps this valid even when d_model is odd.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def subsequent_mask(size: int) -> Tensor:
    """Return ``[1, size, size]`` where True means attention is allowed."""

    future = torch.triu(
        torch.ones(1, size, size, dtype=torch.bool),
        diagonal=1,
    )
    return ~future


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention."""

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Create ``n`` independent deep copies of a module."""

    return nn.ModuleList(deepcopy(module) for _ in range(n))


class MultiHeadedAttention(nn.Module):
    """Project, split, attend in parallel, concatenate, and project again."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by the number of heads")

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn: Tensor | None = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            linear(x)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, x in zip(self.linears[:3], (query, key, value))
        ]

        x, self.attn = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Apply the same two-layer MLP independently at every sequence position."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    """Normalize the last dimension and learn scale and bias."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (std + self.eps) + self.bias


class SublayerConnection(nn.Module):
    """Pre-LN residual connection: ``x + dropout(sublayer(norm(x)))``."""

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """One encoder block: self-attention followed by position-wise FFN."""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        x = self.sublayers[0](
            x,
            lambda x: self.self_attn(x, x, x, mask),
        )
        return self.sublayers[1](x, self.feed_forward)


class Encoder(nn.Module):
    """A stack of encoder layers followed by a final normalization."""

    def __init__(self, layer: EncoderLayer, n: int) -> None:
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """One decoder block with masked self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        x = self.sublayers[0](
            x,
            lambda x: self.self_attn(x, x, x, tgt_mask),
        )
        x = self.sublayers[1](
            x,
            lambda x: self.src_attn(x, memory, memory, src_mask),
        )
        return self.sublayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    """A stack of decoder layers followed by a final normalization."""

    def __init__(self, layer: DecoderLayer, n: int) -> None:
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """Project decoder states to log-probabilities over the target vocabulary."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return torch.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """Full Transformer with source/target embeddings and output generator."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: Tensor, src_mask: Tensor | None) -> Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
    ) -> Tensor:
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(decoded)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Build a complete Transformer and initialize matrix parameters."""

    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                d_model,
                deepcopy(attn),
                deepcopy(ff),
                dropout,
            ),
            n,
        ),
        Decoder(
            DecoderLayer(
                d_model,
                deepcopy(attn),
                deepcopy(attn),
                deepcopy(ff),
                dropout,
            ),
            n,
        ),
        nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model, dropout),
        ),
        nn.Sequential(
            Embeddings(tgt_vocab, d_model),
            PositionalEncoding(d_model, dropout),
        ),
        Generator(d_model, tgt_vocab),
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model
