"""Inspectable scaled dot-product attention and Transformer blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return context and weights for 3-D or 4-D attention tensors."""
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(queries.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("mask must be boolean, where True means allowed")
        expanded_mask = mask
        if expanded_mask.ndim == 2:
            while expanded_mask.ndim < scores.ndim:
                expanded_mask = expanded_mask.unsqueeze(0)
        else:
            while expanded_mask.ndim < scores.ndim:
                expanded_mask = expanded_mask.unsqueeze(1)
        scores = scores.masked_fill(
            ~expanded_mask,
            torch.finfo(scores.dtype).min,
        )
    weights = torch.softmax(scores, dim=-1)
    return weights @ values, weights


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head attention with an explicit boolean mask."""

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_size <= 0 or num_heads <= 0 or embed_size % num_heads:
            raise ValueError("embed_size must be divisible by positive num_heads")
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.query = nn.Linear(embed_size, embed_size, bias=bias)
        self.key = nn.Linear(embed_size, embed_size, bias=bias)
        self.value = nn.Linear(embed_size, embed_size, bias=bias)
        self.output = nn.Linear(embed_size, embed_size, bias=bias)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = inputs.shape
        return inputs.reshape(
            batch_size,
            steps,
            self.num_heads,
            self.head_size,
        ).transpose(1, 2)

    def _merge_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, steps, _ = inputs.shape
        return inputs.transpose(1, 2).reshape(
            batch_size,
            steps,
            self.embed_size,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for tensor in (queries, keys, values):
            if tensor.ndim != 3 or tensor.shape[-1] != self.embed_size:
                raise ValueError("attention inputs must have shape [batch, steps, embed]")
        query_heads = self._split_heads(self.query(queries))
        key_heads = self._split_heads(self.key(keys))
        value_heads = self._split_heads(self.value(values))
        context, weights = scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            mask=mask,
        )
        output = self.output(self._merge_heads(context))
        return (output, weights) if return_weights else output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first sequences."""

    def __init__(self, embed_size: int, max_length: int = 512) -> None:
        super().__init__()
        if embed_size <= 0 or max_length <= 0:
            raise ValueError("embed_size and max_length must be positive")
        positions = torch.arange(max_length).reshape(-1, 1)
        frequencies = torch.exp(
            torch.arange(0, embed_size, 2)
            * (-math.log(10_000.0) / embed_size)
        )
        encoding = torch.zeros(max_length, embed_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.encoding.shape[-1]:
            raise ValueError("inputs must have shape [batch, steps, embed]")
        if inputs.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence is longer than max_length")
        return inputs + self.encoding[:, : inputs.shape[1]]


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Return a lower-triangular boolean self-attention mask."""
    if length <= 0:
        raise ValueError("length must be positive")
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


class TransformerEncoderBlock(nn.Module):
    """Pre-norm-free Transformer encoder block with residual connections."""

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        ffn_hidden_size: int,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_size),
            nn.GELU(),
            nn.Linear(ffn_hidden_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attention(inputs, inputs, inputs, mask=mask)
        inputs = self.norm1(inputs + attended)
        return self.norm2(inputs + self.ffn(inputs))


class TransformerEncoder(nn.Module):
    """Stack Transformer encoder blocks over batch-first embeddings."""

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList(
            TransformerEncoderBlock(embed_size, num_heads, ffn_hidden_size)
            for _ in range(num_layers)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output, mask=mask)
        return output


class CausalTransformerLM(nn.Module):
    """Small decoder-only Transformer language model.

    The causal mask makes each position attend only to itself and earlier
    positions, so the module returns next-token logits without tuple ambiguity.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        num_layers: int = 1,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        if min(vocab_size, embed_size, ffn_hidden_size, max_length) <= 0:
            raise ValueError("model dimensions must be positive")
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            TransformerEncoderBlock(embed_size, num_heads, ffn_hidden_size)
            for _ in range(num_layers)
        )
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, steps]")
        hidden_states = self.embedding(tokens.long()) * math.sqrt(self.embed_size)
        hidden_states = self.position(hidden_states)
        mask = causal_mask(tokens.shape[1], device=tokens.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
        return self.output(hidden_states)
