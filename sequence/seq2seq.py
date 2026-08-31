"""Small GRU encoder-decoder components for sequence-to-sequence learning."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Token embedding and GRU encoder using batch-first inputs."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_tokens.ndim != 2:
            raise ValueError("source_tokens must have shape [batch, steps]")
        return self.rnn(self.embedding(source_tokens.long()))


class Decoder(nn.Module):
    """GRU decoder that concatenates token embeddings with context vectors."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        target_tokens: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_tokens.ndim != 2:
            raise ValueError("target_tokens must have shape [batch, steps]")
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, steps, hidden]")
        if context.shape[0] != target_tokens.shape[0]:
            raise ValueError("context and target batch sizes must match")
        if context.shape[1] == 1 and target_tokens.shape[1] > 1:
            context = context.expand(-1, target_tokens.shape[1], -1)
        if context.shape[1] != target_tokens.shape[1]:
            raise ValueError("context and target step counts must match")
        embeddings = self.embedding(target_tokens.long())
        decoder_input = torch.cat((embeddings, context), dim=-1)
        hidden_states, state = self.rnn(decoder_input, state)
        return self.projection(hidden_states), state


class Seq2Seq(nn.Module):
    """Encoder-decoder with optional teacher forcing."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        if encoder.rnn.hidden_size != decoder.rnn.hidden_size:
            raise ValueError("encoder and decoder hidden sizes must match")
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between zero and one")
        if target_tokens.shape[1] == 0:
            raise ValueError("target_tokens must contain at least one step")

        encoder_outputs, state = self.encoder(source_tokens)
        context = encoder_outputs.mean(dim=1, keepdim=True)
        if teacher_forcing_ratio == 1.0:
            logits, _ = self.decoder(target_tokens, state, context)
            return logits

        outputs: list[torch.Tensor] = []
        current = target_tokens[:, :1]
        for step in range(target_tokens.shape[1]):
            step_context = context
            logits, state = self.decoder(current, state, step_context)
            outputs.append(logits)
            if step + 1 == target_tokens.shape[1]:
                break
            if teacher_forcing_ratio == 0.0:
                current = logits.argmax(dim=-1)
            else:
                use_teacher = (
                    torch.rand((), device=target_tokens.device)
                    < teacher_forcing_ratio
                )
                current = target_tokens[:, step + 1 : step + 2] if use_teacher else logits.argmax(
                    dim=-1
                )
        return torch.cat(outputs, dim=1)

