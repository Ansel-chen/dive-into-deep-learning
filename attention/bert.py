"""Small BERT-style embedding and encoder components."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from attention.transformer import TransformerEncoder


class BertEmbeddings(nn.Module):
    """Token, segment, and position embeddings with layer normalization."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size, max_position_embeddings, type_vocab_size) <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.max_position_embeddings = max_position_embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, steps]")
        batch_size, steps = input_ids.shape
        if steps > self.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                steps,
                device=input_ids.device,
            ).expand(batch_size, -1)
        output = (
            self.token_embeddings(input_ids.long())
            + self.type_embeddings(token_type_ids.long())
            + self.position_embeddings(position_ids.long())
        )
        return self.layer_norm(output)


class BertEncoder(nn.Module):
    """Transformer encoder operating on already-embedded token sequences."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int | None = None,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            embed_size=hidden_size,
            num_heads=num_heads,
            ffn_hidden_size=ffn_hidden_size or hidden_size * 4,
            num_layers=num_layers,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(hidden_states, mask=attention_mask)


class TinyBertModel(nn.Module):
    """A BERT-shaped encoder without a tokenizer or pretrained weights."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
    ) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )
        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, token_type_ids)
        if attention_mask is not None and attention_mask.ndim == 2:
            valid = attention_mask.bool()
            attention_mask = valid[:, None, :] & valid[:, :, None]
        return self.encoder(hidden_states, attention_mask=attention_mask)


class MaskedLanguageModelHead(nn.Module):
    """Predict token identities at selected masked positions."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pred_positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or pred_positions.ndim != 2:
            raise ValueError("expected hidden states [batch, steps, hidden] and positions [batch, masked]")
        batch_size = hidden_states.shape[0]
        batch_indices = torch.arange(
            batch_size,
            device=hidden_states.device,
        ).unsqueeze(1)
        selected = hidden_states[batch_indices, pred_positions.long()]
        return self.projection(selected)


class NextSentencePredictionHead(nn.Module):
    """Classify whether the second segment follows the first segment."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_state: torch.Tensor) -> torch.Tensor:
        if pooled_state.ndim != 2:
            raise ValueError("pooled_state must have shape [batch, hidden]")
        return self.classifier(pooled_state)


class TinyBertForPretraining(nn.Module):
    """Tiny BERT encoder with masked-LM and next-sentence heads."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_position_embeddings: int,
        type_vocab_size: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = TinyBertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
        )
        self.mlm = MaskedLanguageModelHead(hidden_size, vocab_size)
        self.nsp = NextSentencePredictionHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pred_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        hidden_states = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        mlm_logits = (
            None
            if pred_positions is None
            else self.mlm(hidden_states, pred_positions)
        )
        nsp_logits = self.nsp(hidden_states[:, 0])
        return hidden_states, mlm_logits, nsp_logits


def bert_pretraining_loss(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    nsp_logits: torch.Tensor,
    nsp_labels: torch.Tensor,
    mlm_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine masked-language-model and next-sentence losses."""
    if mlm_logits.ndim != 3 or mlm_labels.ndim != 2:
        raise ValueError("MLM logits and labels must be [batch, masked, vocab] and [batch, masked]")
    mlm_loss = F.cross_entropy(
        mlm_logits.reshape(-1, mlm_logits.shape[-1]),
        mlm_labels.long().reshape(-1),
        reduction="none",
    ).reshape_as(mlm_labels)
    if mlm_weights is None:
        mlm_weights = torch.ones_like(mlm_loss)
    if mlm_weights.shape != mlm_loss.shape:
        raise ValueError("mlm_weights must match mlm_labels")
    normalizer = mlm_weights.sum().clamp_min(1.0)
    mlm_loss = (mlm_loss * mlm_weights).sum() / normalizer
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels.long())
    return mlm_loss + nsp_loss
