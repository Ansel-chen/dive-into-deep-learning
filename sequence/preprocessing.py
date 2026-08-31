"""Small, local text-preprocessing utilities."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence

import torch


def normalize_text(text: str) -> str:
    """Lowercase text and collapse non-letter spans into spaces."""
    return re.sub(r"[^a-zA-Z]+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into whitespace-delimited words."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


class Vocabulary:
    """Deterministic token-to-index mapping with explicit special tokens."""

    def __init__(
        self,
        tokens: Iterable[str],
        min_freq: int = 1,
        specials: Sequence[str] = ("<unk>", "<pad>", "<bos>", "<eos>"),
    ) -> None:
        if min_freq <= 0:
            raise ValueError("min_freq must be positive")
        self.idx_to_token: list[str] = []
        for token in specials:
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)

        counts = Counter(tokens)
        learned_tokens = sorted(
            (
                (token, frequency)
                for token, frequency in counts.items()
                if frequency >= min_freq and token not in self.idx_to_token
            ),
            key=lambda item: (-item[1], item[0]),
        )
        self.idx_to_token.extend(token for token, _ in learned_tokens)
        self.token_to_idx = {
            token: index for index, token in enumerate(self.idx_to_token)
        }

    @property
    def unk_index(self) -> int:
        return self.token_to_idx["<unk>"]

    @property
    def pad_index(self) -> int:
        return self.token_to_idx["<pad>"]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_idx

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.unk_index)

    def to_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]

    def to_tokens(self, indices: Iterable[int]) -> list[str]:
        return [
            self.idx_to_token[int(index)] if 0 <= int(index) < len(self) else "<unk>"
            for index in indices
        ]


def batchify(
    token_ids: Sequence[int] | torch.Tensor,
    batch_size: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one contiguous next-token training batch."""
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size and num_steps must be positive")
    tokens = torch.as_tensor(token_ids, dtype=torch.long).flatten()
    required = batch_size * (num_steps + 1)
    if tokens.numel() < required:
        raise ValueError("token_ids is too short for the requested batch")
    rows = tokens[:required].reshape(batch_size, num_steps + 1)
    return rows[:, :-1], rows[:, 1:]


def iter_sequential_batches(
    token_ids: Sequence[int] | torch.Tensor,
    batch_size: int,
    num_steps: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield non-overlapping next-token batches from a token stream."""
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size and num_steps must be positive")
    tokens = torch.as_tensor(token_ids, dtype=torch.long).flatten()
    row_length = tokens.numel() // batch_size
    if row_length <= num_steps:
        return
    rows = tokens[: batch_size * row_length].reshape(batch_size, row_length)
    for start in range(0, row_length - num_steps, num_steps):
        yield rows[:, start : start + num_steps], rows[
            :, start + 1 : start + num_steps + 1
        ]

