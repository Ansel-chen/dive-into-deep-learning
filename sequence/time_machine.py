"""Small sequence iterators inspired by the Time Machine language-model example."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch


def sequential_batches(
    corpus: Sequence[int],
    batch_size: int,
    num_steps: int,
    *,
    offset: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield consecutive feature/target batches with targets shifted by one token."""
    usable = ((len(corpus) - offset - 1) // batch_size) * batch_size
    features = torch.tensor(corpus[offset : offset + usable])
    targets = torch.tensor(corpus[offset + 1 : offset + 1 + usable])
    features = features.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)
    batch_count = features.shape[1] // num_steps
    for start in range(0, batch_count * num_steps, num_steps):
        yield features[:, start : start + num_steps], targets[:, start : start + num_steps]

