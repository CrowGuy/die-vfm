"""Base model abstractions for die_vfm."""

from __future__ import annotations

import abc

import torch
from torch import nn

from die_vfm.models.outputs import ModelOutput


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """Abstract interface for all top-level models in die_vfm.

    A top-level model takes an image batch tensor as input and returns a
    standardized ModelOutput. Concrete implementations should expose a stable
    embedding dimension and keep the forward contract consistent across model
    variants.
    """

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Returns the output embedding dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, image: torch.Tensor) -> ModelOutput:
        """Runs model forward pass on an image batch.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A standardized ModelOutput containing at least the embedding field.
        """
        raise NotImplementedError