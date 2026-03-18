"""Backbone abstractions for die_vfm."""

from __future__ import annotations

import abc

import torch
from torch import nn

from die_vfm.models.outputs import BackboneOutput


class Backbone(nn.Module, metaclass=abc.ABCMeta):
    """Abstract interface for all visual backbones in die_vfm.

    A backbone encodes an image batch tensor into token-level visual features.
    The standardized output contract is BackboneOutput.
    """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Returns the token feature dimension D."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, image: torch.Tensor) -> BackboneOutput:
        """Encodes an image batch into token representations.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A BackboneOutput containing at least:
                - patch_tokens: Tensor[B, N, D]
                - cls_token: Tensor[B, D] or None
                - token_mask: Tensor[B, N] or None
        """
        raise NotImplementedError

    def validate_image_input(self, image: torch.Tensor) -> None:
        """Validates the common backbone image input contract.

        Args:
            image: Image tensor expected to have shape [B, C, H, W].

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If image does not have rank 4.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Expected image to be torch.Tensor, got {type(image).__name__}."
            )

        if image.ndim != 4:
            raise ValueError(
                f"Expected image shape [B, C, H, W], got ndim={image.ndim}."
            )