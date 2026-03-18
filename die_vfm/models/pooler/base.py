"""Pooler abstractions for die_vfm."""

from __future__ import annotations

import abc

import torch
from torch import nn

from die_vfm.models.outputs import PoolerOutput


class Pooler(nn.Module, metaclass=abc.ABCMeta):
    """Abstract interface for all token poolers in die_vfm.

    A pooler converts token-level features into a final embedding tensor.
    The standardized output contract is PoolerOutput.
    """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Returns the final embedding dimension."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> PoolerOutput:
        """Pools token representations into a final embedding.

        Args:
            patch_tokens: Patch/token features with shape [B, N, D].
            cls_token: Optional CLS/global token with shape [B, D].
            token_mask: Optional valid-token mask with shape [B, N].

        Returns:
            A PoolerOutput containing at least:
                - embedding: Tensor[B, D_out]
                - token_weights: Tensor[B, N] or None
        """
        raise NotImplementedError

    def validate_inputs(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> None:
        """Validates the common pooler input contract.

        Args:
            patch_tokens: Patch/token features expected to have shape [B, N, D].
            cls_token: Optional CLS/global token expected to have shape [B, D].
            token_mask: Optional valid-token mask expected to have shape [B, N].

        Raises:
            TypeError: If any input has an invalid type.
            ValueError: If any input has an invalid shape.
        """
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError(
                "Expected patch_tokens to be torch.Tensor, "
                f"got {type(patch_tokens).__name__}."
            )

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Expected patch_tokens shape [B, N, D], "
                f"got ndim={patch_tokens.ndim}."
            )

        batch_size, num_tokens, feature_dim = patch_tokens.shape

        if cls_token is not None:
            if not isinstance(cls_token, torch.Tensor):
                raise TypeError(
                    "Expected cls_token to be torch.Tensor or None, "
                    f"got {type(cls_token).__name__}."
                )
            if cls_token.ndim != 2:
                raise ValueError(
                    "Expected cls_token shape [B, D], "
                    f"got ndim={cls_token.ndim}."
                )
            if cls_token.shape[0] != batch_size:
                raise ValueError(
                    "cls_token batch size mismatch: "
                    f"expected {batch_size}, got {cls_token.shape[0]}."
                )
            if cls_token.shape[1] != feature_dim:
                raise ValueError(
                    "cls_token feature dimension mismatch: "
                    f"expected {feature_dim}, got {cls_token.shape[1]}."
                )

        if token_mask is not None:
            if not isinstance(token_mask, torch.Tensor):
                raise TypeError(
                    "Expected token_mask to be torch.Tensor or None, "
                    f"got {type(token_mask).__name__}."
                )
            if token_mask.ndim != 2:
                raise ValueError(
                    "Expected token_mask shape [B, N], "
                    f"got ndim={token_mask.ndim}."
                )
            if token_mask.shape[0] != batch_size:
                raise ValueError(
                    "token_mask batch size mismatch: "
                    f"expected {batch_size}, got {token_mask.shape[0]}."
                )
            if token_mask.shape[1] != num_tokens:
                raise ValueError(
                    "token_mask token dimension mismatch: "
                    f"expected {num_tokens}, got {token_mask.shape[1]}."
                )