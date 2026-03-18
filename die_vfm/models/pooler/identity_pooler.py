"""Identity pooler for die_vfm."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from die_vfm.models.outputs import PoolerOutput
from die_vfm.models.pooler.base import Pooler


class IdentityPooler(Pooler):
    """Selects a single token representation as the final embedding.

    Pooling behavior:
        1. If cls_token is provided, use cls_token as the embedding.
        2. Otherwise, use the first patch token patch_tokens[:, 0, :].

    Input:
        - patch_tokens: Tensor[B, N, D]
        - cls_token: Optional Tensor[B, D]
        - token_mask: Optional Tensor[B, N] (unused)

    Output:
        - embedding: Tensor[B, D]
        - token_weights: None
    """

    def __init__(self, input_dim: int, l2_norm: bool = False) -> None:
        """Initializes the IdentityPooler.

        Args:
            input_dim: Input token feature dimension D.
            l2_norm: Whether to apply L2 normalization to the output embedding.

        Raises:
            ValueError: If input_dim is invalid.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}.")

        self.input_dim = input_dim
        self.l2_norm = l2_norm

    @property
    def output_dim(self) -> int:
        """Returns the final embedding dimension."""
        return self.input_dim

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> PoolerOutput:
        """Selects the identity embedding from input token representations.

        Args:
            patch_tokens: Patch/token features with shape [B, N, D].
            cls_token: Optional CLS/global token with shape [B, D].
            token_mask: Optional valid-token mask with shape [B, N]. Unused.

        Returns:
            A PoolerOutput containing:
                - embedding: Tensor[B, D]
                - token_weights: None

        Raises:
            ValueError: If input feature dimensions do not match input_dim.
        """
        self.validate_inputs(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            token_mask=token_mask,
        )
        self._validate_feature_dim(patch_tokens, cls_token)

        del token_mask  # Unused by identity pooling.

        if cls_token is not None:
            embedding = cls_token
            source = "cls_token"
        else:
            embedding = patch_tokens[:, 0, :]
            source = "first_patch_token"

        if self.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return PoolerOutput(
            embedding=embedding,
            token_weights=None,
            metadata={
                "pooler_name": "identity",
                "input_dim": self.input_dim,
                "l2_norm": self.l2_norm,
                "source": source,
            },
        )

    def _validate_feature_dim(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None,
    ) -> None:
        """Validates token feature dimensions."""
        patch_feature_dim = patch_tokens.shape[-1]
        if patch_feature_dim != self.input_dim:
            raise ValueError(
                "patch_tokens feature dimension mismatch: "
                f"expected {self.input_dim}, got {patch_feature_dim}."
            )

        if cls_token is not None and cls_token.shape[-1] != self.input_dim:
            raise ValueError(
                "cls_token feature dimension mismatch: "
                f"expected {self.input_dim}, got {cls_token.shape[-1]}."
            )