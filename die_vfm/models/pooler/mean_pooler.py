"""Mean pooler for die_vfm."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from die_vfm.models.outputs import PoolerOutput
from die_vfm.models.pooler.base import Pooler


class MeanPooler(Pooler):
    """Pools token features by masked or unmasked mean reduction.

    Input:
        - patch_tokens: Tensor[B, N, D]
        - cls_token: Unused for mean pooling
        - token_mask: Optional Tensor[B, N]

    Output:
        - embedding: Tensor[B, D]
        - token_weights: None
    """

    def __init__(self, input_dim: int, l2_norm: bool = False) -> None:
        """Initializes the MeanPooler.

        Args:
            input_dim: Input token feature dimension D.
            l2_norm: Whether to apply L2 normalization to the pooled embedding.

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
        """Pools token representations into a mean embedding.

        Args:
            patch_tokens: Patch/token features with shape [B, N, D].
            cls_token: Optional CLS/global token with shape [B, D]. Unused.
            token_mask: Optional valid-token mask with shape [B, N]. Non-zero
                entries are treated as valid tokens.

        Returns:
            A PoolerOutput containing:
                - embedding: Tensor[B, D]
                - token_weights: None
        """
        self.validate_inputs(
        patch_tokens=patch_tokens,
            cls_token=cls_token,
            token_mask=token_mask,
        )
        self._validate_feature_dim(patch_tokens)

        del cls_token

        embedding = self._masked_mean_pool(
            patch_tokens=patch_tokens,
            token_mask=token_mask,
        )

        if self.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return PoolerOutput(
            embedding=embedding,
            token_weights=None,
            metadata={
                "pooler_name": "mean",
                "input_dim": self.input_dim,
                "l2_norm": self.l2_norm,
            },
        )

    def _validate_feature_dim(self, patch_tokens: torch.Tensor) -> None:
        """Validates token feature dimension."""
        feature_dim = patch_tokens.shape[-1]
        if feature_dim != self.input_dim:
            raise ValueError(
                "patch_tokens feature dimension mismatch: "
                f"expected {self.input_dim}, got {feature_dim}."
            )

    def _masked_mean_pool(
        self,
        patch_tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes masked or unmasked mean pooling over tokens.

        Args:
            patch_tokens: Token tensor with shape [B, N, D].
            token_mask: Optional mask tensor with shape [B, N].

        Returns:
            Mean pooled embedding with shape [B, D].

        Raises:
            ValueError: If token_mask produces zero valid tokens for any sample.
        """
        if token_mask is None:
            return patch_tokens.mean(dim=1)

        mask = token_mask.to(dtype=patch_tokens.dtype)
        valid_counts = mask.sum(dim=1, keepdim=True)

        if torch.any(valid_counts <= 0):
            raise ValueError(
                "token_mask must include at least one valid token per sample."
            )

        weighted_tokens = patch_tokens * mask.unsqueeze(-1)
        embedding = weighted_tokens.sum(dim=1) / valid_counts
        return embedding