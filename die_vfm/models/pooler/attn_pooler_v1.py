"""Attention-based pooler for die_vfm."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from die_vfm.models.outputs import PoolerOutput
from die_vfm.models.pooler.base import Pooler


class AttnPoolerV1(Pooler):
    """Pools token features with single-query attention.

    Input:
        - patch_tokens: Tensor[B, N, D]
        - cls_token: Optional Tensor[B, D]
        - token_mask: Optional Tensor[B, N]

    Output:
        - embedding: Tensor[B, D_out]
        - token_weights: Tensor[B, N]

    Attention formulation:
        h_i = tanh(W_k x_i + b_k)
        score_i = w_q^T h_i                      (default)
        or
        score_i = <W_q q, h_i>                  (if use_cls_token_as_query=True)

        alpha = softmax(score, dim=1)
        embedding = sum_i alpha_i * x_i
        optional proj -> output_dim
        optional l2_norm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        l2_norm: bool = False,
        use_cls_token_as_query: bool = False,
        return_token_weights: bool = True,
    ) -> None:
        """Initializes the AttnPoolerV1.

        Args:
            input_dim: Input token feature dimension D.
            hidden_dim: Internal attention hidden dimension.
            output_dim: Optional output embedding dimension. If None, keeps input_dim.
            dropout: Dropout probability applied to attention weights.
            l2_norm: Whether to apply L2 normalization to the final embedding.
            use_cls_token_as_query: Whether to derive the attention query from cls_token.
            return_token_weights: Whether to return token attention weights.

        Raises:
            ValueError: If any constructor argument is invalid.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if output_dim is not None and output_dim <= 0:
            raise ValueError(f"output_dim must be > 0 when provided, got {output_dim}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._output_dim = output_dim if output_dim is not None else input_dim
        self.dropout = dropout
        self.l2_norm = l2_norm
        self.use_cls_token_as_query = use_cls_token_as_query
        self.return_token_weights = return_token_weights

        self.key_proj = nn.Linear(input_dim, hidden_dim)

        if use_cls_token_as_query:
            self.query_proj = nn.Linear(input_dim, hidden_dim)
            self.score_proj = None
        else:
            self.query_proj = None
            self.score_proj = nn.Linear(hidden_dim, 1)

        self.output_proj = (
            nn.Identity()
            if self._output_dim == input_dim
            else nn.Linear(input_dim, self._output_dim)
        )

    @property
    def output_dim(self) -> int:
        """Returns the final embedding dimension."""
        return self._output_dim

    def forward(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> PoolerOutput:
        """Pools token representations into an attention-weighted embedding.

        Args:
            patch_tokens: Patch/token features with shape [B, N, D].
            cls_token: Optional CLS/global token with shape [B, D].
            token_mask: Optional valid-token mask with shape [B, N].
                Non-zero entries are treated as valid tokens.

        Returns:
            A PoolerOutput containing:
                - embedding: Tensor[B, D_out]
                - token_weights: Tensor[B, N] or None

        Raises:
            ValueError: If input feature dimensions are invalid, if cls_token is
                required but missing, or if token_mask masks out all tokens for
                any sample.
        """
        self.validate_inputs(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            token_mask=token_mask,
        )
        self._validate_feature_dim(patch_tokens)

        if self.use_cls_token_as_query and cls_token is None:
            raise ValueError(
                "cls_token must be provided when use_cls_token_as_query=True."
            )

        attn_scores = self._compute_attention_scores(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
        )
        attn_weights = self._masked_softmax(
            scores=attn_scores,
            token_mask=token_mask,
        )

        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
            attn_weights = self._renormalize_after_dropout(
                attn_weights=attn_weights,
                token_mask=token_mask,
            )

        embedding = torch.sum(
            patch_tokens * attn_weights.unsqueeze(-1),
            dim=1,
        )
        embedding = self.output_proj(embedding)

        if self.l2_norm:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return PoolerOutput(
            embedding=embedding,
            token_weights=attn_weights if self.return_token_weights else None,
            metadata={
                "pooler_name": "attn_pooler_v1",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "dropout": self.dropout,
                "l2_norm": self.l2_norm,
                "use_cls_token_as_query": self.use_cls_token_as_query,
                "return_token_weights": self.return_token_weights,
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

    def _compute_attention_scores(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None,
    ) -> torch.Tensor:
        """Computes unnormalized attention scores.

        Args:
            patch_tokens: Tensor[B, N, D]
            cls_token: Optional Tensor[B, D]

        Returns:
            Scores with shape [B, N].
        """
        hidden = torch.tanh(self.key_proj(patch_tokens))  # [B, N, H]

        if self.use_cls_token_as_query:
            assert cls_token is not None
            query = torch.tanh(self.query_proj(cls_token))  # [B, H]
            scores = torch.sum(hidden * query.unsqueeze(1), dim=-1)  # [B, N]
            return scores

        assert self.score_proj is not None
        scores = self.score_proj(hidden).squeeze(-1)  # [B, N]
        return scores

    def _masked_softmax(
        self,
        scores: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes softmax over valid tokens only.

        Args:
            scores: Attention scores with shape [B, N].
            token_mask: Optional valid-token mask with shape [B, N].

        Returns:
            Attention weights with shape [B, N].

        Raises:
            ValueError: If token_mask produces zero valid tokens for any sample.
        """
        if token_mask is None:
            return torch.softmax(scores, dim=1)

        valid = token_mask != 0
        valid_counts = valid.sum(dim=1)

        if torch.any(valid_counts <= 0):
            raise ValueError(
                "token_mask must include at least one valid token per sample."
            )

        masked_scores = scores.masked_fill(~valid, float("-inf"))
        return torch.softmax(masked_scores, dim=1)

    def _renormalize_after_dropout(
        self,
        attn_weights: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Renormalizes attention weights after dropout.

        This keeps weights summing to 1 per sample whenever at least one token
        weight survives dropout. If all weights are dropped for a sample, it falls
        back to the pre-renormalized masked support by assigning uniform weights
        over valid tokens.

        Args:
            attn_weights: Tensor[B, N]
            token_mask: Optional Tensor[B, N]

        Returns:
            Renormalized weights with shape [B, N].
        """
        denom = attn_weights.sum(dim=1, keepdim=True)
        nonzero = denom > 0

        renorm = torch.zeros_like(attn_weights)
        renorm = torch.where(nonzero, attn_weights / denom.clamp_min(1e-12), renorm)

        if torch.all(nonzero):
            return renorm

        if token_mask is None:
            fallback = torch.full_like(attn_weights, 1.0 / attn_weights.shape[1])
        else:
            valid = (token_mask != 0).to(dtype=attn_weights.dtype)
            valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            fallback = valid / valid_counts

        renorm = torch.where(nonzero, renorm, fallback)
        return renorm