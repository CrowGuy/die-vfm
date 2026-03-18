"""Standard output dataclasses for die_vfm model components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BackboneOutput:
    """Standard output contract for visual backbones.

    Attributes:
        patch_tokens: Patch/token features with shape [B, N, D].
        cls_token: Optional CLS/global token with shape [B, D].
        token_mask: Optional valid-token mask with shape [B, N].
        feature_dim: Token feature dimension D.
        patch_grid: Optional patch grid as (grid_h, grid_w).
        metadata: Additional implementation-specific metadata.
    """

    patch_tokens: torch.Tensor
    cls_token: torch.Tensor | None
    token_mask: torch.Tensor | None
    feature_dim: int
    patch_grid: tuple[int, int] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolerOutput:
    """Standard output contract for token pooling modules.

    Attributes:
        embedding: Final pooled embedding with shape [B, D_out].
        token_weights: Optional token-level pooling weights with shape [B, N].
        metadata: Additional implementation-specific metadata.
    """

    embedding: torch.Tensor
    token_weights: torch.Tensor | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Top-level output contract for DieVFM models.

    Attributes:
        embedding: Final model embedding with shape [B, D_out].
        backbone: Optional raw backbone output for debugging or inspection.
        pooler: Optional raw pooler output for debugging or inspection.
        metadata: Additional model-level metadata.
    """

    embedding: torch.Tensor
    backbone: BackboneOutput | None = None
    pooler: PoolerOutput | None = None
    metadata: dict[str, Any] = field(default_factory=dict)