"""Top-level DieVFM model implementation."""

from __future__ import annotations

import torch

from die_vfm.models.base import BaseModel
from die_vfm.models.backbone.base import Backbone
from die_vfm.models.outputs import ModelOutput
from die_vfm.models.pooler.base import Pooler


class DieVFMModel(BaseModel):
    """Canonical die_vfm model composed of backbone and pooler.

    The model contract for PR-3 is:

        image [B, C, H, W]
            -> backbone
            -> token features
            -> pooler
            -> embedding [B, D]

    This module is intentionally minimal. It only orchestrates the backbone and
    pooler without introducing training-specific logic.
    """

    def __init__(
        self,
        backbone: Backbone,
        pooler: Pooler,
        return_debug_outputs: bool = True,
    ) -> None:
        """Initializes the DieVFMModel.

        Args:
            backbone: Visual backbone that encodes images into token features.
            pooler: Pooling module that converts token features into embeddings.
            return_debug_outputs: Whether to attach backbone and pooler outputs
                to the returned ModelOutput for debugging and inspection.
        """
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler
        self.return_debug_outputs = return_debug_outputs

    @property
    def embedding_dim(self) -> int:
        """Returns the final embedding dimension."""
        return self.pooler.output_dim

    def forward(self, image: torch.Tensor) -> ModelOutput:
        """Runs the model forward pass.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A standardized ModelOutput containing at least:
                - embedding: Tensor[B, D]

            When return_debug_outputs is True, the output also includes:
                - backbone: BackboneOutput
                - pooler: PoolerOutput

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If image does not have shape [B, C, H, W].
        """
        self._validate_image_input(image)

        backbone_output = self.backbone(image)
        pooler_output = self.pooler(
            patch_tokens=backbone_output.patch_tokens,
            cls_token=backbone_output.cls_token,
            token_mask=backbone_output.token_mask,
        )

        return ModelOutput(
            embedding=pooler_output.embedding,
            backbone=backbone_output if self.return_debug_outputs else None,
            pooler=pooler_output if self.return_debug_outputs else None,
            metadata={
                "embedding_dim": self.embedding_dim,
            },
        )

    def _validate_image_input(self, image: torch.Tensor) -> None:
        """Validates the image input contract.

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