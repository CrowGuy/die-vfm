"""DINOv2 backbone wrapper for die_vfm."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from die_vfm.models.backbone.base import Backbone
from die_vfm.models.outputs import BackboneOutput


_DINOV2_MODEL_NAME_MAP = {
    "vit_small": "dinov2_vits14",
    "vit_base": "dinov2_vitb14",
    "vit_large": "dinov2_vitl14",
    "vit_giant": "dinov2_vitg14",
}

_DINOV2_OUTPUT_DIM_MAP = {
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_giant": 1536,
}


class DINOv2Backbone(Backbone):
    """DINOv2 visual backbone wrapper.

    This wrapper standardizes DINOv2 outputs into the die_vfm BackboneOutput
    contract:

        - patch_tokens: Tensor[B, N, D]
        - cls_token: Tensor[B, D] or None
        - token_mask: None

    Notes:
        - This implementation assumes square image inputs.
        - DINOv2 uses patch size 14 for the official ViT checkpoints.
        - For PR-3, token_mask is always None because DINOv2 does not emit a
          variable-length token mask in the standard image path.
    """

    def __init__(
        self,
        variant: str,
        pretrained: bool = True,
        freeze: bool = False,
        return_cls_token: bool = True,
    ) -> None:
        """Initializes the DINOv2 backbone.

        Args:
            variant: DINOv2 variant name. One of:
                "vit_small", "vit_base", "vit_large", "vit_giant".
            pretrained: Whether to load pretrained weights.
            freeze: Whether to freeze all backbone parameters.
            return_cls_token: Whether to expose CLS token in BackboneOutput.

        Raises:
            ValueError: If the variant is unsupported.
        """
        super().__init__()
        _validate_variant(variant)

        self.variant = variant
        self.pretrained = pretrained
        self.freeze = freeze
        self.return_cls_token = return_cls_token

        self._output_dim = _DINOV2_OUTPUT_DIM_MAP[variant]
        self._patch_size = 14

        model_name = _DINOV2_MODEL_NAME_MAP[variant]
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )

        if freeze:
            self._freeze_parameters()

    @property
    def output_dim(self) -> int:
        """Returns the token feature dimension D."""
        return self._output_dim

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        """Encodes an image batch into DINOv2 token representations.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A standardized BackboneOutput.

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If image shape is invalid for DINOv2 patchification.
            RuntimeError: If the underlying DINOv2 output format is unsupported.
        """
        self.validate_image_input(image)
        self._validate_spatial_shape(image)

        features = self.model.forward_features(image)
        patch_tokens, cls_token = self._parse_forward_features(features)

        return BackboneOutput(
            patch_tokens=patch_tokens,
            cls_token=cls_token if self.return_cls_token else None,
            token_mask=None,
            feature_dim=self.output_dim,
            patch_grid=self._infer_patch_grid(image),
            metadata={
                "backbone_name": "dinov2",
                "variant": self.variant,
                "pretrained": self.pretrained,
                "freeze": self.freeze,
                "return_cls_token": self.return_cls_token,
                "patch_size": self._patch_size,
            },
        )

    def _freeze_parameters(self) -> None:
        """Freezes all backbone parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _validate_spatial_shape(self, image: torch.Tensor) -> None:
        """Validates image spatial dimensions for DINOv2."""
        _, _, height, width = image.shape

        if height != width:
            raise ValueError(
                "DINOv2Backbone expects square inputs for PR-3: "
                f"got height={height}, width={width}."
            )

        if height % self._patch_size != 0 or width % self._patch_size != 0:
            raise ValueError(
                "Input spatial size must be divisible by DINOv2 patch size "
                f"{self._patch_size}: got [{height}, {width}]."
            )

    def _infer_patch_grid(self, image: torch.Tensor) -> tuple[int, int]:
        """Infers patch grid from the input image shape."""
        _, _, height, width = image.shape
        return height // self._patch_size, width // self._patch_size

    def _parse_forward_features(
        self,
        features: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Parses DINOv2 forward_features output.

        The official DINOv2 implementation returns a mapping-like object with
        keys such as:
            - "x_norm_clstoken"
            - "x_norm_patchtokens"

        Returns:
            A tuple of:
                - patch_tokens: Tensor[B, N, D]
                - cls_token: Tensor[B, D] or None

        Raises:
            RuntimeError: If the output format is unsupported.
        """
        if isinstance(features, Mapping):
            patch_tokens = features.get("x_norm_patchtokens")
            cls_token = features.get("x_norm_clstoken")

            if patch_tokens is None:
                raise RuntimeError(
                    "DINOv2 forward_features output missing "
                    "'x_norm_patchtokens'."
                )

            if not isinstance(patch_tokens, torch.Tensor):
                raise RuntimeError(
                    "Expected 'x_norm_patchtokens' to be a torch.Tensor."
                )

            if cls_token is not None and not isinstance(cls_token, torch.Tensor):
                raise RuntimeError(
                    "Expected 'x_norm_clstoken' to be a torch.Tensor or None."
                )

            return patch_tokens, cls_token

        raise RuntimeError(
            "Unsupported DINOv2 forward_features output type: "
            f"{type(features).__name__}."
        )


def _validate_variant(variant: str) -> None:
    """Validates DINOv2 variant name."""
    if variant not in _DINOV2_MODEL_NAME_MAP:
        supported = ", ".join(sorted(_DINOV2_MODEL_NAME_MAP))
        raise ValueError(
            f"Unsupported DINOv2 variant: '{variant}'. "
            f"Supported variants: {supported}."
        )