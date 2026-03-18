"""Dummy backbone for PR-3 model pipeline smoke tests."""

from __future__ import annotations

import torch
from torch import nn

from die_vfm.models.backbone.base import Backbone
from die_vfm.models.outputs import BackboneOutput


class DummyBackbone(Backbone):
    """A minimal patch-token backbone for smoke tests.

    This backbone splits the input image into non-overlapping patches, projects
    each flattened patch to an embedding dimension, and returns the projected
    patch tokens. It is intentionally simple and deterministic so that unit
    tests can validate the model pipeline without depending on external
    pretrained models.

    Output contract:
        - patch_tokens: [B, N, D]
        - cls_token: None
        - token_mask: None
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        """Initializes the dummy backbone.

        Args:
            image_size: Expected square image size H == W.
            patch_size: Patch size for non-overlapping patchification.
            in_channels: Number of input image channels.
            embed_dim: Output token embedding dimension.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        super().__init__()
        _validate_init_args(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self._output_dim = embed_dim

        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, embed_dim)

        grid_size = image_size // patch_size
        self.patch_grid = (grid_size, grid_size)
        self.num_patches = grid_size * grid_size

    @property
    def output_dim(self) -> int:
        """Returns the token feature dimension D."""
        return self._output_dim

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        """Encodes an image batch into patch tokens.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A BackboneOutput with:
                - patch_tokens: Tensor[B, N, D]
                - cls_token: None
                - token_mask: None

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If image shape does not match the configured contract.
        """
        self.validate_image_input(image)
        self._validate_image_shape(image)

        patch_tokens = self._patchify_and_project(image)

        return BackboneOutput(
            patch_tokens=patch_tokens,
            cls_token=None,
            token_mask=None,
            feature_dim=self.output_dim,
            patch_grid=self.patch_grid,
            metadata={
                "backbone_name": "dummy",
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_patches": self.num_patches,
            },
        )

    def _validate_image_shape(self, image: torch.Tensor) -> None:
        """Validates image shape against configured backbone expectations."""
        batch_size, channels, height, width = image.shape

        del batch_size  # Unused but kept for readability.

        if channels != self.in_channels:
            raise ValueError(
                "Input channel mismatch: "
                f"expected {self.in_channels}, got {channels}."
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                "Input spatial size mismatch: "
                f"expected [{self.image_size}, {self.image_size}], "
                f"got [{height}, {width}]."
            )

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "Input spatial size must be divisible by patch_size: "
                f"image_size={height}x{width}, patch_size={self.patch_size}."
            )

    def _patchify_and_project(self, image: torch.Tensor) -> torch.Tensor:
        """Converts images into projected patch tokens.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            Projected patch tokens with shape [B, N, D].
        """
        batch_size, channels, height, width = image.shape
        patch_size = self.patch_size

        grid_h = height // patch_size
        grid_w = width // patch_size
        num_patches = grid_h * grid_w
        patch_dim = channels * patch_size * patch_size

        patches = image.reshape(
            batch_size,
            channels,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(batch_size, num_patches, patch_dim)

        patch_tokens = self.proj(patches)
        return patch_tokens


def _validate_init_args(
    image_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
) -> None:
    """Validates DummyBackbone initialization arguments."""
    if image_size <= 0:
        raise ValueError(f"image_size must be > 0, got {image_size}.")

    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}.")

    if in_channels <= 0:
        raise ValueError(f"in_channels must be > 0, got {in_channels}.")

    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be > 0, got {embed_dim}.")

    if image_size % patch_size != 0:
        raise ValueError(
            "image_size must be divisible by patch_size: "
            f"got image_size={image_size}, patch_size={patch_size}."
        )