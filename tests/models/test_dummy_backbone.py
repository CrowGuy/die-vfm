"""Tests for die_vfm.models.backbone.dummy_backbone."""

from __future__ import annotations

import pytest
import torch

from die_vfm.models.backbone.dummy_backbone import DummyBackbone
from die_vfm.models.outputs import BackboneOutput


def test_dummy_backbone_output_dim() -> None:
    """Returns the configured token feature dimension."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )

    assert backbone.output_dim == 192


def test_dummy_backbone_forward_output_contract() -> None:
    """Produces a valid BackboneOutput with expected shapes."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )
    image = torch.randn(4, 3, 224, 224)

    output = backbone(image)

    assert isinstance(output, BackboneOutput)
    assert output.patch_tokens.shape == (4, 196, 192)
    assert output.cls_token is None
    assert output.token_mask is None
    assert output.feature_dim == 192
    assert output.patch_grid == (14, 14)

    assert output.metadata["backbone_name"] == "dummy"
    assert output.metadata["image_size"] == 224
    assert output.metadata["patch_size"] == 16
    assert output.metadata["num_patches"] == 196


def test_dummy_backbone_forward_preserves_device_and_dtype() -> None:
    """Produces patch tokens on the same device and dtype as the module."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=64,
    )
    image = torch.randn(2, 3, 224, 224)

    output = backbone(image)

    assert output.patch_tokens.device == image.device
    assert output.patch_tokens.dtype == backbone.proj.weight.dtype


@pytest.mark.parametrize(
    ("image_size", "patch_size", "in_channels", "embed_dim"),
    [
        (0, 16, 3, 192),
        (224, 0, 3, 192),
        (224, 16, 0, 192),
        (224, 16, 3, 0),
        (230, 16, 3, 192),
    ],
)
def test_dummy_backbone_init_raises_for_invalid_args(
    image_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
) -> None:
    """Raises when initialization arguments are invalid."""
    with pytest.raises(ValueError):
        DummyBackbone(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )


def test_dummy_backbone_raises_for_non_tensor_input() -> None:
    """Raises when input is not a torch.Tensor."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        backbone("not a tensor")  # type: ignore[arg-type]


def test_dummy_backbone_raises_for_non_4d_input() -> None:
    """Raises when input rank is not 4."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )
    image = torch.randn(3, 224, 224)

    with pytest.raises(ValueError, match="\\[B, C, H, W\\]"):
        backbone(image)


def test_dummy_backbone_raises_for_channel_mismatch() -> None:
    """Raises when input channel count does not match config."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )
    image = torch.randn(2, 1, 224, 224)

    with pytest.raises(ValueError, match="channel mismatch"):
        backbone(image)


def test_dummy_backbone_raises_for_spatial_size_mismatch() -> None:
    """Raises when input spatial size does not match config."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
    )
    image = torch.randn(2, 3, 256, 256)

    with pytest.raises(ValueError, match="spatial size mismatch"):
        backbone(image)


def test_dummy_backbone_num_patches_matches_patch_grid() -> None:
    """Keeps num_patches consistent with patch_grid."""
    backbone = DummyBackbone(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=32,
    )
    image = torch.randn(1, 3, 224, 224)

    output = backbone(image)

    grid_h, grid_w = output.patch_grid
    assert grid_h * grid_w == output.patch_tokens.shape[1]
    assert backbone.num_patches == output.patch_tokens.shape[1]