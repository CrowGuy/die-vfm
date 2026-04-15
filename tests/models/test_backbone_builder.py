"""Tests for die_vfm.models.backbone.builder."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from die_vfm.models.backbone.builder import build_backbone
from die_vfm.models.backbone.dummy_backbone import DummyBackbone


def test_build_backbone_with_dummy_config() -> None:
    """Builds the current supported dummy backbone from config."""
    cfg = OmegaConf.create(
        {
            "name": "dummy",
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 192,
        }
    )

    backbone = build_backbone(cfg)

    assert isinstance(backbone, DummyBackbone)
    assert backbone.image_size == 224
    assert backbone.patch_size == 16
    assert backbone.in_channels == 3
    assert backbone.output_dim == 192


def test_build_backbone_raises_for_missing_dummy_field() -> None:
    """Raises when the dummy backbone config omits a required field."""
    cfg = OmegaConf.create(
        {
            "name": "dummy",
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
        }
    )

    with pytest.raises(KeyError, match="embed_dim"):
        build_backbone(cfg)


def test_build_backbone_raises_for_missing_name() -> None:
    """Raises when the backbone config omits the name field."""
    cfg = OmegaConf.create(
        {
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 192,
        }
    )

    with pytest.raises(KeyError, match="name"):
        build_backbone(cfg)


def test_build_backbone_raises_for_unsupported_name() -> None:
    """Raises when the backbone name is not currently supported."""
    cfg = OmegaConf.create({"name": "unsupported_backbone"})

    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_backbone(cfg)
