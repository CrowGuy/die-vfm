"""Tests for die_vfm.models.backbone.builder."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from die_vfm.models.backbone.builder import build_backbone
from die_vfm.models.backbone.dinov2_backbone import DINOv2Backbone
from die_vfm.models.backbone.dummy_backbone import DummyBackbone


class _FakeHubModel(nn.Module):
    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        self._feature_dim = feature_dim

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = image.shape
        num_patches = (height // 14) * (width // 14)
        return {
            "x_norm_patchtokens": torch.ones(
                (batch_size, num_patches, self._feature_dim),
            ),
            "x_norm_clstoken": torch.ones((batch_size, self._feature_dim)),
        }


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


def test_build_backbone_with_dinov2_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds a DINOv2 backbone from config without hitting network hub."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(),
    )
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": False,
            "freeze": False,
            "return_cls_token": True,
            "allow_network": True,
            "local_repo_path": None,
            "local_checkpoint_path": None,
        }
    )

    backbone = build_backbone(cfg)

    assert isinstance(backbone, DINOv2Backbone)
    assert backbone.variant == "vit_base"
    assert backbone.pretrained is False
    assert backbone.freeze is False
    assert backbone.return_cls_token is True
    assert backbone.output_dim == 768


def test_build_backbone_raises_for_missing_dinov2_variant() -> None:
    """Raises when required dinov2 fields are missing."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": True,
        }
    )

    with pytest.raises(KeyError, match="variant"):
        build_backbone(cfg)


def test_build_backbone_raises_for_unsupported_dinov2_variant() -> None:
    """Raises with explicit variant support list for unsupported variant."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_xx",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": True,
            "allow_network": True,
            "local_repo_path": None,
            "local_checkpoint_path": None,
        }
    )

    with pytest.raises(ValueError, match="Unsupported DINOv2 variant"):
        build_backbone(cfg)


def test_build_backbone_raises_for_non_bool_dinov2_pretrained() -> None:
    """Raises when dinov2.pretrained is not bool."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": "true",
            "freeze": False,
            "return_cls_token": True,
            "allow_network": True,
            "local_repo_path": None,
            "local_checkpoint_path": None,
        }
    )

    with pytest.raises(TypeError, match="pretrained"):
        build_backbone(cfg)


def test_build_backbone_raises_for_non_bool_dinov2_return_cls_token() -> None:
    """Raises when dinov2.return_cls_token is not bool."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": 1,
            "allow_network": True,
            "local_repo_path": None,
            "local_checkpoint_path": None,
        }
    )

    with pytest.raises(TypeError, match="return_cls_token"):
        build_backbone(cfg)


def test_build_backbone_raises_for_non_bool_dinov2_allow_network() -> None:
    """Raises when dinov2.allow_network is not bool."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": True,
            "allow_network": "false",
            "local_repo_path": None,
            "local_checkpoint_path": None,
        }
    )

    with pytest.raises(TypeError, match="allow_network"):
        build_backbone(cfg)


def test_build_backbone_raises_for_non_string_dinov2_local_repo_path() -> None:
    """Raises when dinov2.local_repo_path is not str/None."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": True,
            "allow_network": True,
            "local_repo_path": 123,
            "local_checkpoint_path": None,
        }
    )

    with pytest.raises(TypeError, match="local_repo_path"):
        build_backbone(cfg)


def test_build_backbone_raises_for_non_string_dinov2_local_checkpoint_path() -> None:
    """Raises when dinov2.local_checkpoint_path is not str/None."""
    cfg = OmegaConf.create(
        {
            "name": "dinov2",
            "variant": "vit_base",
            "pretrained": True,
            "freeze": False,
            "return_cls_token": True,
            "allow_network": True,
            "local_repo_path": None,
            "local_checkpoint_path": 123,
        }
    )

    with pytest.raises(TypeError, match="local_checkpoint_path"):
        build_backbone(cfg)
