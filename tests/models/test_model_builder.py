"""Tests for die_vfm.models.builder."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from die_vfm.models.builder import build_model
from die_vfm.models.model import DieVFMModel
from die_vfm.models.backbone.dummy_backbone import DummyBackbone
from die_vfm.models.pooler.identity_pooler import IdentityPooler
from die_vfm.models.pooler.mean_pooler import MeanPooler


def test_build_model_with_mean_pooler() -> None:
    """Builds a DieVFMModel with dummy backbone and mean pooler."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 192,
            },
            "pooler": {
                "name": "mean",
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    model = build_model(cfg)

    assert isinstance(model, DieVFMModel)
    assert isinstance(model.backbone, DummyBackbone)
    assert isinstance(model.pooler, MeanPooler)
    assert model.embedding_dim == 192
    assert model.return_debug_outputs is True


def test_build_model_with_identity_pooler() -> None:
    """Builds a DieVFMModel with dummy backbone and identity pooler."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 128,
            },
            "pooler": {
                "name": "identity",
                "l2_norm": True,
            },
            "return_debug_outputs": False,
        }
    )

    model = build_model(cfg)

    assert isinstance(model, DieVFMModel)
    assert isinstance(model.backbone, DummyBackbone)
    assert isinstance(model.pooler, IdentityPooler)
    assert model.embedding_dim == 128
    assert model.return_debug_outputs is False


def test_build_model_defaults_return_debug_outputs_to_true() -> None:
    """Uses default return_debug_outputs=True when config omits the field."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 64,
            },
            "pooler": {
                "name": "mean",
            },
        }
    )

    model = build_model(cfg)

    assert isinstance(model, DieVFMModel)
    assert model.return_debug_outputs is True


@pytest.mark.parametrize(
    "missing_field",
    [
        "backbone",
        "pooler",
    ],
)
def test_build_model_raises_for_missing_required_top_level_field(
    missing_field: str,
) -> None:
    """Raises when required top-level model config fields are missing."""
    cfg_dict = {
        "backbone": {
            "name": "dummy",
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 192,
        },
        "pooler": {
            "name": "mean",
            "l2_norm": False,
        },
        "return_debug_outputs": True,
    }
    cfg_dict.pop(missing_field)

    cfg = OmegaConf.create(cfg_dict)

    with pytest.raises(KeyError, match=missing_field):
        build_model(cfg)


def test_build_model_raises_for_unsupported_backbone() -> None:
    """Raises when backbone name is unsupported."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "unknown_backbone",
            },
            "pooler": {
                "name": "mean",
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_model(cfg)


def test_build_model_raises_for_unsupported_pooler() -> None:
    """Raises when pooler name is unsupported."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 192,
            },
            "pooler": {
                "name": "unknown_pooler",
            },
            "return_debug_outputs": True,
        }
    )

    with pytest.raises(ValueError, match="Unsupported pooler"):
        build_model(cfg)