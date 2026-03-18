"""Smoke tests for the end-to-end die_vfm model forward path."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from die_vfm.models.builder import build_model
from die_vfm.models.outputs import BackboneOutput
from die_vfm.models.outputs import ModelOutput
from die_vfm.models.outputs import PoolerOutput


def test_model_forward_smoke_with_mean_pooler() -> None:
    """Runs the PR-3 model pipeline end to end with mean pooling."""
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
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "label": None,
        "image_id": [f"img_{i}" for i in range(4)],
        "meta": [{} for _ in range(4)],
    }

    output = model(batch["image"])

    assert isinstance(output, ModelOutput)
    assert output.embedding.shape == (4, 192)

    assert output.backbone is not None
    assert output.pooler is not None

    assert isinstance(output.backbone, BackboneOutput)
    assert isinstance(output.pooler, PoolerOutput)

    assert output.backbone.patch_tokens.shape == (4, 196, 192)
    assert output.backbone.cls_token is None
    assert output.backbone.token_mask is None
    assert output.backbone.feature_dim == 192
    assert output.backbone.patch_grid == (14, 14)

    assert output.pooler.embedding.shape == (4, 192)
    assert output.pooler.token_weights is None

    assert output.metadata["embedding_dim"] == 192


def test_model_forward_smoke_with_identity_pooler() -> None:
    """Runs the PR-3 model pipeline end to end with identity pooling."""
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
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    model = build_model(cfg)
    image = torch.randn(2, 3, 224, 224)

    output = model(image)

    assert isinstance(output, ModelOutput)
    assert output.embedding.shape == (2, 128)

    assert output.backbone is not None
    assert output.pooler is not None
    assert output.backbone.patch_tokens.shape == (2, 196, 128)
    assert output.pooler.embedding.shape == (2, 128)


def test_model_forward_omits_debug_outputs_when_disabled() -> None:
    """Omits backbone and pooler outputs when debug outputs are disabled."""
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
                "l2_norm": False,
            },
            "return_debug_outputs": False,
        }
    )

    model = build_model(cfg)
    image = torch.randn(3, 3, 224, 224)

    output = model(image)

    assert isinstance(output, ModelOutput)
    assert output.embedding.shape == (3, 64)
    assert output.backbone is None
    assert output.pooler is None
    assert output.metadata["embedding_dim"] == 64


def test_model_forward_raises_for_non_tensor_input() -> None:
    """Raises when model input is not a torch.Tensor."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 32,
            },
            "pooler": {
                "name": "mean",
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    model = build_model(cfg)

    with pytest.raises(TypeError, match="torch.Tensor"):
        model("not a tensor")  # type: ignore[arg-type]


def test_model_forward_raises_for_non_4d_input() -> None:
    """Raises when model input rank is not 4."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 32,
            },
            "pooler": {
                "name": "mean",
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    model = build_model(cfg)
    image = torch.randn(3, 224, 224)

    with pytest.raises(ValueError, match="\\[B, C, H, W\\]"):
        model(image)


def test_model_forward_raises_for_invalid_spatial_size() -> None:
    """Raises when image spatial size does not match backbone config."""
    cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 32,
            },
            "pooler": {
                "name": "mean",
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )

    model = build_model(cfg)
    image = torch.randn(2, 3, 256, 256)

    with pytest.raises(ValueError, match="spatial size mismatch"):
        model(image)