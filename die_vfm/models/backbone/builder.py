"""Backbone builder for die_vfm."""

from __future__ import annotations

from omegaconf import DictConfig

from die_vfm.models.backbone.base import Backbone
from die_vfm.models.backbone.dummy_backbone import DummyBackbone


def build_backbone(cfg: DictConfig) -> Backbone:
    """Builds a backbone from config.

    Expected config schema example:

        backbone:
          name: dummy
          image_size: 224
          patch_size: 16
          in_channels: 3
          embed_dim: 192

    Args:
        cfg: Backbone config.

    Returns:
        An initialized Backbone instance.

    Raises:
        KeyError: If required config fields are missing.
        ValueError: If the backbone name is unsupported.
    """
    _validate_backbone_config(cfg)

    name = cfg.name

    if name == "dummy":
        return DummyBackbone(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )

    raise ValueError(f"Unsupported backbone: '{name}'.")


def _validate_backbone_config(cfg: DictConfig) -> None:
    """Validates the minimal backbone config contract.

    Args:
        cfg: Backbone config.

    Raises:
        KeyError: If required config fields are missing.
    """
    if "name" not in cfg:
        raise KeyError("Missing required backbone config field: 'name'.")

    if cfg.name == "dummy":
        required_fields = (
            "image_size",
            "patch_size",
            "in_channels",
            "embed_dim",
        )
        for field_name in required_fields:
            if field_name not in cfg:
                raise KeyError(
                    f"Missing required dummy backbone config field: "
                    f"'{field_name}'."
                )