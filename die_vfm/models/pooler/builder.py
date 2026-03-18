"""Pooler builder for die_vfm."""

from __future__ import annotations

from omegaconf import DictConfig

from die_vfm.models.pooler.base import Pooler
from die_vfm.models.pooler.identity_pooler import IdentityPooler
from die_vfm.models.pooler.mean_pooler import MeanPooler


def build_pooler(cfg: DictConfig, backbone_output_dim: int) -> Pooler:
    """Builds a pooler from config.

    Expected config schema example:

        pooler:
          name: mean
          l2_norm: false

    Args:
        cfg: Pooler config.
        backbone_output_dim: Token feature dimension produced by the backbone.

    Returns:
        An initialized Pooler instance.

    Raises:
        KeyError: If required config fields are missing.
        ValueError: If the pooler name is unsupported or config is invalid.
    """
    _validate_pooler_config(cfg, backbone_output_dim)

    name = cfg.name

    if name == "mean":
        return MeanPooler(
            input_dim=backbone_output_dim,
            l2_norm=cfg.get("l2_norm", False),
        )

    if name == "identity":
        return IdentityPooler(
            input_dim=backbone_output_dim,
            l2_norm=cfg.get("l2_norm", False),
        )

    raise ValueError(f"Unsupported pooler: '{name}'.")


def _validate_pooler_config(
    cfg: DictConfig,
    backbone_output_dim: int,
) -> None:
    """Validates the minimal pooler config contract.

    Args:
        cfg: Pooler config.
        backbone_output_dim: Token feature dimension from the backbone.

    Raises:
        KeyError: If required fields are missing.
        ValueError: If config values are invalid.
    """
    if "name" not in cfg:
        raise KeyError("Missing required pooler config field: 'name'.")

    if backbone_output_dim <= 0:
        raise ValueError(
            "backbone_output_dim must be > 0, "
            f"got {backbone_output_dim}."
        )