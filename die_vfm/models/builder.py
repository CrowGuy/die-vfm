"""Model builder for die_vfm."""

from __future__ import annotations

from omegaconf import DictConfig

from die_vfm.models.backbone.builder import build_backbone
from die_vfm.models.model import DieVFMModel
from die_vfm.models.pooler.builder import build_pooler


def build_model(cfg: DictConfig) -> DieVFMModel:
    """Builds the top-level DieVFM model from config."""
    _validate_model_config(cfg)

    backbone = build_backbone(cfg.backbone)
    pooler = build_pooler(
        cfg.pooler,
        backbone_output_dim=backbone.output_dim,
    )

    return_debug_outputs = cfg.get("return_debug_outputs", True)

    return DieVFMModel(
        backbone=backbone,
        pooler=pooler,
        return_debug_outputs=return_debug_outputs,
    )


def _validate_model_config(cfg: DictConfig) -> None:
    """Validates the minimal model config contract."""
    required_fields = ("backbone", "pooler")

    for field_name in required_fields:
        if field_name not in cfg:
            raise KeyError(f"Missing required model config field: '{field_name}'.")