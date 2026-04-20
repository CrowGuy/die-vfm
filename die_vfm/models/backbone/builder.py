"""Backbone builder for die_vfm."""

from __future__ import annotations

from omegaconf import DictConfig

from die_vfm.models.backbone.base import Backbone
from die_vfm.models.backbone.dinov2_backbone import DINOv2Backbone
from die_vfm.models.backbone.dummy_backbone import DummyBackbone

_DINOV2_SUPPORTED_VARIANTS = (
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_giant",
)


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

    if name == "dinov2":
        return DINOv2Backbone(
            variant=cfg.variant,
            pretrained=cfg.pretrained,
            freeze=cfg.freeze,
            return_cls_token=cfg.return_cls_token,
            allow_network=cfg.allow_network,
            local_repo_path=cfg.local_repo_path,
            local_checkpoint_path=cfg.local_checkpoint_path,
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
        return

    if cfg.name == "dinov2":
        required_fields = (
            "variant",
            "pretrained",
            "freeze",
            "return_cls_token",
            "allow_network",
            "local_repo_path",
            "local_checkpoint_path",
        )
        for field_name in required_fields:
            if field_name not in cfg:
                raise KeyError(
                    f"Missing required dinov2 backbone config field: "
                    f"'{field_name}'."
                )

        if not isinstance(cfg.variant, str):
            raise TypeError(
                "Expected dinov2 backbone config field 'variant' to be str."
            )

        if cfg.variant not in _DINOV2_SUPPORTED_VARIANTS:
            supported = ", ".join(_DINOV2_SUPPORTED_VARIANTS)
            raise ValueError(
                f"Unsupported DINOv2 variant: '{cfg.variant}'. "
                f"Supported variants: {supported}."
            )

        bool_fields = ("pretrained", "freeze", "return_cls_token")
        for field_name in bool_fields:
            if not isinstance(cfg[field_name], bool):
                raise TypeError(
                    "Expected dinov2 backbone config field "
                    f"'{field_name}' to be bool."
                )

        if not isinstance(cfg.allow_network, bool):
            raise TypeError(
                "Expected dinov2 backbone config field 'allow_network' to be bool."
            )

        path_fields = ("local_repo_path", "local_checkpoint_path")
        for field_name in path_fields:
            value = cfg[field_name]
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    "Expected dinov2 backbone config field "
                    f"'{field_name}' to be str or None."
                )
