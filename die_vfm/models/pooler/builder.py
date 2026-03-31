"""Factory for building pooler modules."""

from __future__ import annotations

from omegaconf import DictConfig

from die_vfm.models.pooler.attn_pooler_v1 import AttnPoolerV1
from die_vfm.models.pooler.identity_pooler import IdentityPooler
from die_vfm.models.pooler.mean_pooler import MeanPooler


SUPPORTED_POOLERS = {
    "identity",
    "mean",
    "attn_pooler_v1",
}


def build_pooler(cfg: DictConfig, backbone_output_dim: int):
    """Builds a pooler from config.

    Args:
        cfg: Pooler config.
        backbone_output_dim: Output feature dimension from backbone.

    Returns:
        Instantiated pooler module.

    Raises:
        ValueError: If config is invalid or pooler name is unsupported.
    """
    _validate_pooler_config(cfg=cfg, backbone_output_dim=backbone_output_dim)

    name = str(cfg.name).lower()

    if name == "identity":
        return IdentityPooler(input_dim=backbone_output_dim)

    if name == "mean":
        return MeanPooler(input_dim=backbone_output_dim)

    if name == "attn_pooler_v1":
        return AttnPoolerV1(
            input_dim=backbone_output_dim,
            hidden_dim=int(cfg.hidden_dim),
            output_dim=_optional_int(cfg.get("output_dim", None)),
            dropout=float(cfg.get("dropout", 0.0)),
            l2_norm=bool(cfg.get("l2_norm", False)),
            use_cls_token_as_query=bool(cfg.get("use_cls_token_as_query", False)),
            return_token_weights=bool(cfg.get("return_token_weights", True)),
        )

    raise ValueError(
        f"Unsupported pooler name: {cfg.name}. "
        f"Supported poolers: {sorted(SUPPORTED_POOLERS)}."
    )


def _validate_pooler_config(cfg: DictConfig, backbone_output_dim: int) -> None:
    """Validates pooler config before construction."""
    if cfg is None:
        raise ValueError("Pooler config must not be None.")

    if "name" not in cfg or cfg.name is None:
        raise ValueError("Pooler config must include a non-empty 'name' field.")

    if backbone_output_dim <= 0:
        raise ValueError(
            f"backbone_output_dim must be > 0, got {backbone_output_dim}."
        )

    name = str(cfg.name).lower()
    if name not in SUPPORTED_POOLERS:
        raise ValueError(
            f"Unsupported pooler name: {cfg.name}. "
            f"Supported poolers: {sorted(SUPPORTED_POOLERS)}."
        )

    if name == "attn_pooler_v1":
        if "hidden_dim" not in cfg or cfg.hidden_dim is None:
            raise ValueError(
                "Pooler 'attn_pooler_v1' requires a non-empty 'hidden_dim' field."
            )

        hidden_dim = int(cfg.hidden_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")

        output_dim = cfg.get("output_dim", None)
        if output_dim is not None and int(output_dim) <= 0:
            raise ValueError(f"output_dim must be > 0 when provided, got {output_dim}.")

        dropout = float(cfg.get("dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")


def _optional_int(value: object) -> int | None:
    """Converts an optional config value to int."""
    if value is None:
        return None
    return int(value)