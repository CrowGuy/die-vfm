from __future__ import annotations

from omegaconf import OmegaConf

from die_vfm.models.pooler.attn_pooler_v1 import AttnPoolerV1
from die_vfm.models.pooler.builder import build_pooler

def test_build_pooler_attn_pooler_v1() -> None:
    """Builds AttnPoolerV1 from config."""
    cfg = OmegaConf.create(
        {
            "name": "attn_pooler_v1",
            "hidden_dim": 128,
            "output_dim": 64,
            "dropout": 0.1,
            "l2_norm": True,
            "use_cls_token_as_query": True,
            "return_token_weights": False,
        }
    )

    pooler = build_pooler(cfg=cfg, backbone_output_dim=256)

    assert isinstance(pooler, AttnPoolerV1)
    assert pooler.input_dim == 256
    assert pooler.hidden_dim == 128
    assert pooler.output_dim == 64
    assert pooler.dropout == 0.1
    assert pooler.l2_norm is True
    assert pooler.use_cls_token_as_query is True
    assert pooler.return_token_weights is False

def test_build_pooler_attn_pooler_v1_invalid_hidden_dim_raises() -> None:
    """Raises when hidden_dim is invalid for AttnPoolerV1."""
    from omegaconf import OmegaConf
    import pytest

    from die_vfm.models.pooler.builder import build_pooler

    # hidden_dim = 0
    cfg_zero = OmegaConf.create(
        {
            "name": "attn_pooler_v1",
            "hidden_dim": 0,
        }
    )

    with pytest.raises(ValueError, match="hidden_dim must be > 0"):
        build_pooler(cfg=cfg_zero, backbone_output_dim=256)

    # hidden_dim < 0
    cfg_negative = OmegaConf.create(
        {
            "name": "attn_pooler_v1",
            "hidden_dim": -16,
        }
    )

    with pytest.raises(ValueError, match="hidden_dim must be > 0"):
        build_pooler(cfg=cfg_negative, backbone_output_dim=256)


def test_build_pooler_attn_pooler_v1_missing_hidden_dim_raises() -> None:
    """Raises when hidden_dim is missing for AttnPoolerV1."""
    from omegaconf import OmegaConf
    import pytest

    from die_vfm.models.pooler.builder import build_pooler

    cfg = OmegaConf.create(
        {
            "name": "attn_pooler_v1",
        }
    )

    with pytest.raises(ValueError, match="requires.*hidden_dim"):
        build_pooler(cfg=cfg, backbone_output_dim=256)