"""Tests for die_vfm.models.backbone.dinov2_backbone."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from die_vfm.models.backbone.dinov2_backbone import DINOv2Backbone
from die_vfm.models.outputs import BackboneOutput


class _FakeHubModel(nn.Module):
    def __init__(self, *, feature_dim: int, include_cls_token: bool = True) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.include_cls_token = include_cls_token
        self.weight = nn.Parameter(torch.ones(1))

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = image.shape
        num_patches = (height // 14) * (width // 14)
        output = {
            "x_norm_patchtokens": torch.randn(
                (batch_size, num_patches, self.feature_dim),
                device=image.device,
                dtype=image.dtype,
            ),
        }
        if self.include_cls_token:
            output["x_norm_clstoken"] = torch.randn(
                (batch_size, self.feature_dim),
                device=image.device,
                dtype=image.dtype,
            )
        return output


class _FakeHubModelUnsupportedOutput(nn.Module):
    def forward_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [image]


@pytest.mark.parametrize(
    ("variant", "expected_model_name", "expected_output_dim"),
    [
        ("vit_small", "dinov2_vits14", 384),
        ("vit_base", "dinov2_vitb14", 768),
        ("vit_large", "dinov2_vitl14", 1024),
        ("vit_giant", "dinov2_vitg14", 1536),
    ],
)
def test_dinov2_backbone_variant_mapping(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_model_name: str,
    expected_output_dim: int,
) -> None:
    """Maps variant to expected torch.hub entry and output_dim."""
    call_args: dict[str, object] = {}

    def _fake_load(repo_or_dir: str, model_name: str, **kwargs) -> nn.Module:
        call_args["repo_or_dir"] = repo_or_dir
        call_args["model_name"] = model_name
        call_args.update(kwargs)
        return _FakeHubModel(feature_dim=expected_output_dim)

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        _fake_load,
    )

    backbone = DINOv2Backbone(
        variant=variant,
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )

    assert backbone.output_dim == expected_output_dim
    assert call_args["repo_or_dir"] == "facebookresearch/dinov2"
    assert call_args["model_name"] == expected_model_name
    assert call_args["pretrained"] is False
    assert call_args["source"] == "github"


def test_dinov2_backbone_freeze_disables_parameter_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freezes all hub-model parameters when freeze=True."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )

    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=True,
        return_cls_token=True,
    )

    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


def test_dinov2_backbone_forward_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produces valid BackboneOutput from parsed forward_features mapping."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )
    image = torch.randn(2, 3, 224, 224)

    output = backbone(image)

    assert isinstance(output, BackboneOutput)
    assert output.patch_tokens.shape == (2, 256, 768)
    assert output.cls_token is not None
    assert output.cls_token.shape == (2, 768)
    assert output.token_mask is None
    assert output.feature_dim == 768
    assert output.patch_grid == (16, 16)
    assert output.metadata["backbone_name"] == "dinov2"
    assert output.metadata["variant"] == "vit_base"
    assert output.metadata["patch_size"] == 14


def test_dinov2_backbone_forward_omits_cls_token_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suppresses cls_token from output when return_cls_token=False."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=False,
    )
    image = torch.randn(1, 3, 224, 224)

    output = backbone(image)

    assert output.cls_token is None


def test_dinov2_backbone_raises_for_non_square_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises when H and W do not match."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )
    image = torch.randn(1, 3, 224, 196)

    with pytest.raises(ValueError, match="square inputs"):
        backbone(image)


def test_dinov2_backbone_raises_for_non_divisible_spatial_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises when image size is not divisible by patch size 14."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )
    image = torch.randn(1, 3, 225, 225)

    with pytest.raises(ValueError, match="divisible by DINOv2 patch size 14"):
        backbone(image)


def test_dinov2_backbone_raises_for_unsupported_forward_features_output_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises when hub model output is not mapping-like."""
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModelUnsupportedOutput(),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )
    image = torch.randn(1, 3, 224, 224)

    with pytest.raises(RuntimeError, match="Unsupported DINOv2 forward_features"):
        backbone(image)


def test_dinov2_backbone_raises_for_missing_patch_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raises when forward_features omits x_norm_patchtokens."""

    class _MissingPatchTokens(nn.Module):
        def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"x_norm_clstoken": torch.randn(image.shape[0], 768)}

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _MissingPatchTokens(),
    )
    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
    )
    image = torch.randn(1, 3, 224, 224)

    with pytest.raises(RuntimeError, match="missing 'x_norm_patchtokens'"):
        backbone(image)


def test_dinov2_backbone_uses_local_repo_source_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Uses source=local and local repo path for architecture resolution."""
    local_repo = tmp_path / "dinov2_repo"
    local_repo.mkdir()
    call_args: dict[str, object] = {}

    def _fake_load(repo_or_dir: str, model_name: str, **kwargs) -> nn.Module:
        call_args["repo_or_dir"] = repo_or_dir
        call_args["model_name"] = model_name
        call_args.update(kwargs)
        return _FakeHubModel(feature_dim=768)

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        _fake_load,
    )

    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=False,
        freeze=False,
        return_cls_token=True,
        allow_network=False,
        local_repo_path=str(local_repo),
        local_checkpoint_path=None,
    )

    assert isinstance(backbone, DINOv2Backbone)
    assert call_args["repo_or_dir"] == str(local_repo)
    assert call_args["model_name"] == "dinov2_vitb14"
    assert call_args["source"] == "local"
    assert call_args["pretrained"] is False


def test_dinov2_backbone_rejects_offline_mode_without_local_repo() -> None:
    """Fails fast when offline mode has no architecture source."""
    with pytest.raises(ValueError, match="architecture source is unavailable"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=False,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=None,
            local_checkpoint_path=None,
        )


def test_dinov2_backbone_rejects_pretrained_offline_without_local_checkpoint(
    tmp_path: Path,
) -> None:
    """Fails fast when offline pretrained mode has no local checkpoint."""
    local_repo = tmp_path / "dinov2_repo"
    local_repo.mkdir()

    with pytest.raises(ValueError, match="pretrained offline load requires"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=True,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=str(local_repo),
            local_checkpoint_path=None,
        )


def test_dinov2_backbone_rejects_nonexistent_local_repo_path() -> None:
    """Fails fast for missing local repo path."""
    with pytest.raises(ValueError, match="local repo does not exist"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=False,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path="/tmp/not_exists_dinov2_repo",
            local_checkpoint_path=None,
        )


def test_dinov2_backbone_rejects_local_repo_path_that_is_file(
    tmp_path: Path,
) -> None:
    """Fails fast when local_repo_path points to a file."""
    local_repo_file = tmp_path / "repo.txt"
    local_repo_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="local repo must point to a directory"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=False,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=str(local_repo_file),
            local_checkpoint_path=None,
        )


def test_dinov2_backbone_rejects_local_checkpoint_when_pretrained_false(
    tmp_path: Path,
) -> None:
    """Fails fast when checkpoint is set while pretrained=false."""
    checkpoint_path = tmp_path / "local.ckpt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="only valid when model.backbone.pretrained=true"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=False,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=None,
            local_checkpoint_path=str(checkpoint_path),
        )


def test_dinov2_backbone_rejects_nonexistent_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fails fast when local checkpoint path is missing."""
    local_repo = tmp_path / "dinov2_repo"
    local_repo.mkdir()

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )

    missing_checkpoint = local_repo / "missing.ckpt"
    with pytest.raises(ValueError, match="local checkpoint does not exist"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=True,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=str(local_repo),
            local_checkpoint_path=str(missing_checkpoint),
        )


def test_dinov2_backbone_loads_local_checkpoint_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Loads local checkpoint and disables hub pretrained resolution."""
    local_repo = tmp_path / "dinov2_repo"
    local_repo.mkdir()
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    call_args: dict[str, object] = {}

    def _fake_load(repo_or_dir: str, model_name: str, **kwargs) -> nn.Module:
        call_args["repo_or_dir"] = repo_or_dir
        call_args["model_name"] = model_name
        call_args.update(kwargs)
        return _FakeHubModel(feature_dim=768)

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        _fake_load,
    )
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.load",
        lambda *args, **kwargs: {"weight": torch.ones(1)},
    )

    backbone = DINOv2Backbone(
        variant="vit_base",
        pretrained=True,
        freeze=False,
        return_cls_token=True,
        allow_network=False,
        local_repo_path=str(local_repo),
        local_checkpoint_path=str(checkpoint_path),
    )

    assert isinstance(backbone, DINOv2Backbone)
    assert call_args["source"] == "local"
    assert call_args["pretrained"] is False


def test_dinov2_backbone_rejects_invalid_local_checkpoint_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fails fast when local checkpoint payload has no valid state_dict."""
    local_repo = tmp_path / "dinov2_repo"
    local_repo.mkdir()
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.hub.load",
        lambda *args, **kwargs: _FakeHubModel(feature_dim=768),
    )
    monkeypatch.setattr(
        "die_vfm.models.backbone.dinov2_backbone.torch.load",
        lambda *args, **kwargs: {"foo": "bar"},
    )

    with pytest.raises(ValueError, match="does not contain a valid state dict"):
        DINOv2Backbone(
            variant="vit_base",
            pretrained=True,
            freeze=False,
            return_cls_token=True,
            allow_network=False,
            local_repo_path=str(local_repo),
            local_checkpoint_path=str(checkpoint_path),
        )
