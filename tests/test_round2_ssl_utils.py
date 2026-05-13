from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from die_vfm.models.builder import build_model
from die_vfm.trainer.round2_ssl import FlipMetadata
from die_vfm.trainer.round2_ssl import Round2SSLModule
from die_vfm.trainer.round2_ssl import apply_update_mode
from die_vfm.trainer.round2_ssl import canonicalize_patch_tokens
from die_vfm.trainer.round2_ssl import reconcile_token_loss_trainability
from die_vfm.trainer.round2_ssl import resolve_trainable_parameters
from die_vfm.trainer.round2_ssl import update_teacher_ema
from die_vfm.trainer.round2_ssl import validate_round2_train_contract


def _build_cfg():
    return OmegaConf.create(
        {
            "train": {
                "update_mode": "full_backbone",
                "last_n_blocks": None,
                "precision_mode": "fp32",
            },
            "round2": {
                "ema": {
                    "policy": "fixed",
                    "momentum": 0.996,
                    "final_momentum": 0.999,
                },
                "evaluation": {
                    "cadence": "end_only",
                },
                "loss": {
                    "token_loss_weight": 0.2,
                },
            },
        }
    )


def _build_ssl_module() -> Round2SSLModule:
    model_cfg = OmegaConf.create(
        {
            "backbone": {
                "name": "dummy",
                "image_size": 32,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 8,
            },
            "pooler": {
                "name": "attn_pooler_v1",
                "hidden_dim": 8,
                "dropout": 0.0,
                "l2_norm": False,
            },
            "return_debug_outputs": True,
        }
    )
    student_encoder = build_model(model_cfg)
    return Round2SSLModule.from_student_encoder(
        student_encoder=student_encoder,
        global_hidden_dim=None,
        global_output_dim=None,
        global_num_layers=2,
        token_hidden_dim=None,
        token_output_dim=None,
        token_num_layers=2,
    )


def test_validate_round2_train_contract_rejects_invalid_last_n_blocks() -> None:
    cfg = _build_cfg()
    cfg.train.update_mode = "full_backbone"
    cfg.train.last_n_blocks = 2

    with pytest.raises(ValueError, match="train.last_n_blocks must be null"):
        validate_round2_train_contract(cfg)


def test_apply_update_mode_projector_pooler_only_excludes_teacher_params() -> None:
    ssl_module = _build_ssl_module()

    resolution = apply_update_mode(
        ssl_module,
        update_mode="projector_pooler_only",
        last_n_blocks=None,
    )

    assert resolution.update_mode == "projector_pooler_only"
    assert resolution.trainable_block_indices == []
    assert all(
        not parameter.requires_grad
        for parameter in ssl_module.teacher_encoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in ssl_module.student_encoder.backbone.parameters()
    )
    pooler_parameters = list(ssl_module.student_encoder.pooler.parameters())
    assert pooler_parameters
    assert all(parameter.requires_grad for parameter in pooler_parameters)

    trainable_parameters = resolve_trainable_parameters(ssl_module)
    assert trainable_parameters
    assert all(parameter.requires_grad for parameter in trainable_parameters)


def test_reconcile_token_loss_trainability_freezes_student_token_projector_when_disabled() -> None:
    ssl_module = _build_ssl_module()

    update_resolution = apply_update_mode(
        ssl_module,
        update_mode="projector_pooler_only",
        last_n_blocks=None,
    )
    reconciled_resolution = reconcile_token_loss_trainability(
        ssl_module,
        update_resolution=update_resolution,
        token_loss_enabled=False,
    )

    assert "student_token_projector" not in reconciled_resolution.trainable_module_names
    assert all(
        not parameter.requires_grad
        for parameter in ssl_module.student_token_projector.parameters()
    )


def test_reconcile_token_loss_trainability_keeps_student_token_projector_when_enabled() -> None:
    ssl_module = _build_ssl_module()

    update_resolution = apply_update_mode(
        ssl_module,
        update_mode="projector_pooler_only",
        last_n_blocks=None,
    )
    reconciled_resolution = reconcile_token_loss_trainability(
        ssl_module,
        update_resolution=update_resolution,
        token_loss_enabled=True,
    )

    assert "student_token_projector" in reconciled_resolution.trainable_module_names
    projector_parameters = list(ssl_module.student_token_projector.parameters())
    assert projector_parameters
    assert all(parameter.requires_grad for parameter in projector_parameters)


def test_update_teacher_ema_tracks_student_parameters() -> None:
    ssl_module = _build_ssl_module()

    student_parameter = next(ssl_module.student_global_projector.parameters())
    teacher_parameter = next(ssl_module.teacher_global_projector.parameters())
    original_teacher = teacher_parameter.detach().clone()

    with torch.no_grad():
        student_parameter.add_(1.0)

    update_teacher_ema(ssl_module, momentum=0.0)
    assert not torch.allclose(original_teacher, teacher_parameter)
    assert torch.allclose(student_parameter, teacher_parameter)


@pytest.mark.parametrize(
    ("horizontal", "vertical", "expected"),
    [
        (False, False, [0.0, 1.0, 2.0, 3.0]),
        (True, False, [0.0, 1.0, 2.0, 3.0]),
        (False, True, [0.0, 1.0, 2.0, 3.0]),
        (True, True, [0.0, 1.0, 2.0, 3.0]),
    ],
)
def test_canonicalize_patch_tokens_handles_flip_variants(
    horizontal: bool,
    vertical: bool,
    expected: list[float],
) -> None:
    original = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    viewed = original.reshape(1, 2, 2, 1)
    if horizontal:
        viewed = torch.flip(viewed, dims=[2])
    if vertical:
        viewed = torch.flip(viewed, dims=[1])
    viewed = viewed.reshape(1, 4, 1)

    canonical = canonicalize_patch_tokens(
        viewed,
        patch_grid=(2, 2),
        flip_metadata=FlipMetadata(
            horizontal=torch.tensor([horizontal]),
            vertical=torch.tensor([vertical]),
        ),
    )

    assert canonical.squeeze(-1).tolist()[0] == expected
