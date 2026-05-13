"""Round2 SSL helper modules and utilities."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from die_vfm.models.model import DieVFMModel


VALID_UPDATE_MODES = frozenset({
    "projector_pooler_only",
    "last_n_blocks",
    "full_backbone",
})
VALID_PRECISION_MODES = frozenset({"fp32", "bf16"})
VALID_EMA_POLICIES = frozenset({"fixed", "schedule"})


@dataclass(frozen=True)
class UpdateModeResolution:
    """Resolved trainable-boundary information for one Round2 run."""

    update_mode: str
    last_n_blocks: int | None
    total_backbone_blocks: int
    trainable_module_names: list[str]
    trainable_block_indices: list[int]


@dataclass(frozen=True)
class FlipMetadata:
    """Per-sample flip metadata for one augmented view."""

    horizontal: torch.Tensor
    vertical: torch.Tensor


class ProjectionHead(nn.Module):
    """Lightweight MLP projector that supports [B, D] and [B, N, D] inputs."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int | None,
        output_dim: int | None,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}.")

        hidden = int(hidden_dim or input_dim)
        output = int(output_dim or input_dim)

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for layer_index in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden))
            layers.append(nn.GELU())
            current_dim = hidden
        layers.append(nn.Linear(current_dim, output))
        self.network = nn.Sequential(*layers)
        self.output_dim = output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        if inputs.ndim not in {2, 3}:
            raise ValueError(
                "ProjectionHead expects rank-2 or rank-3 tensor inputs, "
                f"got shape={tuple(inputs.shape)}."
            )

        flattened = inputs.reshape(-1, int(original_shape[-1]))
        projected = self.network(flattened)
        return projected.reshape(*original_shape[:-1], self.output_dim)


class Round2SSLModule(nn.Module):
    """Composite Round2 SSL module containing student/teacher and heads."""

    def __init__(
        self,
        *,
        student_encoder: DieVFMModel,
        teacher_encoder: DieVFMModel,
        student_global_projector: ProjectionHead,
        teacher_global_projector: ProjectionHead,
        student_token_projector: ProjectionHead,
        teacher_token_projector: ProjectionHead,
    ) -> None:
        super().__init__()
        self.student_encoder = student_encoder
        self.teacher_encoder = teacher_encoder
        self.student_global_projector = student_global_projector
        self.teacher_global_projector = teacher_global_projector
        self.student_token_projector = student_token_projector
        self.teacher_token_projector = teacher_token_projector

    @classmethod
    def from_student_encoder(
        cls,
        *,
        student_encoder: DieVFMModel,
        global_hidden_dim: int | None,
        global_output_dim: int | None,
        global_num_layers: int,
        token_hidden_dim: int | None,
        token_output_dim: int | None,
        token_num_layers: int,
    ) -> "Round2SSLModule":
        teacher_encoder = deepcopy(student_encoder)

        student_global_projector = ProjectionHead(
            input_dim=student_encoder.embedding_dim,
            hidden_dim=global_hidden_dim,
            output_dim=global_output_dim,
            num_layers=global_num_layers,
        )
        teacher_global_projector = deepcopy(student_global_projector)

        token_input_dim = int(student_encoder.backbone.output_dim)
        student_token_projector = ProjectionHead(
            input_dim=token_input_dim,
            hidden_dim=token_hidden_dim,
            output_dim=token_output_dim,
            num_layers=token_num_layers,
        )
        teacher_token_projector = deepcopy(student_token_projector)

        module = cls(
            student_encoder=student_encoder,
            teacher_encoder=teacher_encoder,
            student_global_projector=student_global_projector,
            teacher_global_projector=teacher_global_projector,
            student_token_projector=student_token_projector,
            teacher_token_projector=teacher_token_projector,
        )
        freeze_module_parameters(module.teacher_encoder)
        freeze_module_parameters(module.teacher_global_projector)
        freeze_module_parameters(module.teacher_token_projector)
        return module


def validate_round2_train_contract(cfg: Any) -> None:
    """Validates root Round2 config fields that do not require a built model."""
    update_mode = str(cfg.train.update_mode)
    if update_mode not in VALID_UPDATE_MODES:
        raise ValueError(
            "Unsupported Round2 train.update_mode. "
            f"Got {update_mode!r}, expected one of {sorted(VALID_UPDATE_MODES)}."
        )

    precision_mode = str(cfg.train.precision_mode)
    if precision_mode not in VALID_PRECISION_MODES:
        raise ValueError(
            "Unsupported Round2 train.precision_mode. "
            f"Got {precision_mode!r}, expected one of {sorted(VALID_PRECISION_MODES)}."
        )

    ema_policy = str(cfg.round2.ema.policy)
    if ema_policy not in VALID_EMA_POLICIES:
        raise ValueError(
            "Unsupported Round2 round2.ema.policy. "
            f"Got {ema_policy!r}, expected one of {sorted(VALID_EMA_POLICIES)}."
        )

    cadence = str(cfg.round2.evaluation.cadence)
    if cadence != "end_only":
        raise ValueError(
            "Round2 v1 only supports round2.evaluation.cadence=end_only."
        )

    if update_mode != "last_n_blocks" and cfg.train.last_n_blocks is not None:
        raise ValueError(
            "train.last_n_blocks must be null unless "
            "train.update_mode=last_n_blocks."
        )
    if update_mode == "last_n_blocks" and cfg.train.last_n_blocks is None:
        raise ValueError(
            "train.last_n_blocks is required when "
            "train.update_mode=last_n_blocks."
        )

    if float(cfg.round2.loss.token_loss_weight) < 0.0:
        raise ValueError("round2.loss.token_loss_weight must be >= 0.")


def freeze_module_parameters(module: nn.Module) -> None:
    """Freezes every parameter in one module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module_parameters(module: nn.Module) -> None:
    """Unfreezes every parameter in one module."""
    for parameter in module.parameters():
        parameter.requires_grad = True


def get_backbone_blocks(backbone: nn.Module) -> list[nn.Module]:
    """Returns the trainable block list used by last_n_blocks resolution."""
    if hasattr(backbone, "model") and hasattr(backbone.model, "blocks"):
        blocks = list(backbone.model.blocks)
        if blocks:
            return blocks
    if hasattr(backbone, "blocks"):
        blocks = list(backbone.blocks)
        if blocks:
            return blocks
    return [backbone]


def resolve_total_backbone_blocks(backbone: nn.Module) -> int:
    """Returns the effective backbone block count for update-mode validation."""
    return len(get_backbone_blocks(backbone))


def apply_update_mode(
    ssl_module: Round2SSLModule,
    *,
    update_mode: str,
    last_n_blocks: int | None,
) -> UpdateModeResolution:
    """Applies one Round2 update mode to the student branch."""
    if update_mode not in VALID_UPDATE_MODES:
        raise ValueError(f"Unsupported update_mode: {update_mode!r}")

    freeze_module_parameters(ssl_module.student_encoder.backbone)
    freeze_module_parameters(ssl_module.student_encoder.pooler)
    freeze_module_parameters(ssl_module.student_global_projector)
    freeze_module_parameters(ssl_module.student_token_projector)

    trainable_module_names = [
        "student_encoder.pooler",
        "student_global_projector",
        "student_token_projector",
    ]
    trainable_block_indices: list[int] = []

    unfreeze_module_parameters(ssl_module.student_encoder.pooler)
    unfreeze_module_parameters(ssl_module.student_global_projector)
    unfreeze_module_parameters(ssl_module.student_token_projector)

    backbone_blocks = get_backbone_blocks(ssl_module.student_encoder.backbone)
    total_blocks = len(backbone_blocks)

    if update_mode == "full_backbone":
        unfreeze_module_parameters(ssl_module.student_encoder.backbone)
        trainable_module_names.insert(0, "student_encoder.backbone")
    elif update_mode == "last_n_blocks":
        if last_n_blocks is None:
            raise ValueError(
                "last_n_blocks must be provided when update_mode=last_n_blocks."
            )
        if last_n_blocks < 1 or last_n_blocks > total_blocks:
            raise ValueError(
                "train.last_n_blocks must satisfy "
                f"1 <= n <= total_backbone_blocks ({total_blocks}), got {last_n_blocks}."
            )
        start_index = total_blocks - int(last_n_blocks)
        for block_index in range(start_index, total_blocks):
            unfreeze_module_parameters(backbone_blocks[block_index])
            trainable_block_indices.append(block_index)
            trainable_module_names.append(
                f"student_encoder.backbone.blocks[{block_index}]"
            )

    return UpdateModeResolution(
        update_mode=update_mode,
        last_n_blocks=last_n_blocks,
        total_backbone_blocks=total_blocks,
        trainable_module_names=trainable_module_names,
        trainable_block_indices=trainable_block_indices,
    )


def reconcile_token_loss_trainability(
    ssl_module: Round2SSLModule,
    *,
    update_resolution: UpdateModeResolution,
    token_loss_enabled: bool,
) -> UpdateModeResolution:
    """Aligns token projector trainability with token-loss runtime intent."""
    if token_loss_enabled:
        return update_resolution

    freeze_module_parameters(ssl_module.student_token_projector)
    filtered_module_names = [
        module_name
        for module_name in update_resolution.trainable_module_names
        if module_name != "student_token_projector"
    ]
    return UpdateModeResolution(
        update_mode=update_resolution.update_mode,
        last_n_blocks=update_resolution.last_n_blocks,
        total_backbone_blocks=update_resolution.total_backbone_blocks,
        trainable_module_names=filtered_module_names,
        trainable_block_indices=list(update_resolution.trainable_block_indices),
    )


def resolve_trainable_parameters(ssl_module: Round2SSLModule) -> list[nn.Parameter]:
    """Returns the trainable student parameters only."""
    return [
        parameter
        for parameter in ssl_module.parameters()
        if parameter.requires_grad
    ]


def clone_bool_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Returns one detached boolean tensor."""
    return tensor.detach().clone().to(dtype=torch.bool)


def generate_augmented_view(
    images: torch.Tensor,
    *,
    horizontal_flip_prob: float,
    vertical_flip_prob: float,
) -> tuple[torch.Tensor, FlipMetadata]:
    """Builds one batch view using only H/V flips."""
    if images.ndim != 4:
        raise ValueError(
            f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}."
        )

    batch_size = int(images.shape[0])
    device = images.device
    horizontal = torch.rand(batch_size, device=device) < float(horizontal_flip_prob)
    vertical = torch.rand(batch_size, device=device) < float(vertical_flip_prob)

    augmented = images.clone()
    if bool(horizontal.any()):
        augmented[horizontal] = torch.flip(augmented[horizontal], dims=[3])
    if bool(vertical.any()):
        augmented[vertical] = torch.flip(augmented[vertical], dims=[2])

    return augmented, FlipMetadata(
        horizontal=clone_bool_tensor(horizontal),
        vertical=clone_bool_tensor(vertical),
    )


def canonicalize_patch_tokens(
    patch_tokens: torch.Tensor,
    *,
    patch_grid: tuple[int, int] | None,
    flip_metadata: FlipMetadata,
) -> torch.Tensor:
    """Maps patch tokens back to canonical orientation using flip metadata."""
    if patch_tokens.ndim != 3:
        raise ValueError(
            "patch_tokens must have shape [B, N, D], "
            f"got {tuple(patch_tokens.shape)}."
        )
    if patch_grid is None:
        raise ValueError("patch_grid is required for token-level alignment.")

    batch_size, num_patches, feature_dim = patch_tokens.shape
    grid_h, grid_w = patch_grid
    if grid_h * grid_w != num_patches:
        raise ValueError(
            "patch_grid does not match patch token count. "
            f"grid={patch_grid}, num_patches={num_patches}."
        )

    canonical = patch_tokens.reshape(batch_size, grid_h, grid_w, feature_dim)
    if bool(flip_metadata.horizontal.any()):
        canonical[flip_metadata.horizontal] = torch.flip(
            canonical[flip_metadata.horizontal],
            dims=[2],
        )
    if bool(flip_metadata.vertical.any()):
        canonical[flip_metadata.vertical] = torch.flip(
            canonical[flip_metadata.vertical],
            dims=[1],
        )
    return canonical.reshape(batch_size, num_patches, feature_dim)


def projected_cosine_loss(
    student_projection: torch.Tensor,
    teacher_projection: torch.Tensor,
) -> torch.Tensor:
    """Computes cosine alignment loss over the last dimension."""
    if student_projection.shape != teacher_projection.shape:
        raise ValueError(
            "Projected cosine loss requires equal-shaped inputs, got "
            f"{tuple(student_projection.shape)} vs {tuple(teacher_projection.shape)}."
        )

    student_norm = F.normalize(student_projection.float(), dim=-1)
    teacher_norm = F.normalize(teacher_projection.float(), dim=-1)
    cosine = (student_norm * teacher_norm).sum(dim=-1)
    return 1.0 - cosine.mean()


def update_teacher_ema(
    ssl_module: Round2SSLModule,
    *,
    momentum: float,
) -> None:
    """Applies one EMA step from student weights into teacher weights."""
    with torch.no_grad():
        _ema_update_module(
            teacher=ssl_module.teacher_encoder,
            student=ssl_module.student_encoder,
            momentum=momentum,
        )
        _ema_update_module(
            teacher=ssl_module.teacher_global_projector,
            student=ssl_module.student_global_projector,
            momentum=momentum,
        )
        _ema_update_module(
            teacher=ssl_module.teacher_token_projector,
            student=ssl_module.student_token_projector,
            momentum=momentum,
        )


def _ema_update_module(
    *,
    teacher: nn.Module,
    student: nn.Module,
    momentum: float,
) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters(),
        strict=True,
    ):
        teacher_parameter.data.mul_(momentum).add_(
            student_parameter.data,
            alpha=1.0 - momentum,
        )

    for teacher_buffer, student_buffer in zip(
        teacher.buffers(),
        student.buffers(),
        strict=True,
    ):
        teacher_buffer.copy_(student_buffer)


def resolve_ema_momentum(
    *,
    cfg: Any,
    epoch_index: int,
    num_epochs: int,
) -> float:
    """Resolves the runtime EMA momentum according to fixed or scheduled policy."""
    policy = str(cfg.round2.ema.policy)
    base_momentum = float(cfg.round2.ema.momentum)
    if policy == "fixed":
        return base_momentum

    final_momentum = float(cfg.round2.ema.final_momentum)
    if num_epochs <= 1:
        return final_momentum
    progress = float(epoch_index) / float(max(num_epochs - 1, 1))
    return base_momentum + progress * (final_momentum - base_momentum)


def autocast_context(
    *,
    device: torch.device,
    precision_mode: str,
):
    """Returns the precision context manager for one forward path."""
    if precision_mode == "fp32":
        return nullcontext()
    if precision_mode != "bf16":
        raise ValueError(f"Unsupported precision_mode: {precision_mode!r}")

    device_type = device.type if device.type in {"cpu", "cuda"} else "cpu"
    return torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=True,
    )
