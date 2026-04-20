"""DINOv2 backbone wrapper for die_vfm."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from die_vfm.models.backbone.base import Backbone
from die_vfm.models.outputs import BackboneOutput


_DINOV2_MODEL_NAME_MAP = {
    "vit_small": "dinov2_vits14",
    "vit_base": "dinov2_vitb14",
    "vit_large": "dinov2_vitl14",
    "vit_giant": "dinov2_vitg14",
}

_DINOV2_OUTPUT_DIM_MAP = {
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_giant": 1536,
}


class _FakeDINOv2HubModel(nn.Module):
    """Deterministic local fake for offline DINOv2 runtime smoke tests."""

    def __init__(self, *, feature_dim: int, patch_size: int) -> None:
        super().__init__()
        self._feature_dim = int(feature_dim)
        self._patch_size = int(patch_size)

    def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, _, height, width = image.shape
        num_patches = (height // self._patch_size) * (width // self._patch_size)

        patch_tokens = torch.zeros(
            (batch_size, num_patches, self._feature_dim),
            device=image.device,
            dtype=image.dtype,
        )
        cls_token = torch.zeros(
            (batch_size, self._feature_dim),
            device=image.device,
            dtype=image.dtype,
        )
        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": cls_token,
        }


class DINOv2Backbone(Backbone):
    """DINOv2 visual backbone wrapper.

    This wrapper standardizes DINOv2 outputs into the die_vfm BackboneOutput
    contract:

        - patch_tokens: Tensor[B, N, D]
        - cls_token: Tensor[B, D] or None
        - token_mask: None

    Notes:
        - This implementation assumes square image inputs.
        - DINOv2 uses patch size 14 for the official ViT checkpoints.
        - For PR-3, token_mask is always None because DINOv2 does not emit a
          variable-length token mask in the standard image path.
    """

    def __init__(
        self,
        variant: str,
        pretrained: bool = True,
        freeze: bool = False,
        return_cls_token: bool = True,
        allow_network: bool = True,
        local_repo_path: str | None = None,
        local_checkpoint_path: str | None = None,
    ) -> None:
        """Initializes the DINOv2 backbone.

        Args:
            variant: DINOv2 variant name. One of:
                "vit_small", "vit_base", "vit_large", "vit_giant".
            pretrained: Whether to load pretrained weights.
            freeze: Whether to freeze all backbone parameters.
            return_cls_token: Whether to expose CLS token in BackboneOutput.
            allow_network: Whether network-backed hub resolution is allowed.
            local_repo_path: Optional local DINOv2 repo path for architecture
                source resolution.
            local_checkpoint_path: Optional local checkpoint path for pretrained
                weight loading.

        Raises:
            ValueError: If the variant is unsupported.
        """
        super().__init__()
        _validate_variant(variant)

        self.variant = variant
        self.pretrained = pretrained
        self.freeze = freeze
        self.return_cls_token = return_cls_token
        self.allow_network = allow_network
        self.local_repo_path = _normalize_optional_path(
            field_name="local_repo_path",
            value=local_repo_path,
        )
        self.local_checkpoint_path = _normalize_optional_path(
            field_name="local_checkpoint_path",
            value=local_checkpoint_path,
        )

        self._output_dim = _DINOV2_OUTPUT_DIM_MAP[variant]
        self._patch_size = 14

        self._validate_loading_semantics()

        architecture_source = self._resolve_architecture_source()
        hub_pretrained = self.pretrained and self.local_checkpoint_path is None
        model_name = _DINOV2_MODEL_NAME_MAP[variant]
        self.model = _load_dinov2_model(
            repo_or_dir=architecture_source.repo_or_dir,
            source=architecture_source.source,
            model_name=model_name,
            pretrained=hub_pretrained,
            output_dim=self._output_dim,
            patch_size=self._patch_size,
        )

        if self.pretrained and self.local_checkpoint_path is not None:
            _load_local_checkpoint(
                model=self.model,
                checkpoint_path=self.local_checkpoint_path,
            )

        if freeze:
            self._freeze_parameters()

    @property
    def output_dim(self) -> int:
        """Returns the token feature dimension D."""
        return self._output_dim

    def forward(self, image: torch.Tensor) -> BackboneOutput:
        """Encodes an image batch into DINOv2 token representations.

        Args:
            image: Image tensor with shape [B, C, H, W].

        Returns:
            A standardized BackboneOutput.

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If image shape is invalid for DINOv2 patchification.
            RuntimeError: If the underlying DINOv2 output format is unsupported.
        """
        self.validate_image_input(image)
        self._validate_spatial_shape(image)

        features = self.model.forward_features(image)
        patch_tokens, cls_token = self._parse_forward_features(features)

        return BackboneOutput(
            patch_tokens=patch_tokens,
            cls_token=cls_token if self.return_cls_token else None,
            token_mask=None,
            feature_dim=self.output_dim,
            patch_grid=self._infer_patch_grid(image),
            metadata={
                "backbone_name": "dinov2",
                "variant": self.variant,
                "pretrained": self.pretrained,
                "freeze": self.freeze,
                "return_cls_token": self.return_cls_token,
                "allow_network": self.allow_network,
                "local_repo_path": (
                    str(self.local_repo_path)
                    if self.local_repo_path is not None
                    else None
                ),
                "local_checkpoint_path": (
                    str(self.local_checkpoint_path)
                    if self.local_checkpoint_path is not None
                    else None
                ),
                "patch_size": self._patch_size,
            },
        )

    def _freeze_parameters(self) -> None:
        """Freezes all backbone parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _validate_spatial_shape(self, image: torch.Tensor) -> None:
        """Validates image spatial dimensions for DINOv2."""
        _, _, height, width = image.shape

        if height != width:
            raise ValueError(
                "DINOv2Backbone expects square inputs for PR-3: "
                f"got height={height}, width={width}."
            )

        if height % self._patch_size != 0 or width % self._patch_size != 0:
            raise ValueError(
                "Input spatial size must be divisible by DINOv2 patch size "
                f"{self._patch_size}: got [{height}, {width}]."
            )

    def _infer_patch_grid(self, image: torch.Tensor) -> tuple[int, int]:
        """Infers patch grid from the input image shape."""
        _, _, height, width = image.shape
        return height // self._patch_size, width // self._patch_size

    def _parse_forward_features(
        self,
        features: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Parses DINOv2 forward_features output.

        The official DINOv2 implementation returns a mapping-like object with
        keys such as:
            - "x_norm_clstoken"
            - "x_norm_patchtokens"

        Returns:
            A tuple of:
                - patch_tokens: Tensor[B, N, D]
                - cls_token: Tensor[B, D] or None

        Raises:
            RuntimeError: If the output format is unsupported.
        """
        if isinstance(features, Mapping):
            patch_tokens = features.get("x_norm_patchtokens")
            cls_token = features.get("x_norm_clstoken")

            if patch_tokens is None:
                raise RuntimeError(
                    "DINOv2 forward_features output missing "
                    "'x_norm_patchtokens'."
                )

            if not isinstance(patch_tokens, torch.Tensor):
                raise RuntimeError(
                    "Expected 'x_norm_patchtokens' to be a torch.Tensor."
                )

            if cls_token is not None and not isinstance(cls_token, torch.Tensor):
                raise RuntimeError(
                    "Expected 'x_norm_clstoken' to be a torch.Tensor or None."
                )

            return patch_tokens, cls_token

        raise RuntimeError(
            "Unsupported DINOv2 forward_features output type: "
            f"{type(features).__name__}."
        )

    def _validate_loading_semantics(self) -> None:
        """Validates loading semantics and offline-ready fail-fast boundaries."""
        if not isinstance(self.allow_network, bool):
            raise TypeError("Expected allow_network to be bool.")

        if not self.pretrained and self.local_checkpoint_path is not None:
            raise ValueError(
                "DINOv2 local_checkpoint_path is only valid when "
                "model.backbone.pretrained=true."
            )

        if self.local_repo_path is not None:
            _validate_local_repo_path(self.local_repo_path)
        elif not self.allow_network:
            raise ValueError(
                "DINOv2 architecture source is unavailable: set "
                "model.backbone.local_repo_path or enable "
                "model.backbone.allow_network=true."
            )

        if self.local_checkpoint_path is not None:
            _validate_local_checkpoint_path(self.local_checkpoint_path)
        elif self.pretrained and not self.allow_network:
            raise ValueError(
                "DINOv2 pretrained offline load requires "
                "model.backbone.local_checkpoint_path when "
                "model.backbone.allow_network=false."
            )

    def _resolve_architecture_source(self) -> "_ArchitectureSource":
        """Resolves architecture source according to offline-ready policy."""
        if self.local_repo_path is not None:
            return _ArchitectureSource(
                repo_or_dir=str(self.local_repo_path),
                source="local",
            )

        return _ArchitectureSource(
            repo_or_dir="facebookresearch/dinov2",
            source="github",
        )


def _validate_variant(variant: str) -> None:
    """Validates DINOv2 variant name."""
    if variant not in _DINOV2_MODEL_NAME_MAP:
        supported = ", ".join(sorted(_DINOV2_MODEL_NAME_MAP))
        raise ValueError(
            f"Unsupported DINOv2 variant: '{variant}'. "
            f"Supported variants: {supported}."
        )


def _load_dinov2_model(
    *,
    repo_or_dir: str,
    source: str,
    model_name: str,
    pretrained: bool,
    output_dim: int,
    patch_size: int,
) -> nn.Module:
    """Loads DINOv2 hub model with an opt-in offline fake path for tests."""
    if os.getenv("DIE_VFM_DINOV2_FAKE_HUB", "").strip() == "1":
        return _FakeDINOv2HubModel(
            feature_dim=output_dim,
            patch_size=patch_size,
        )

    return torch.hub.load(
        repo_or_dir,
        model_name,
        source=source,
        pretrained=pretrained,
    )


class _ArchitectureSource:
    """Resolved architecture source for DINOv2 model construction."""

    def __init__(self, *, repo_or_dir: str, source: str) -> None:
        self.repo_or_dir = repo_or_dir
        self.source = source


def _normalize_optional_path(
    *,
    field_name: str,
    value: str | None,
) -> Path | None:
    """Normalizes an optional path-like config field."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"Expected {field_name} to be str or None, got {type(value).__name__}."
        )

    normalized = value.strip()
    if not normalized:
        return None

    return Path(normalized)


def _validate_local_repo_path(path: Path) -> None:
    """Validates local repo path contract for DINOv2 architecture loading."""
    if not path.exists():
        raise ValueError(f"Configured DINOv2 local repo does not exist: path={path}.")
    if not path.is_dir():
        raise ValueError(
            f"Configured DINOv2 local repo must point to a directory: path={path}."
        )


def _validate_local_checkpoint_path(path: Path) -> None:
    """Validates local checkpoint path contract for DINOv2 weight loading."""
    if not path.exists():
        raise ValueError(
            f"Configured DINOv2 local checkpoint does not exist: path={path}."
        )
    if not path.is_file():
        raise ValueError(
            "Configured DINOv2 local checkpoint must point to a file: "
            f"path={path}."
        )


def _load_local_checkpoint(
    *,
    model: nn.Module,
    checkpoint_path: Path,
) -> None:
    """Loads a local checkpoint into a constructed DINOv2 model."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Configured DINOv2 local checkpoint is incompatible with model "
            f"architecture: path={checkpoint_path}."
        ) from exc


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    """Extracts a model state_dict from common checkpoint payload shapes."""
    if not isinstance(payload, Mapping):
        raise ValueError("Configured DINOv2 local checkpoint has invalid payload type.")

    if _looks_like_state_dict(payload):
        return payload

    candidate_keys = (
        "state_dict",
        "model_state_dict",
        "model",
    )
    for key in candidate_keys:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping) and _looks_like_state_dict(candidate):
            return candidate

    raise ValueError(
        "Configured DINOv2 local checkpoint does not contain a valid state dict."
    )


def _looks_like_state_dict(payload: Mapping[str, Any]) -> bool:
    """Returns True when payload resembles a torch model state_dict."""
    if not payload:
        return False

    has_tensor_value = False
    for key, value in payload.items():
        if not isinstance(key, str):
            return False
        if isinstance(value, torch.Tensor):
            has_tensor_value = True
    return has_tensor_value
