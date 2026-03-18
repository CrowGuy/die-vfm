"""Tests for die_vfm.models.pooler.mean_pooler."""

from __future__ import annotations

import pytest
import torch

from die_vfm.models.outputs import PoolerOutput
from die_vfm.models.pooler.mean_pooler import MeanPooler


def test_mean_pooler_output_dim() -> None:
    """Returns the configured output embedding dimension."""
    pooler = MeanPooler(input_dim=192, l2_norm=False)

    assert pooler.output_dim == 192


def test_mean_pooler_forward_without_mask() -> None:
    """Computes unmasked mean pooling over patch tokens."""
    pooler = MeanPooler(input_dim=2, l2_norm=False)
    patch_tokens = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]],
            [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        ]
    )

    output = pooler(patch_tokens)

    expected = torch.tensor(
        [
            [3.0, 5.0],
            [4.0, 6.0],
        ]
    )

    assert isinstance(output, PoolerOutput)
    assert output.embedding.shape == (2, 2)
    assert torch.allclose(output.embedding, expected)
    assert output.token_weights is None
    assert output.metadata["pooler_name"] == "mean"
    assert output.metadata["input_dim"] == 2
    assert output.metadata["l2_norm"] is False


def test_mean_pooler_forward_with_mask() -> None:
    """Computes masked mean pooling over valid tokens only."""
    pooler = MeanPooler(input_dim=2, l2_norm=False)
    patch_tokens = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
        ]
    )
    token_mask = torch.tensor(
        [
            [1, 1, 0],
            [0, 1, 1],
        ]
    )

    output = pooler(patch_tokens, token_mask=token_mask)

    expected = torch.tensor(
        [
            [2.0, 2.0],
            [5.0, 5.0],
        ]
    )

    assert torch.allclose(output.embedding, expected)


def test_mean_pooler_ignores_cls_token() -> None:
    """Ignores cls_token and pools only patch tokens."""
    pooler = MeanPooler(input_dim=2, l2_norm=False)
    patch_tokens = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0]]]
    )
    cls_token = torch.tensor([[100.0, 100.0]])

    output = pooler(patch_tokens, cls_token=cls_token)

    expected = torch.tensor([[2.0, 4.0]])
    assert torch.allclose(output.embedding, expected)


def test_mean_pooler_applies_l2_norm() -> None:
    """Applies L2 normalization when enabled."""
    pooler = MeanPooler(input_dim=2, l2_norm=True)
    patch_tokens = torch.tensor(
        [[[3.0, 0.0], [3.0, 0.0]]]
    )

    output = pooler(patch_tokens)

    expected = torch.tensor([[1.0, 0.0]])
    norms = torch.linalg.norm(output.embedding, dim=-1)

    assert torch.allclose(output.embedding, expected)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_mean_pooler_raises_for_invalid_input_dim() -> None:
    """Raises when input_dim is invalid."""
    with pytest.raises(ValueError, match="input_dim must be > 0"):
        MeanPooler(input_dim=0, l2_norm=False)


def test_mean_pooler_raises_for_non_tensor_patch_tokens() -> None:
    """Raises when patch_tokens is not a tensor."""
    pooler = MeanPooler(input_dim=2)

    with pytest.raises(TypeError, match="patch_tokens"):
        pooler("not a tensor")  # type: ignore[arg-type]


def test_mean_pooler_raises_for_non_3d_patch_tokens() -> None:
    """Raises when patch_tokens rank is not 3."""
    pooler = MeanPooler(input_dim=2)
    patch_tokens = torch.randn(2, 2)

    with pytest.raises(ValueError, match="patch_tokens shape"):
        pooler(patch_tokens)


def test_mean_pooler_raises_for_feature_dim_mismatch() -> None:
    """Raises when patch token feature dim does not match input_dim."""
    pooler = MeanPooler(input_dim=4, l2_norm=False)
    patch_tokens = torch.randn(2, 3, 2)

    with pytest.raises(ValueError, match="feature dimension mismatch"):
        pooler(patch_tokens)


def test_mean_pooler_raises_for_invalid_cls_token_shape() -> None:
    """Raises when cls_token shape is invalid."""
    pooler = MeanPooler(input_dim=2)
    patch_tokens = torch.randn(2, 3, 2)
    cls_token = torch.randn(2, 3, 2)

    with pytest.raises(ValueError, match="cls_token shape"):
        pooler(patch_tokens, cls_token=cls_token)


def test_mean_pooler_raises_for_invalid_token_mask_shape() -> None:
    """Raises when token_mask shape is invalid."""
    pooler = MeanPooler(input_dim=2)
    patch_tokens = torch.randn(2, 3, 2)
    token_mask = torch.ones(2, 3, 1)

    with pytest.raises(ValueError, match="token_mask shape"):
        pooler(patch_tokens, token_mask=token_mask)


def test_mean_pooler_raises_for_token_mask_token_dim_mismatch() -> None:
    """Raises when token_mask token dimension does not match patch tokens."""
    pooler = MeanPooler(input_dim=2)
    patch_tokens = torch.randn(2, 3, 2)
    token_mask = torch.ones(2, 4)

    with pytest.raises(ValueError, match="token dimension mismatch"):
        pooler(patch_tokens, token_mask=token_mask)


def test_mean_pooler_raises_when_mask_has_no_valid_tokens() -> None:
    """Raises when any sample has zero valid tokens in token_mask."""
    pooler = MeanPooler(input_dim=2, l2_norm=False)
    patch_tokens = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
        ]
    )
    token_mask = torch.tensor(
        [
            [1, 1],
            [0, 0],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid token"):
        pooler(patch_tokens, token_mask=token_mask)