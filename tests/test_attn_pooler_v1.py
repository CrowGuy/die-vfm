"""Unit tests for AttnPoolerV1."""

from __future__ import annotations

import pytest
import torch

from die_vfm.models.pooler.attn_pooler_v1 import AttnPoolerV1


def test_attn_pooler_output_shape_default() -> None:
    """Returns [B, D] embedding and [B, N] token weights by default."""
    batch_size = 2
    num_tokens = 5
    input_dim = 8

    pooler = AttnPoolerV1(
        input_dim=input_dim,
        hidden_dim=16,
    )

    patch_tokens = torch.randn(batch_size, num_tokens, input_dim)
    output = pooler(patch_tokens)

    assert output.embedding.shape == (batch_size, input_dim)
    assert output.token_weights is not None
    assert output.token_weights.shape == (batch_size, num_tokens)
    assert output.metadata["pooler_name"] == "attn_pooler_v1"


def test_attn_pooler_output_shape_with_output_projection() -> None:
    """Projects pooled embedding to output_dim when requested."""
    batch_size = 3
    num_tokens = 7
    input_dim = 8
    output_dim = 4

    pooler = AttnPoolerV1(
        input_dim=input_dim,
        hidden_dim=12,
        output_dim=output_dim,
    )

    patch_tokens = torch.randn(batch_size, num_tokens, input_dim)
    output = pooler(patch_tokens)

    assert output.embedding.shape == (batch_size, output_dim)
    assert output.token_weights is not None
    assert output.token_weights.shape == (batch_size, num_tokens)


def test_attn_pooler_weights_sum_to_one_without_mask() -> None:
    """Attention weights should sum to 1 across tokens."""
    batch_size = 4
    num_tokens = 6
    input_dim = 10

    pooler = AttnPoolerV1(
        input_dim=input_dim,
        hidden_dim=20,
    )

    patch_tokens = torch.randn(batch_size, num_tokens, input_dim)
    output = pooler(patch_tokens)

    assert output.token_weights is not None
    weight_sums = output.token_weights.sum(dim=1)
    expected = torch.ones(batch_size)

    assert torch.allclose(weight_sums, expected, atol=1e-6)


def test_attn_pooler_respects_mask_zeroing_invalid_tokens() -> None:
    """Masked-out tokens should receive zero attention weight."""
    batch_size = 2
    num_tokens = 4
    input_dim = 6

    pooler = AttnPoolerV1(
        input_dim=input_dim,
        hidden_dim=8,
    )

    patch_tokens = torch.randn(batch_size, num_tokens, input_dim)
    token_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )

    output = pooler(patch_tokens, token_mask=token_mask)

    assert output.token_weights is not None

    masked_weights = output.token_weights[~token_mask]
    assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-7)

    valid_weight_sums = output.token_weights.sum(dim=1)
    assert torch.allclose(valid_weight_sums, torch.ones(batch_size), atol=1e-6)


def test_attn_pooler_raises_when_all_tokens_masked() -> None:
    """Raises if any sample has zero valid tokens after masking."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
    )

    patch_tokens = torch.randn(2, 4, 8)
    token_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="at least one valid token"):
        pooler(patch_tokens, token_mask=token_mask)


def test_attn_pooler_raises_on_feature_dim_mismatch() -> None:
    """Raises when patch token feature dim does not match input_dim."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
    )

    patch_tokens = torch.randn(2, 5, 7)

    with pytest.raises(ValueError, match="feature dimension mismatch"):
        pooler(patch_tokens)


def test_attn_pooler_cls_query_requires_cls_token() -> None:
    """Raises when cls-token query mode is enabled but cls_token is missing."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
        use_cls_token_as_query=True,
    )

    patch_tokens = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="cls_token must be provided"):
        pooler(patch_tokens)


def test_attn_pooler_cls_query_runs_with_cls_token() -> None:
    """Runs successfully when cls-token query mode is enabled and cls_token exists."""
    batch_size = 2
    num_tokens = 5
    input_dim = 8

    pooler = AttnPoolerV1(
        input_dim=input_dim,
        hidden_dim=16,
        use_cls_token_as_query=True,
    )

    patch_tokens = torch.randn(batch_size, num_tokens, input_dim)
    cls_token = torch.randn(batch_size, input_dim)

    output = pooler(patch_tokens, cls_token=cls_token)

    assert output.embedding.shape == (batch_size, input_dim)
    assert output.token_weights is not None
    assert output.token_weights.shape == (batch_size, num_tokens)


def test_attn_pooler_can_hide_token_weights() -> None:
    """Returns token_weights=None when return_token_weights is disabled."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
        return_token_weights=False,
    )

    patch_tokens = torch.randn(2, 5, 8)
    output = pooler(patch_tokens)

    assert output.embedding.shape == (2, 8)
    assert output.token_weights is None


def test_attn_pooler_l2_norm_normalizes_output() -> None:
    """Applies L2 normalization to the final embedding when enabled."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
        l2_norm=True,
    )

    patch_tokens = torch.randn(3, 5, 8)
    output = pooler(patch_tokens)

    norms = torch.norm(output.embedding, p=2, dim=1)
    expected = torch.ones_like(norms)

    assert torch.allclose(norms, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_dim": 0, "hidden_dim": 16}, "input_dim must be > 0"),
        ({"input_dim": 8, "hidden_dim": 0}, "hidden_dim must be > 0"),
        ({"input_dim": 8, "hidden_dim": 16, "output_dim": 0}, "output_dim must be > 0"),
        ({"input_dim": 8, "hidden_dim": 16, "dropout": -0.1}, "dropout must be in"),
        ({"input_dim": 8, "hidden_dim": 16, "dropout": 1.0}, "dropout must be in"),
    ],
)
def test_attn_pooler_invalid_init_args_raise(
    kwargs: dict[str, int | float],
    match: str,
) -> None:
    """Constructor should fail fast on invalid arguments."""
    with pytest.raises(ValueError, match=match):
        AttnPoolerV1(**kwargs)


def test_attn_pooler_accepts_non_bool_mask() -> None:
    """Non-zero mask values should be treated as valid tokens."""
    pooler = AttnPoolerV1(
        input_dim=8,
        hidden_dim=16,
    )

    patch_tokens = torch.randn(2, 4, 8)
    token_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [2, 0, 3, 0],
        ],
        dtype=torch.int64,
    )

    output = pooler(patch_tokens, token_mask=token_mask)

    assert output.token_weights is not None
    invalid_positions = token_mask == 0
    assert torch.allclose(
        output.token_weights[invalid_positions],
        torch.zeros_like(output.token_weights[invalid_positions]),
        atol=1e-7,
    )
    assert torch.allclose(
        output.token_weights.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )