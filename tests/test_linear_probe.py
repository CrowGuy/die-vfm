"""Tests for die_vfm.evaluator.linear_probe."""

from __future__ import annotations

import pytest
import torch

from die_vfm.evaluator.linear_probe import (
    LinearProbeClassifier,
    LinearProbeSpec,
    build_linear_probe,
)


def test_linear_probe_spec_accepts_valid_arguments() -> None:
    """Builds a valid probe spec."""
    spec = LinearProbeSpec(
        input_dim=128,
        num_classes=5,
        bias=False,
    )

    assert spec.input_dim == 128
    assert spec.num_classes == 5
    assert spec.bias is False


def test_linear_probe_spec_raises_for_non_positive_input_dim() -> None:
    """Rejects non-positive input dimensions."""
    with pytest.raises(ValueError, match="input_dim must be positive"):
        LinearProbeSpec(
            input_dim=0,
            num_classes=2,
        )


def test_linear_probe_spec_raises_for_invalid_num_classes() -> None:
    """Rejects class counts smaller than 2."""
    with pytest.raises(ValueError, match="greater than 1"):
        LinearProbeSpec(
            input_dim=16,
            num_classes=1,
        )


def test_linear_probe_classifier_initializes_with_bias() -> None:
    """Creates a classifier with a learnable bias term."""
    model = LinearProbeClassifier(
        input_dim=32,
        num_classes=4,
        bias=True,
    )

    assert model.input_dim == 32
    assert model.num_classes == 4
    assert model.use_bias is True

    assert model.classifier.in_features == 32
    assert model.classifier.out_features == 4
    assert model.classifier.bias is not None


def test_linear_probe_classifier_initializes_without_bias() -> None:
    """Creates a classifier without a bias term."""
    model = LinearProbeClassifier(
        input_dim=64,
        num_classes=3,
        bias=False,
    )

    assert model.input_dim == 64
    assert model.num_classes == 3
    assert model.use_bias is False
    assert model.classifier.bias is None


def test_linear_probe_classifier_forward_returns_logits_with_expected_shape() -> None:
    """Maps [N, D] embeddings to [N, C] logits."""
    model = LinearProbeClassifier(
        input_dim=6,
        num_classes=3,
    )
    embeddings = torch.randn(5, 6)

    logits = model(embeddings)

    assert logits.shape == (5, 3)
    assert logits.dtype == embeddings.dtype


def test_linear_probe_classifier_forward_preserves_batch_size_for_single_sample() -> None:
    """Handles a batch of size 1."""
    model = LinearProbeClassifier(
        input_dim=10,
        num_classes=2,
    )
    embeddings = torch.randn(1, 10)

    logits = model(embeddings)

    assert logits.shape == (1, 2)


def test_linear_probe_classifier_raises_for_invalid_embedding_rank() -> None:
    """Rejects inputs that are not rank-2 tensors."""
    model = LinearProbeClassifier(
        input_dim=8,
        num_classes=2,
    )
    embeddings = torch.randn(4, 8, 1)

    with pytest.raises(ValueError, match="shape \\[N, D\\]"):
        model(embeddings)


def test_linear_probe_classifier_raises_for_embedding_dim_mismatch() -> None:
    """Rejects inputs whose feature dimension does not match input_dim."""
    model = LinearProbeClassifier(
        input_dim=8,
        num_classes=2,
    )
    embeddings = torch.randn(4, 7)

    with pytest.raises(ValueError, match="Expected D=8, got D=7"):
        model(embeddings)


def test_build_linear_probe_returns_classifier_instance() -> None:
    """The builder returns a LinearProbeClassifier."""
    model = build_linear_probe(
        input_dim=12,
        num_classes=5,
        bias=False,
    )

    assert isinstance(model, LinearProbeClassifier)
    assert model.input_dim == 12
    assert model.num_classes == 5
    assert model.use_bias is False


def test_linear_probe_classifier_extra_repr_includes_core_fields() -> None:
    """extra_repr exposes key model settings."""
    model = LinearProbeClassifier(
        input_dim=20,
        num_classes=7,
        bias=True,
    )

    extra = model.extra_repr()

    assert "input_dim=20" in extra
    assert "num_classes=7" in extra
    assert "bias=True" in extra


def test_linear_probe_classifier_parameters_are_trainable() -> None:
    """Probe parameters should require gradients."""
    model = LinearProbeClassifier(
        input_dim=4,
        num_classes=2,
    )

    parameters = list(model.parameters())

    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)


def test_linear_probe_classifier_backward_pass_computes_gradients() -> None:
    """A simple backward pass should populate parameter gradients."""
    torch.manual_seed(0)

    model = LinearProbeClassifier(
        input_dim=5,
        num_classes=3,
    )
    embeddings = torch.randn(6, 5)
    labels = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)

    logits = model(embeddings)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.shape == model.classifier.weight.shape

    if model.classifier.bias is not None:
        assert model.classifier.bias.grad is not None
        assert model.classifier.bias.grad.shape == model.classifier.bias.shape