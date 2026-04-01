"""Linear probe model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LinearProbeSpec:
    """Configuration for a linear probe classifier.

    Attributes:
        input_dim: Embedding feature dimension D.
        num_classes: Number of target classes C.
        bias: Whether to include a learnable bias term.
    """

    input_dim: int
    num_classes: int
    bias: bool = True

    def __post_init__(self) -> None:
        """Validates the probe specification."""
        if self.input_dim <= 0:
            raise ValueError(
                f"input_dim must be positive, got {self.input_dim}."
            )
        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1 for linear probing, "
                f"got {self.num_classes}."
            )


class LinearProbeClassifier(nn.Module):
    """A minimal linear classifier over frozen embedding features.

    This module implements the M1 linear probe for embedding artifacts:
        logits = W x + b

    The input is expected to be a 2D tensor with shape [N, D], where:
        - N is the number of samples in the batch
        - D is the embedding dimension

    The output is a 2D tensor with shape [N, C], where:
        - C is the number of classes
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
    ) -> None:
        """Initializes the linear probe classifier.

        Args:
            input_dim: Embedding feature dimension D.
            num_classes: Number of target classes C.
            bias: Whether to include a learnable bias term.
        """
        super().__init__()

        self._spec = LinearProbeSpec(
            input_dim=input_dim,
            num_classes=num_classes,
            bias=bias,
        )
        self.classifier = nn.Linear(
            in_features=self._spec.input_dim,
            out_features=self._spec.num_classes,
            bias=self._spec.bias,
        )

    @property
    def input_dim(self) -> int:
        """Returns the expected embedding dimension."""
        return self._spec.input_dim

    @property
    def num_classes(self) -> int:
        """Returns the number of output classes."""
        return self._spec.num_classes

    @property
    def use_bias(self) -> bool:
        """Returns whether the classifier includes bias parameters."""
        return self._spec.bias

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Computes classification logits from embeddings.

        Args:
            embeddings: Input embedding tensor with shape [N, D].

        Returns:
            Logits tensor with shape [N, C].

        Raises:
            ValueError: If the input tensor shape is invalid.
        """
        self._validate_embeddings(embeddings)
        return self.classifier(embeddings)

    def extra_repr(self) -> str:
        """Returns a concise module representation."""
        return (
            f"input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"bias={self.use_bias}"
        )

    def _validate_embeddings(self, embeddings: torch.Tensor) -> None:
        """Validates input embeddings for the linear probe."""
        if embeddings.ndim != 2:
            raise ValueError(
                "LinearProbeClassifier expects embeddings with shape [N, D], "
                f"got ndim={embeddings.ndim}."
            )

        actual_dim = int(embeddings.shape[1])
        if actual_dim != self.input_dim:
            raise ValueError(
                "Embedding dimension mismatch for LinearProbeClassifier. "
                f"Expected D={self.input_dim}, got D={actual_dim}."
            )


def build_linear_probe(
    input_dim: int,
    num_classes: int,
    bias: bool = True,
) -> LinearProbeClassifier:
    """Builds a linear probe classifier.

    Args:
        input_dim: Embedding feature dimension D.
        num_classes: Number of target classes C.
        bias: Whether to include a learnable bias term.

    Returns:
        A LinearProbeClassifier instance.
    """
    return LinearProbeClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        bias=bias,
    )