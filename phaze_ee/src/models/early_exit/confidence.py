# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Early exit decision strategies.

This module implements different strategies for deciding when to exit early
based on intermediate predictions.
"""

import torch
import torch.nn.functional as F


def confidence_based_exit(
    logits: torch.Tensor,
    threshold: float = 0.9,
) -> tuple:
    """Decide whether to exit based on prediction confidence.

    Args:
        logits: Model output logits (batch, num_classes)
        threshold: Confidence threshold for early exit

    Returns:
        tuple: (should_exit, confidences)
            - should_exit: Boolean tensor indicating which samples should exit
            - confidences: Confidence values (max softmax probability)
    """
    probs = F.softmax(logits, dim=-1)
    confidences, _ = torch.max(probs, dim=-1)
    should_exit = confidences >= threshold
    return should_exit, confidences


def entropy_based_exit(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> tuple:
    """Decide whether to exit based on prediction entropy.

    Low entropy indicates high confidence, suggesting early exit is safe.

    Args:
        logits: Model output logits (batch, num_classes)
        threshold: Entropy threshold for early exit (lower = more confident)

    Returns:
        tuple: (should_exit, entropies)
            - should_exit: Boolean tensor indicating which samples should exit
            - entropies: Entropy values
    """
    probs = F.softmax(logits, dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    should_exit = entropies <= threshold
    return should_exit, entropies


class LearnedExitModule(torch.nn.Module):
    """Learned exit decision module.

    Uses a small learned network to decide whether to exit early
    based on intermediate features.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        """Initialize learned exit module.

        Args:
            feature_dim: Dimension of intermediate features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.decision_network = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> tuple:
        """Compute exit decision.

        Args:
            features: Intermediate features

        Returns:
            tuple: (should_exit, exit_scores)
                - should_exit: Boolean tensor
                - exit_scores: Continuous exit scores [0, 1]
        """
        exit_scores = self.decision_network(features).squeeze(-1)
        should_exit = exit_scores >= 0.5
        return should_exit, exit_scores
