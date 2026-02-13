# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Early exit branch architectures.

This module defines the exit branch modules that attach to intermediate layers
of the base model to enable early predictions.
"""

import torch
import torch.nn as nn


class MLPExitBranch(nn.Module):
    """MLP-based exit branch for early predictions.

    A simple MLP classifier that can be attached to intermediate representations
    to enable early exit predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """Initialize MLP exit branch.

        Args:
            input_dim: Dimension of input features from intermediate layer
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through exit branch.

        Args:
            x: Input features from intermediate layer

        Returns:
            torch.Tensor: Class logits
        """
        return self.branch(x)


class ConvExitBranch(nn.Module):
    """Convolutional exit branch for spatial features.

    For models with spatial features (e.g., CNNs), this provides a
    convolutional exit branch with global pooling.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        """Initialize convolutional exit branch.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.conv = nn.Conv1d(input_channels, input_channels // 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_channels // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional exit branch.

        Args:
            x: Input features with shape (batch, channels, length)

        Returns:
            torch.Tensor: Class logits
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LinearExitBranch(nn.Module):
    """Simple linear exit branch for spatial features.
    
    The simplest naive exit branch: applies global pooling (matching ParticleNet's 
    pooling strategy) and a single linear layer. No conv layers or hidden dimensions.
    This is the baseline for benchmarking early exit strategies.
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        use_counts: bool = True,
    ):
        """Initialize linear exit branch.
        
        Args:
            input_channels: Number of input channels from EdgeConv block
            num_classes: Number of output classes
            use_counts: Whether to use particle counts for mean pooling (matching ParticleNet)
        """
        super().__init__()
        self.use_counts = use_counts
        self.fc = nn.Linear(input_channels, num_classes)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through linear exit branch.
        
        Args:
            x: Input features with shape (batch, channels, num_points)
            mask: Optional mask with shape (batch, 1, num_points)
        
        Returns:
            torch.Tensor: Class logits with shape (batch, num_classes)
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask
        
        # Global pooling matching ParticleNet's strategy
        if self.use_counts and mask is not None:
            # Mean pooling: sum over points / counts
            counts = mask.sum(dim=-1, keepdim=False)  # (batch, 1)
            x = x.sum(dim=-1)  # (batch, channels)
            x = x / counts.clamp(min=1)  # Avoid division by zero
        else:
            # Simple mean pooling
            x = x.mean(dim=-1)  # (batch, channels)
        
        # Linear classifier
        x = self.fc(x)  # (batch, num_classes)
        return x
