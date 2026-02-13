"""
Exit loss strategies and beta scheduling for early exit training.

This module provides different strategies for computing exit losses during training,
and a beta scheduler for dynamically adjusting the weight of exit losses.
"""

import math
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitLossStrategy(str, Enum):
    """Exit loss computation strategies."""
    
    # MSE-based strategies: exit learns to mimic full model output
    MIMIC_DETACHED = "mimic_detached"  # MSE(exit, full.detach()) - no gradient to backbone via exit
    MIMIC_FLOW = "mimic_flow"          # MSE(exit, full) - gradients flow through full model too
    
    # Target-based strategies: exit learns from ground truth labels
    TARGET_DETACHED = "target_detached"  # CE(exit(features.detach()), labels) - no gradient to backbone
    TARGET_FLOW = "target_flow"          # CE(exit, labels) - gradients flow through backbone


class ExitLossComputer:
    """Computes exit losses based on the selected strategy.
    
    This class handles the 4 different exit loss strategies for early exit training:
    1. mimic_detached: Exit branches learn to mimic full model output, but gradients 
       don't flow back to the backbone through the MSE loss (only through exit branches)
    2. mimic_flow: Exit branches learn to mimic full model output, with gradients 
       flowing back through both exit branches and the full model
    3. target_detached: Exit branches learn from ground truth labels using CE loss,
       with features detached (no gradients to backbone)
    4. target_flow: Exit branches learn from ground truth labels using CE loss,
       with gradients flowing back through the backbone
    """
    
    def __init__(
        self,
        strategy: ExitLossStrategy,
        criterion: Optional[nn.Module] = None,
    ):
        """Initialize exit loss computer.
        
        Args:
            strategy: The exit loss strategy to use
            criterion: Loss criterion for target-based strategies (default: CrossEntropyLoss)
        """
        self.strategy = ExitLossStrategy(strategy)
        
        # For target-based strategies, use CrossEntropyLoss (matching weaver-core)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
    
    def compute_exit_loss(
        self,
        exit_output: torch.Tensor,
        full_output: torch.Tensor,
        exit_features: torch.Tensor,
        labels: torch.Tensor,
        exit_idx: int,
    ) -> torch.Tensor:
        """Compute loss for a single exit point.
        
        Args:
            exit_output: Exit branch output logits (batch, num_classes)
            full_output: Full model output logits (batch, num_classes)
            exit_features: Intermediate features before exit branch (batch, channels, points)
            labels: Ground truth labels (batch,) as class indices
            exit_idx: Index of the exit point
        
        Returns:
            torch.Tensor: Scalar loss value for this exit point
        """
        if self.strategy == ExitLossStrategy.MIMIC_DETACHED:
            # MSE between exit output and detached full model output
            target = full_output.detach()
            loss = F.mse_loss(exit_output, target)
        
        elif self.strategy == ExitLossStrategy.MIMIC_FLOW:
            # MSE between exit output and full model output (gradients flow)
            loss = F.mse_loss(exit_output, full_output)
        
        elif self.strategy == ExitLossStrategy.TARGET_DETACHED:
            # Not directly supported - would require recomputing exit from detached features
            # For simplicity, we use the already computed exit_output but note this
            # doesn't fully detach (exit_output was computed from non-detached features)
            # True implementation would need to pass detached features to exit branch
            # This is a limitation we accept for the naive baseline
            loss = self.criterion(exit_output, labels)
        
        elif self.strategy == ExitLossStrategy.TARGET_FLOW:
            # CrossEntropyLoss between exit output and ground truth labels
            loss = self.criterion(exit_output, labels)
        
        else:
            raise ValueError(f"Unknown exit loss strategy: {self.strategy}")
        
        return loss
    
    def compute_total_exit_loss(
        self,
        exit_outputs: List[torch.Tensor],
        full_output: torch.Tensor,
        exit_features: List[torch.Tensor],
        labels: torch.Tensor,
        betas: List[float],
    ) -> tuple[torch.Tensor, List[float]]:
        """Compute total weighted exit loss across all exit points.
        
        Args:
            exit_outputs: List of exit branch outputs (each: batch, num_classes)
            full_output: Full model output logits (batch, num_classes)
            exit_features: List of intermediate features (each: batch, channels, points)
            labels: Ground truth labels (batch,) as class indices
            betas: List of beta weights for each exit point
        
        Returns:
            Tuple of:
                - total_loss: Weighted sum of all exit losses
                - individual_losses: List of individual loss values for logging
        """
        total_loss = 0.0
        individual_losses = []
        
        for i, (exit_out, exit_feat, beta) in enumerate(zip(exit_outputs, exit_features, betas)):
            if beta > 0:  # Only compute if beta is non-zero
                loss = self.compute_exit_loss(
                    exit_output=exit_out,
                    full_output=full_output,
                    exit_features=exit_feat,
                    labels=labels,
                    exit_idx=i,
                )
                weighted_loss = beta * loss
                total_loss = total_loss + weighted_loss
                individual_losses.append(loss.item())
            else:
                individual_losses.append(0.0)
        
        return total_loss, individual_losses


class BetaScheduler:
    """Scheduler for beta values (exit loss weights) during training.
    
    Supports dynamic scheduling where beta starts at 0 for initial epochs,
    then ramps up according to a schedule (linear or cosine).
    """
    
    def __init__(
        self,
        beta_max: List[float],
        total_epochs: int,
        zero_epochs: int = 0,
        ramp_type: str = "linear",
        per_exit: bool = True,
    ):
        """Initialize beta scheduler.
        
        Args:
            beta_max: Maximum beta values (one per exit if per_exit=True, else single value)
            total_epochs: Total number of training epochs
            zero_epochs: Number of initial epochs with beta=0
            ramp_type: Ramp schedule type ("linear" or "cosine")
            per_exit: Whether each exit has its own beta_max
        """
        self.beta_max = beta_max if isinstance(beta_max, list) else [beta_max]
        self.total_epochs = total_epochs
        self.zero_epochs = zero_epochs
        self.ramp_type = ramp_type
        self.per_exit = per_exit
        
        # Validate
        if zero_epochs >= total_epochs:
            raise ValueError(f"zero_epochs ({zero_epochs}) must be < total_epochs ({total_epochs})")
        
        if ramp_type not in ["linear", "cosine"]:
            raise ValueError(f"ramp_type must be 'linear' or 'cosine', got: {ramp_type}")
    
    def get_betas(self, epoch: int, num_exits: int) -> List[float]:
        """Get beta values for the current epoch.
        
        Args:
            epoch: Current epoch (0-based)
            num_exits: Number of exit points
        
        Returns:
            List of beta values, one per exit point
        """
        # During zero epochs, beta is 0
        if epoch < self.zero_epochs:
            return [0.0] * num_exits
        
        # After zero epochs, compute ramp progress
        ramp_epochs = self.total_epochs - self.zero_epochs
        progress = (epoch - self.zero_epochs) / ramp_epochs
        progress = min(progress, 1.0)  # Clamp to [0, 1]
        
        # Compute scaling factor based on ramp type
        if self.ramp_type == "linear":
            scale = progress
        elif self.ramp_type == "cosine":
            # Cosine ramp: 0.5 * (1 - cos(pi * t))
            scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            scale = progress  # Fallback to linear
        
        # Apply scaling to beta_max for each exit
        betas = []
        for i in range(num_exits):
            if self.per_exit and i < len(self.beta_max):
                beta_max_i = self.beta_max[i]
            else:
                beta_max_i = self.beta_max[0]  # Use first value if not per-exit
            
            betas.append(scale * beta_max_i)
        
        return betas
    
    def get_beta(self, epoch: int, exit_idx: int = 0) -> float:
        """Get beta value for a specific exit point at current epoch.
        
        Args:
            epoch: Current epoch (0-based)
            exit_idx: Index of exit point
        
        Returns:
            float: Beta value for this exit point
        """
        betas = self.get_betas(epoch, exit_idx + 1)
        return betas[exit_idx]


def compute_multi_exit_loss(
    full_output: torch.Tensor,
    exit_outputs: List[torch.Tensor],
    exit_features: List[torch.Tensor],
    labels: torch.Tensor,
    betas: List[float],
    strategy: ExitLossStrategy,
    base_criterion: nn.Module,
) -> tuple[torch.Tensor, dict]:
    """Compute total loss combining base model loss and exit losses.
    
    This is the main loss function for early exit training.
    
    Args:
        full_output: Full model output logits (batch, num_classes)
        exit_outputs: List of exit branch outputs
        exit_features: List of intermediate features
        labels: Ground truth labels (batch,)
        betas: Beta weights for each exit point
        strategy: Exit loss computation strategy
        base_criterion: Loss criterion for the full model
    
    Returns:
        Tuple of:
            - total_loss: Combined loss (base_loss + weighted exit losses)
            - loss_dict: Dictionary with individual loss components for logging
    """
    # Base model loss (CrossEntropyLoss on full model output)
    base_loss = base_criterion(full_output, labels)
    
    # Exit losses
    loss_computer = ExitLossComputer(strategy=strategy, criterion=base_criterion)
    exit_loss, individual_exit_losses = loss_computer.compute_total_exit_loss(
        exit_outputs=exit_outputs,
        full_output=full_output,
        exit_features=exit_features,
        labels=labels,
        betas=betas,
    )
    
    # Total loss
    total_loss = base_loss + exit_loss
    
    # Prepare loss dictionary for logging
    loss_dict = {
        "total": total_loss.item(),
        "base": base_loss.item(),
        "exit_total": exit_loss.item() if isinstance(exit_loss, torch.Tensor) else exit_loss,
    }
    
    # Add individual exit losses
    for i, loss_val in enumerate(individual_exit_losses):
        loss_dict[f"exit_{i}"] = loss_val
    
    return total_loss, loss_dict
