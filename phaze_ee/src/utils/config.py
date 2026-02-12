# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Configuration dataclass for PHAZE-EE.

This module defines the Config dataclass that holds all configuration parameters
for early exit model training, evaluation, and benchmarking.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration dataclass for PHAZE-EE early exit training and evaluation.

    This class holds all configuration parameters that can be set in project-specific
    config files. It mirrors BEAD's Config pattern but is adapted for early exit workflows.
    """

    # Model configuration
    model_name: str = "ParticleNet"
    num_classes: int = 10
    num_exit_points: int = 3
    
    # Training configuration
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 512
    optimizer: str = "ranger"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Early exit specific
    exit_loss_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])
    exit_threshold: float = 0.9
    exit_strategy: str = "confidence"
    
    # Data configuration
    data_config: str = ""
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    
    # Distributed training
    use_ddp: bool = False
    use_amp: bool = True
    is_ddp_active: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Logging and checkpointing
    log_interval: int = 10
    val_interval: int = 1
    save_interval: int = 10
    num_workers: int = 4
    
    # Benchmarking
    profile_flops: bool = True
    num_profile_samples: int = 1000
    
    # Plotting
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    plot_dpi: int = 300
    
    # Additional runtime flags
    skip_to_roc: bool = False
    file_type: str = "h5"
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["PHAZE-EE Configuration:"]
        lines.append(f"  Model: {self.model_name} with {self.num_exit_points} exit points")
        lines.append(f"  Training: {self.epochs} epochs, lr={self.lr}, batch_size={self.batch_size}")
        lines.append(f"  Early Exit: {self.exit_strategy} strategy, threshold={self.exit_threshold}")
        lines.append(f"  Distributed: DDP={'enabled' if self.use_ddp else 'disabled'}, AMP={'enabled' if self.use_amp else 'disabled'}")
        return "\n".join(lines)
