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
    network_config: str = ""  # Path to network config .py file (for weaver compatibility)
    
    # ParticleNet architecture (matching weaver-core example)
    conv_params: List = field(default_factory=lambda: [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
    ])
    fc_params: List = field(default_factory=lambda: [(256, 0.1)])
    use_fusion: bool = False
    use_fts_bn: bool = True
    use_counts: bool = True
    
    # Training configuration
    epochs: int = 50  # Changed to 50 to match weaver-core JetClass example
    lr: float = 1e-2  # Changed to 1e-2 for ParticleNet (weaver default)
    batch_size: int = 512
    optimizer: str = "ranger"
    scheduler: str = "flat+decay"  # Changed to match weaver-core default
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    samples_per_epoch: Optional[int] = None  # Limit samples per epoch (for large datasets)
    
    # Early exit loss strategies
    exit_loss_strategy: str = "mimic_detached"  # mimic_detached | mimic_flow | target_detached | target_flow
    
    # Beta scheduling for exit losses
    beta_max: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])  # Max beta per exit
    beta_zero_epochs: int = 10  # Number of initial epochs with beta=0
    beta_ramp_type: str = "linear"  # linear | cosine
    
    # Legacy exit config (for backward compatibility)
    exit_loss_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 1.0])
    exit_threshold: float = 0.9
    exit_strategy: str = "confidence"
    
    # Data configuration
    data_config: str = ""  # Path to YAML data config file (weaver format)
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
        lines.append(f"  Architecture: {len(self.conv_params)} EdgeConv blocks, fusion={self.use_fusion}")
        lines.append(f"  Training: {self.epochs} epochs, lr={self.lr}, batch_size={self.batch_size}")
        lines.append(f"  Exit Loss: {self.exit_loss_strategy}")
        lines.append(f"  Beta Schedule: max={self.beta_max}, zero_epochs={self.beta_zero_epochs}, ramp={self.beta_ramp_type}")
        lines.append(f"  Distributed: DDP={'enabled' if self.use_ddp else 'disabled'}, AMP={'enabled' if self.use_amp else 'disabled'}")
        return "\n".join(lines)
