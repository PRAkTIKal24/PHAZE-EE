"""
ParticleNet with Early Exit (ParticleNetEE).

This module wraps the ParticleNet model and adds early exit branches after each
EdgeConv block for training and benchmarking early exit strategies.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from phaze_ee.src.models.weaver_models.ParticleNet import ParticleNet
from phaze_ee.src.models.early_exit.exit_branches import LinearExitBranch


class ParticleNetEE(nn.Module):
    """ParticleNet with Early Exit branches.
    
    Wraps the original ParticleNet model and adds lightweight exit branches
    after each EdgeConv block. The original model architecture is preserved
    and remains identical to weaver-core's ParticleNet.
    
    Exit branches use simple global pooling + linear layers to produce
    class logits at intermediate stages.
    """
    
    def __init__(
        self,
        base_model: ParticleNet,
        num_classes: int,
        num_exit_points: Optional[int] = None,
        use_counts: bool = True,
    ):
        """Initialize ParticleNetEE.
        
        Args:
            base_model: The base ParticleNet model (from weaver_models)
            num_classes: Number of output classes
            num_exit_points: Number of exit points to add (default: number of EdgeConv blocks)
            use_counts: Whether to use particle counts for global pooling
        """
        super().__init__()
        
        # Store the base model
        self.base_model = base_model
        self.num_classes = num_classes
        self.use_counts = use_counts
        
        # Determine number of EdgeConv blocks in the base model
        num_edge_convs = len(base_model.edge_convs)
        if num_exit_points is None:
            num_exit_points = num_edge_convs
        else:
            num_exit_points = min(num_exit_points, num_edge_convs)
        
        self.num_exit_points = num_exit_points
        
        # Create exit branches - one per EdgeConv block
        # Each branch takes the output channels of the corresponding EdgeConv block
        self.exit_branches = nn.ModuleList()
        for i in range(num_exit_points):
            edge_conv = base_model.edge_convs[i]
            # Get output channels from the EdgeConv block
            # out_feats is a tuple/list; last element is the output channel count
            out_channels = edge_conv.out_feats[-1]
            
            branch = LinearExitBranch(
                input_channels=out_channels,
                num_classes=num_classes,
                use_counts=use_counts,
            )
            self.exit_branches.append(branch)
    
    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_exit_outputs: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass with early exit branches.
        
        This method intercepts the forward pass of ParticleNet, captures intermediate
        features after each EdgeConv block, applies exit branches, and returns both
        the final output and all intermediate predictions.
        
        Args:
            points: Point cloud coordinates (batch, 2, num_points) - eta/phi coords
            features: Point features (batch, num_features, num_points)
            mask: Optional binary mask (batch, 1, num_points)
            return_exit_outputs: Whether to compute and return exit outputs
        
        Returns:
            Tuple containing:
                - full_output: Final model output (batch, num_classes)
                - exit_outputs: List of exit branch outputs, one per exit point
                - exit_features: List of intermediate features (for detached strategies)
        """
        # ===== Replicate ParticleNet forward pass with exit interception =====
        
        # Prepare mask (from ParticleNet forward)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        
        # Optional batch norm on input features
        if hasattr(self.base_model, 'bn_fts') and self.base_model.bn_fts is not None:
            fts = self.base_model.bn_fts(features * mask) * mask
        else:
            fts = features
        
        # Lists to store exit outputs and features
        exit_outputs = []
        exit_features = []
        outputs_for_fusion = []  # In case fusion is used in base model
        
        # Run through EdgeConv blocks and capture intermediate outputs
        for idx, conv in enumerate(self.base_model.edge_convs):
            fts = conv(points, fts)
            
            # Apply exit branch if this is an exit point
            if return_exit_outputs and idx < self.num_exit_points:
                # Store the intermediate features (for detached strategies)
                exit_features.append(fts)
                
                # Apply exit branch
                exit_output = self.exit_branches[idx](fts, mask)
                exit_outputs.append(exit_output)
            
            # Store for fusion if base model uses fusion
            if self.base_model.use_fusion:
                outputs_for_fusion.append(fts)
        
        # Continue with the rest of ParticleNet's forward pass
        if self.base_model.use_fusion:
            fts = self.base_model.fusion_block(torch.cat(outputs_for_fusion, dim=1))
            if mask is not None:
                fts = fts * mask
        
        # Global pooling (matching ParticleNet's pooling strategy)
        if mask is not None:
            if self.base_model.use_counts:
                counts = mask.float().sum(dim=-1)
                fts = (fts * mask).sum(dim=-1) / counts.clamp(min=1)
            else:
                fts = (fts * mask).sum(dim=-1) / mask.sum(dim=-1)
        else:
            fts = fts.mean(dim=-1)
        
        # FC layers
        full_output = self.base_model.fc(fts)
        
        # Apply softmax if for_inference mode
        if self.base_model.for_inference:
            full_output = torch.softmax(full_output, dim=1)
            exit_outputs = [torch.softmax(out, dim=1) for out in exit_outputs]
        
        return full_output, exit_outputs, exit_features
    
    def forward_to_exit(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        exit_idx: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass up to a specific exit point (for FLOPs counting).
        
        This is useful for benchmarking the computational cost of early exits.
        
        Args:
            points: Point cloud coordinates (batch, 2, num_points)
            features: Point features (batch, num_features, num_points)
            exit_idx: Index of exit point to stop at (0-based)
            mask: Optional binary mask (batch, 1, num_points)
        
        Returns:
            torch.Tensor: Output at the specified exit point (batch, num_classes)
        """
        # Prepare mask
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)
        
        # Optional batch norm on input features
        if hasattr(self.base_model, 'bn_fts') and self.base_model.bn_fts is not None:
            fts = self.base_model.bn_fts(features * mask) * mask
        else:
            fts = features
        
        # Run through EdgeConv blocks up to exit_idx
        for idx, conv in enumerate(self.base_model.edge_convs):
            fts = conv(points, fts)
            if idx == exit_idx:
                # Apply exit branch and return
                return self.exit_branches[exit_idx](fts, mask)
        
        raise ValueError(f"exit_idx {exit_idx} out of range (max: {self.num_exit_points - 1})")
    
    def get_num_parameters_to_exit(self, exit_idx: int) -> int:
        """Get the total number of trainable parameters up to a given exit point.
        
        Args:
            exit_idx: Index of exit point (0-based)
        
        Returns:
            int: Total number of trainable parameters
        """
        total_params = 0
        
        # Count parameters in input batch norm (if used)
        if hasattr(self.base_model, 'bn_fts') and self.base_model.bn_fts is not None:
            total_params += sum(p.numel() for p in self.base_model.bn_fts.parameters() if p.requires_grad)
        
        # Count parameters in EdgeConv blocks up to exit_idx
        for idx in range(exit_idx + 1):
            total_params += sum(p.numel() for p in self.base_model.edge_convs[idx].parameters() if p.requires_grad)
        
        # Count parameters in the exit branch
        total_params += sum(p.numel() for p in self.exit_branches[exit_idx].parameters() if p.requires_grad)
        
        return total_params
    
    def get_full_model_parameters(self) -> int:
        """Get the total number of trainable parameters in the full model.
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)


def create_particle_net_ee(
    input_dims: int,
    num_classes: int,
    conv_params: List[Tuple[int, Tuple[int, ...]]],
    fc_params: List[Tuple[int, float]],
    num_exit_points: Optional[int] = None,
    use_fusion: bool = False,
    use_fts_bn: bool = True,
    use_counts: bool = True,
    for_inference: bool = False,
    **kwargs,
) -> ParticleNetEE:
    """Factory function to create ParticleNetEE model.
    
    Args:
        input_dims: Number of input feature dimensions
        num_classes: Number of output classes
        conv_params: EdgeConv parameters [(k, (c1, c2, ...)), ...]
        fc_params: FC layer parameters [(units, dropout), ...]
        num_exit_points: Number of exit points (default: all EdgeConv blocks)
        use_fusion: Whether to use fusion of all EdgeConv outputs
        use_fts_bn: Whether to use batch norm on input features
        use_counts: Whether to use particle counts for global pooling
        for_inference: Whether model is for inference (applies softmax)
        **kwargs: Additional arguments for ParticleNet
    
    Returns:
        ParticleNetEE: Model with early exit branches
    """
    # Create base ParticleNet model
    base_model = ParticleNet(
        input_dims=input_dims,
        num_classes=num_classes,
        conv_params=conv_params,
        fc_params=fc_params,
        use_fusion=use_fusion,
        use_fts_bn=use_fts_bn,
        use_counts=use_counts,
        for_inference=for_inference,
        **kwargs,
    )
    
    # Wrap with early exit
    model = ParticleNetEE(
        base_model=base_model,
        num_classes=num_classes,
        num_exit_points=num_exit_points,
        use_counts=use_counts,
    )
    
    return model
