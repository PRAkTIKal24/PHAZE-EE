"""
ParticleNet network configuration for JetClass dataset.
Replicates the official weaver-core example architecture:
- 3 EdgeConv blocks: k=16, channels (64,64,64), (128,128,128), (256,256,256)
- No fusion (use_fusion=False)
- FC layer: 256 units with 0.1 dropout
- Output: 10 classes (JetClass)
"""

import torch
import torch.nn as nn
from phaze_ee.src.models.weaver_models.ParticleNet import ParticleNet


class ParticleNetWrapper(nn.Module):
    """
    Wrapper to adapt 4-input data format (points, features, lorentz_vectors, mask)
    to ParticleNet's 3-input signature (points, features, mask).
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ParticleNet(**kwargs)
    
    def forward(self, points, features, lorentz_vectors, mask):
        # ParticleNet doesn't use lorentz_vectors, so we ignore it
        return self.mod(points, features, mask)


def get_model(data_config, **kwargs):
    """
    Instantiate ParticleNet model for JetClass.
    
    Args:
        data_config: DataConfig object from weaver
        **kwargs: Additional options (e.g., for_inference, for_segmentation)
    
    Returns:
        model: ParticleNetWrapper instance
        model_info: dict with input/output metadata for weaver
    """
    # Architecture matching weaver-core ParticleNet example
    conv_params = [
        (16, (64, 64, 64)),      # EdgeConv block 0: k=16, channels=(64,64,64)
        (16, (128, 128, 128)),   # EdgeConv block 1: k=16, channels=(128,128,128)
        (16, (256, 256, 256)),   # EdgeConv block 2: k=16, channels=(256,256,256)
    ]
    fc_params = [(256, 0.1)]  # FC: 256 units, 0.1 dropout
    
    # Get input dimensions from data config
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    
    # Create model
    model = ParticleNetWrapper(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        conv_params=conv_params,
        fc_params=fc_params,
        use_fusion=False,   # No fusion in this architecture
        use_fts_bn=True,    # Use batch norm on input features
        use_counts=True,    # Use particle counts for global pooling
        for_inference=kwargs.get('for_inference', False),
        for_segmentation=kwargs.get('for_segmentation', False),
    )
    
    # Model info for weaver
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            **{'softmax': {0: 'N'}},
        },
    }
    
    return model, model_info


def get_loss(data_config, **kwargs):
    """
    Loss function for classification.
    Returns CrossEntropyLoss to match weaver-core default.
    """
    return torch.nn.CrossEntropyLoss()
