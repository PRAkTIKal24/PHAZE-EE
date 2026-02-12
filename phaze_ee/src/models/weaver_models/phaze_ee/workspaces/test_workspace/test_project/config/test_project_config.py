# Configuration file for test_project
# Workspace: test_workspace

def set_config(c):
    """Set configuration parameters for PHAZE-EE early exit training.
    
    Args:
        c: Config dataclass to populate
    """
    # Model configuration
    c.model_name = "ParticleNet"  # Options: ParticleNet, ParticleTransformer, ParticleNeXt
    c.num_classes = 10  # Number of output classes (e.g., JetClass has 10 classes)
    c.num_exit_points = 3  # Number of early exit points to add
    
    # Training configuration
    c.epochs = 100
    c.lr = 1e-3
    c.batch_size = 512
    c.optimizer = "ranger"  # Options: adam, adamw, ranger
    c.scheduler = "cosine"  # Options: cosine, step, plateau
    c.weight_decay = 1e-4
    
    # Early exit specific
    c.exit_loss_weights = [0.3, 0.5, 1.0]  # Loss weights for each exit (increasing)
    c.exit_threshold = 0.9  # Confidence threshold for early exit during inference
    c.exit_strategy = "confidence"  # Options: confidence, entropy, learned
    
    # Data configuration
    c.data_config = "data_configs/JetClass_full.yaml"  # Path to weaver data config
    c.train_files = []  # Auto-populated from data_config
    c.val_files = []
    c.test_files = []
    
    # Distributed training
    c.use_ddp = False  # Set True for multi-GPU training
    c.use_amp = True  # Automatic Mixed Precision
    
    # Logging and checkpointing
    c.log_interval = 10  # Log every N batches
    c.val_interval = 1  # Validate every N epochs
    c.save_interval = 10  # Save checkpoint every N epochs
    c.num_workers = 4  # DataLoader workers
    
    # Benchmarking
    c.profile_flops = True  # Profile FLOPs at each exit point
    c.num_profile_samples = 1000  # Number of samples for profiling
    
    # Plotting
    c.plot_formats = ["png", "pdf"]  # Output formats for plots
    c.plot_dpi = 300
