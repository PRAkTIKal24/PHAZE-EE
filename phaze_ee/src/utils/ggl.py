# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Global Governance Layer (GGL) for PHAZE-EE.

This module provides the argument parsing, project creation, and mode orchestration
for the PHAZE-EE framework. It handles command-line argument parsing and dispatches
to appropriate mode functions for training, evaluation, benchmarking, and plotting.
"""

import argparse
import importlib
import os
import sys
from dataclasses import dataclass

import art as ar
from tqdm import tqdm

from .config import Config


def get_arguments():
    """Parse command-line arguments.

    Returns:
        tuple: (config, mode, options, workspace_name, project_name, verbose)
            - config: Config dataclass instance (None for new_project mode)
            - mode: str, execution mode
            - options: str, additional options (for chain mode)
            - workspace_name: str, workspace name
            - project_name: str, project name
            - verbose: bool, verbose output flag
    """
    parser = argparse.ArgumentParser(
        prog="phaze_ee",
        description=(
            ar.text2art(" PHAZE-EE ", font="varsity")
            + "\n       Early Exit Mechanisms for High Energy Physics\n"
            + "       ================================================\n\n"
            + "       Fast inference via strategic early exits in ML models.\n"
            + "       Part of the PHAZE framework for optimized particle physics ML.\n\n"
        ),
        epilog="Happy Optimizing!",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help=(
            "new_project \t creates new workspace and project directories\n"
            "train \t\t trains early exit model with configured architecture\n"
            "evaluate \t evaluates early exit performance at each exit point\n"
            "benchmark \t profiles FLOPs and parameters for each exit strategy\n"
            "plot \t\t generates parameter reduction vs performance plots\n"
            "compare \t generates comparison plots across multiple projects\n"
            "chain \t\t executes full pipeline (use with -o option)\n"
        ),
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        required=True,
        nargs=2,
        metavar=("WORKSPACE", "PROJECT"),
        help=(
            "Specifies workspace and project.\n"
            "e.g. < --project JetClass first_test >\n"
            "Creates: phaze_ee/workspaces/JetClass/first_test/\n"
        ),
    )
    parser.add_argument(
        "-o",
        "--options",
        type=str,
        required=False,
        help=(
            "Additional options for chain mode.\n"
            "e.g. < -o newproject_train_evaluate_plot >\n"
            "Executes modes in sequence: new_project -> train -> evaluate -> plot\n"
        ),
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs='+',
        required=False,
        help=(
            "List of project names for comparison mode.\n"
            "e.g. < --projects strategy_mimic strategy_target >\n"
            "Compares results from multiple projects in the same workspace.\n"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose mode - detailed logging",
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    workspace_name = args.project[0]
    project_name = args.project[1]
    project_path = os.path.join("phaze_ee/workspaces", workspace_name, project_name)
    config_path = f"phaze_ee.workspaces.{workspace_name}.{project_name}.config.{project_name}_config"

    if args.mode == "new_project":
        config = None
    elif args.mode == "compare":
        # Compare mode doesn't need a specific project config
        config = None
    else:
        if not os.path.exists(project_path):
            print(
                f"Project path {project_path} does not exist. Please run --mode=new_project first."
            )
            sys.exit()
        else:
            config = Config
            try:
                importlib.import_module(config_path).set_config(config)
            except ModuleNotFoundError:
                print(
                    f"Config file not found at {config_path}. Please check your project setup."
                )
                sys.exit()

    return (config, args.mode, args.options, workspace_name, project_name, args.verbose, args.projects)


def create_new_project(
    workspace_name: str,
    project_name: str,
    verbose: bool = False,
    base_path: str = "phaze_ee/workspaces",
) -> None:
    """Create a new PHAZE-EE project with directory structure and default config.

    Args:
        workspace_name: Name of the workspace (e.g., 'JetClass')
        project_name: Name of the project (e.g., 'first_test')
        verbose: If True, print detailed progress information
        base_path: Base path for workspaces directory

    Creates:
        - Workspace and project directories
        - Data subdirectories (h5, root, parquet)
        - Config directory with default config file
        - Output directories for models, plots, and results
    """
    workspace_path = os.path.join(base_path, workspace_name)
    project_path = os.path.join(base_path, workspace_name, project_name)

    if os.path.exists(project_path):
        print(f"The workspace and project ({project_path}) already exists.")
        return

    os.makedirs(project_path)

    required_directories = [
        os.path.join(workspace_path, "data", "h5"),
        os.path.join(workspace_path, "data", "root"),
        os.path.join(workspace_path, "data", "parquet"),
        os.path.join(project_path, "config"),
        os.path.join(project_path, "output", "results"),
        os.path.join(project_path, "output", "plots", "performance"),
        os.path.join(project_path, "output", "plots", "loss"),
        os.path.join(project_path, "output", "plots", "exit_distribution"),
        os.path.join(project_path, "output", "plots", "param_reduction"),
        os.path.join(project_path, "output", "models"),
        os.path.join(project_path, "output", "logs"),
    ]

    for directory in tqdm(required_directories, desc="Creating directories"):
        os.makedirs(directory, exist_ok=True)

    # Write default config file
    config_content = create_default_config(workspace_name, project_name)
    config_file_path = os.path.join(project_path, "config", f"{project_name}_config.py")

    with open(config_file_path, "w") as f:
        f.write(config_content)

    if verbose:
        print(f"\nProject created successfully at: {project_path}")
        print(f"Config file: {config_file_path}")
        print("\nNext steps:")
        print(f"  1. Edit config: {config_file_path}")
        print(f"  2. Add data to: {os.path.join(workspace_path, 'data')}")
        print(f"  3. Train model: phaze_ee -m train -p {workspace_name} {project_name}")


def create_default_config(workspace_name: str, project_name: str) -> str:
    """Generate default configuration file content.

    Args:
        workspace_name: Name of the workspace
        project_name: Name of the project

    Returns:
        str: Python code for default configuration file
    """
    return f'''# Configuration file for {project_name}
# Workspace: {workspace_name}
# PHAZE-EE Early Exit Training Configuration

def set_config(c):
    """Set configuration parameters for PHAZE-EE early exit training.
    
    Args:
        c: Config dataclass to populate
    """
    # ===== Model Configuration =====
    c.model_name = "ParticleNet"  # Options: ParticleNet, ParticleTransformer, ParticleNeXt
    c.num_classes = 10  # Number of output classes (JetClass has 10 classes)
    c.num_exit_points = 3  # Number of early exit points to add
    
    # ParticleNet Architecture (matching weaver-core example for JetClass)
    c.conv_params = [
        (16, (64, 64, 64)),      # EdgeConv block 0: k=16, channels=(64,64,64)
        (16, (128, 128, 128)),   # EdgeConv block 1: k=16, channels=(128,128,128)
        (16, (256, 256, 256)),   # EdgeConv block 2: k=16, channels=(256,256,256)
    ]
    c.fc_params = [(256, 0.1)]  # FC: 256 units, 0.1 dropout
    c.use_fusion = False         # No fusion in weaver example
    c.use_fts_bn = True          # Use batch norm on input features
    c.use_counts = True          # Use particle counts for global pooling
    
    # ===== Training Configuration =====
    c.epochs = 50                # Total training epochs (weaver default for JetClass)
    c.lr = 1e-2                  # Learning rate (1e-2 for ParticleNet, 1e-3 for ParT)
    c.batch_size = 512           # Batch size
    c.optimizer = "ranger"       # Options: adam, adamw, ranger (weaver default)
    c.scheduler = "flat+decay"   # Options: flat+decay, cosine, step (weaver default)
    c.weight_decay = 1e-4        # Weight decay for optimizer
    c.gradient_clip = 1.0        # Gradient clipping value (0 to disable)
    c.samples_per_epoch = None   # Limit samples per epoch (None for full dataset)
    
    # ===== Early Exit Loss Strategy =====
    # Options: mimic_detached, mimic_flow, target_detached, target_flow
    c.exit_loss_strategy = "mimic_detached"
    # - mimic_detached: Exit learns to mimic full model output (MSE, no gradient to backbone via exit)
    # - mimic_flow: Exit learns to mimic full model output (MSE, gradients flow through full model)
    # - target_detached: Exit learns from ground truth (CE, features detached)
    # - target_flow: Exit learns from ground truth (CE, gradients flow through backbone)
    
    # ===== Beta Scheduling (Exit Loss Weights) =====
    c.beta_max = [0.1, 0.1, 0.1]     # Max beta value for each exit point
    c.beta_zero_epochs = 10          # Number of initial epochs with beta=0
    c.beta_ramp_type = "linear"      # Ramp type: linear or cosine
    
    # ===== Data Configuration =====
    c.data_config = "data_configs/JetClass_full.yaml"  # Path to weaver data config YAML
    
    # Data file paths (auto-populated from data_config or manually specified)
    # Example: c.train_files = glob('/path/to/JetClass/train/**/*.root')
    c.train_files = []  # Training data files
    c.val_files = []    # Validation data files
    c.test_files = []   # Test data files
    
    # ===== Distributed Training =====
    c.use_ddp = False   # Set True for multi-GPU training with torchrun
    c.use_amp = True    # Automatic Mixed Precision (fp16)
    
    # ===== Logging and Checkpointing =====
    c.log_interval = 10   # Log metrics every N batches
    c.val_interval = 1    # Validate every N epochs
    c.save_interval = 10  # Save checkpoint every N epochs
    c.num_workers = 4     # DataLoader workers (increase for faster I/O)
    
    # ===== Benchmarking =====
    c.profile_flops = True        # Profile FLOPs at each exit point
    c.num_profile_samples = 1000  # Number of samples for profiling
    
    # ===== Plotting =====
    c.plot_formats = ["png", "pdf"]  # Output formats for plots
    c.plot_dpi = 300                 # Resolution for saved plots
    
    # ===== Legacy Fields (for backward compatibility) =====
    c.exit_threshold = 0.9     # Confidence threshold for dynamic exit (future work)
    c.exit_strategy = "confidence"  # Exit decision strategy (future work)
'''


def run_training(paths: dict, config, verbose: bool = False) -> None:
    """Train early exit model.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with training parameters
        verbose: If True, print detailed progress information
    """
    import torch
    from weaver.utils.data.config import DataConfig
    from torch.utils.data import DataLoader
    
    from phaze_ee.src.models.early_exit.particle_net_ee import create_particle_net_ee
    from phaze_ee.src.trainers.training import train_early_exit_model
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHAZE-EE Training: {config.model_name}")
        print(f"{'='*60}")
        print(f"Exit points: {config.num_exit_points}")
        print(f"Exit loss strategy: {config.exit_loss_strategy}")
        print(f"Beta schedule: {config.beta_ramp_type}, max={config.beta_max}")
        print(f"Output directory: {paths['output_path']}")
        print(f"{'='*60}\n")
    
    # Load data configuration
    if not config.data_config:
        raise ValueError("data_config must be specified in project config")
    
    data_config = DataConfig.load(config.data_config)
    
    if verbose:
        print(f"Data config: {config.data_config}")
        print(f"Num classes: {len(data_config.label_value)}")
        print(f"Input features: {data_config.input_names}")
    
    # Create data loaders using weaver's data pipeline
    try:
        from weaver.utils.dataset import SimpleIterDataset
        
        # Training dataset
        train_dataset = SimpleIterDataset(
            data_config,
            config.train_files if config.train_files else [],
            for_training=True,
            load_range=(0, 1),  # Load all data
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        # Validation dataset
        val_dataset = SimpleIterDataset(
            data_config,
            config.val_files if config.val_files else [],
            for_training=False,
            load_range=(0, 1),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        
        if verbose:
            print(f"Training samples: {len(config.train_files)} files")
            print(f"Validation samples: {len(config.val_files)} files")
    
    except ImportError:
        print("Warning: weaver.utils.dataset not available. Using dummy data loaders for testing.")
        # Fallback to dummy loaders for development/testing
        # In production, weaver must be properly installed
        train_loader = None
        val_loader = None
    
    # Create early exit model
    if config.model_name == "ParticleNet":
        # Get input dimensions from data config
        pf_features_dims = len(data_config.input_dicts['pf_features'])
        num_classes = len(data_config.label_value)
        
        model = create_particle_net_ee(
            input_dims=pf_features_dims,
            num_classes=num_classes,
            conv_params=config.conv_params,
            fc_params=config.fc_params,
            num_exit_points=config.num_exit_points,
            use_fusion=config.use_fusion,
            use_fts_bn=config.use_fts_bn,
            use_counts=config.use_counts,
            for_inference=False,
        )
        
        if verbose:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nModel: ParticleNetEE")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"EdgeConv blocks: {len(config.conv_params)}")
            print(f"Exit branches: {config.num_exit_points}\n")
    else:
        raise NotImplementedError(f"Model {config.model_name} not yet implemented for early exit")
    
    # Train the model
    if train_loader is not None and val_loader is not None:
        history = train_early_exit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            paths=paths,
            verbose=verbose,
        )
        
        if verbose:
            print("\nTraining completed successfully!")
    else:
        print("Skipping training due to missing data loaders.")
        print("Please ensure weaver-core is properly installed and data files are configured.")


def run_evaluation(paths: dict, config, verbose: bool = False) -> None:
    """Evaluate early exit performance.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with evaluation parameters
        verbose: If True, print detailed progress information
    """
    import json
    import torch
    import numpy as np
    from pathlib import Path
    from weaver.utils.data.config import DataConfig
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    from phaze_ee.src.models.early_exit.particle_net_ee import create_particle_net_ee
    from phaze_ee.src.trainers.training import load_checkpoint
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHAZE-EE Evaluation: {config.model_name}")
        print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data configuration
    data_config = DataConfig.load(config.data_config)
    
    # Load test dataset
    try:
        from weaver.utils.dataset import SimpleIterDataset
        
        test_dataset = SimpleIterDataset(
            data_config,
            config.test_files if config.test_files else config.val_files,
            for_training=False,
            load_range=(0, 1),
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    except ImportError:
        print("Warning: weaver.utils.dataset not available. Cannot run evaluation.")
        return
    
    # Create model
    pf_features_dims = len(data_config.input_dicts['pf_features'])
    num_classes = len(data_config.label_value)
    
    model = create_particle_net_ee(
        input_dims=pf_features_dims,
        num_classes=num_classes,
        conv_params=config.conv_params,
        fc_params=config.fc_params,
        num_exit_points=config.num_exit_points,
        use_fusion=config.use_fusion,
        use_fts_bn=config.use_fts_bn,
        use_counts=config.use_counts,
        for_inference=True,  # Apply softmax for evaluation
    )
    
    # Load trained weights
    model_path = Path(paths['output_path']) / 'models' / 'best_model.pt'
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first using --mode train")
        return
    
    checkpoint = load_checkpoint(model, model_path, device=device)
    model = model.to(device)
    model.eval()
    
    if verbose:
        print(f"Loaded model from epoch {checkpoint['epoch']+1}")
        print(f"Running evaluation on test set...\n")
    
    # Collect predictions
    all_labels = []
    full_preds = []
    exit_preds = [[] for _ in range(config.num_exit_points)]
    
    with torch.no_grad():
        for X, y, _ in test_loader:
            points = X.get('pf_points', X.get('points')).to(device)
            features = X.get('pf_features', X.get('features')).to(device)
            mask = X.get('pf_mask', X.get('mask', None))
            if mask is not None:
                mask = mask.to(device)
            
            labels = y.get('_label_', y.get('label')).long()
            
            # Forward pass
            full_output, exit_outputs, _ = model(
                points, features, mask, return_exit_outputs=True
            )
            
            # Collect predictions (probabilities)
            all_labels.append(labels.cpu().numpy())
            full_preds.append(full_output.cpu().numpy())
            
            for i, exit_out in enumerate(exit_outputs):
                exit_preds[i].append(exit_out.cpu().numpy())
    
    # Concatenate batches
    all_labels = np.concatenate(all_labels)
    full_preds = np.concatenate(full_preds)
    exit_preds = [np.concatenate(ep) for ep in exit_preds]
    
    # Compute metrics
    results = {}
    
    # Full model metrics
    full_pred_labels = full_preds.argmax(axis=1)
    full_acc = accuracy_score(all_labels, full_pred_labels)
    full_auc = roc_auc_score(all_labels, full_preds, multi_class='ovo', average='macro')
    
    results['full_model'] = {
        'accuracy': float(full_acc),
        'roc_auc': float(full_auc),
    }
    
    # Exit point metrics
    for i, exit_pred in enumerate(exit_preds):
        exit_pred_labels = exit_pred.argmax(axis=1)
        exit_acc = accuracy_score(all_labels, exit_pred_labels)
        exit_auc = roc_auc_score(all_labels, exit_pred, multi_class='ovo', average='macro')
        
        results[f'exit_{i}'] = {
            'accuracy': float(exit_acc),
            'roc_auc': float(exit_auc),
        }
    
    # Save results
    results_path = Path(paths['output_path']) / 'results' / 'evaluation_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"{'='*60}")
        print(f"Full Model: Accuracy={full_acc:.4f}, ROC-AUC={full_auc:.4f}")
        for i in range(config.num_exit_points):
            exit_res = results[f'exit_{i}']
            print(f"Exit {i}: Accuracy={exit_res['accuracy']:.4f}, ROC-AUC={exit_res['roc_auc']:.4f}")
        print(f"\nResults saved to: {results_path}")


def run_benchmark(paths: dict, config, verbose: bool = False) -> None:
    """Benchmark FLOPs and parameters for each exit point.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with benchmark parameters
        verbose: If True, print detailed progress information
    """
    from phaze_ee.src.utils.benchmark import benchmark_exit_points
    
    if verbose:
        print(f"\n{'='*60}")
        print("PHAZE-EE Benchmarking")
        print(f"{'='*60}")
    
    # Run benchmarking
    benchmark_results = benchmark_exit_points(
        paths=paths,
        config=config,
        verbose=verbose,
    )
    
    if verbose and benchmark_results:
        print(f"\n{'='*60}")
        print("Benchmark Results:")
        print(f"{'='*60}")
        for key, metrics in benchmark_results.items():
            print(f"\n{key}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")


def run_plots(paths: dict, config, verbose: bool = False) -> None:
    """Generate visualization plots for early exit analysis.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with plotting parameters
        verbose: If True, print detailed progress information
    """
    from phaze_ee.src.utils.plotting import (
        plot_flops_vs_accuracy,
        plot_params_vs_accuracy,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("PHAZE-EE Plotting")
        print(f"{'='*60}")
    
    # Generate plots
    plot_flops_vs_accuracy(paths=paths, config=config, verbose=verbose)
    plot_params_vs_accuracy(paths=paths, config=config, verbose=verbose)
    
    if verbose:
        print("\nPlots generated successfully!")


def run_comparison(
    workspace_path: str,
    project_names: list,
    output_path: str = None,
    verbose: bool = False,
) -> None:
    """Generate comparison plots across multiple projects.
    
    Args:
        workspace_path: Path to workspace directory
        project_names: List of project names to compare
        output_path: Optional custom output path for comparison plots
        verbose: If True, print detailed progress information
    """
    from phaze_ee.src.utils.plotting import (
        plot_comparison_flops_vs_accuracy,
        plot_comparison_params_vs_accuracy,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("PHAZE-EE Multi-Project Comparison")
        print(f"{'='*60}")
        print(f"Workspace: {workspace_path}")
        print(f"Comparing projects: {', '.join(project_names)}")
        print(f"{'='*60}\n")
    
    # Determine output directory
    if output_path is None:
        # Save comparison plots in workspace/comparisons/ directory
        output_path = os.path.join(workspace_path, 'comparisons')
    
    os.makedirs(output_path, exist_ok=True)
    
    # Generate comparison plots
    flops_save_path = os.path.join(output_path, 'comparison_flops_vs_accuracy.png')
    params_save_path = os.path.join(output_path, 'comparison_params_vs_accuracy.png')
    
    if verbose:
        print("Generating FLOPs vs Accuracy comparison plot...")
    
    plot_comparison_flops_vs_accuracy(
        workspace_path=workspace_path,
        project_names=project_names,
        save_path=flops_save_path,
        plot_formats=['png', 'pdf'],
        plot_dpi=300,
        verbose=verbose,
    )
    
    if verbose:
        print("\nGenerating Parameters vs Accuracy comparison plot...")
    
    plot_comparison_params_vs_accuracy(
        workspace_path=workspace_path,
        project_names=project_names,
        save_path=params_save_path,
        plot_formats=['png', 'pdf'],
        plot_dpi=300,
        verbose=verbose,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print("Comparison plots generated successfully!")
        print(f"Output directory: {output_path}")
        print(f"{'='*60}\n")


def run_full_chain(
    workspace_name: str,
    project_name: str,
    paths: dict,
    config,
    options: str,
    verbose: bool = False,
) -> None:
    """Execute full pipeline based on chain options.

    Args:
        workspace_name: Name of the workspace
        project_name: Name of the project
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass (may be None if newproject is in chain)
        options: Underscore-separated mode names (e.g., 'newproject_train_evaluate_plot')
        verbose: If True, print detailed progress information
    """
    OPTION_TO_MODE = {
        "newproject": "new_project",
        "train": "train",
        "evaluate": "evaluate",
        "benchmark": "benchmark",
        "plot": "plot",
    }

    MODE_OPERATIONS = {
        "new_project": {
            "func": create_new_project,
            "args": (workspace_name, project_name, verbose),
        },
        "train": {"func": run_training, "args": (paths, config, verbose)},
        "evaluate": {"func": run_evaluation, "args": (paths, config, verbose)},
        "benchmark": {"func": run_benchmark, "args": (paths, config, verbose)},
        "plot": {"func": run_plots, "args": (paths, config, verbose)},
    }

    workflow = options.split("_")
    
    if verbose:
        print(f"Running chain workflow: {' -> '.join(workflow)}")
    
    for step in workflow:
        if step not in OPTION_TO_MODE:
            print(f"Warning: Unknown chain option '{step}', skipping...")
            continue
        
        mode_name = OPTION_TO_MODE[step]
        operation = MODE_OPERATIONS[mode_name]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Executing: {mode_name}")
            print(f"{'='*60}\n")
        
        operation["func"](*operation["args"])
