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

    return (config, args.mode, args.options, workspace_name, project_name, args.verbose)


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
'''


def run_training(paths: dict, config, verbose: bool = False) -> None:
    """Train early exit model.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with training parameters
        verbose: If True, print detailed progress information
    """
    if verbose:
        print(f"Starting training for {config.model_name} with {config.num_exit_points} exits...")
        print(f"Output directory: {paths['output_path']}")
    
    # TODO: Implement training loop
    # This will be implemented in phaze_ee/src/trainers/training.py
    print("Training mode: Not yet implemented. Will use trainers/training.py")


def run_evaluation(paths: dict, config, verbose: bool = False) -> None:
    """Evaluate early exit performance.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with evaluation parameters
        verbose: If True, print detailed progress information
    """
    if verbose:
        print(f"Evaluating early exit points for {config.model_name}...")
    
    # TODO: Implement evaluation
    print("Evaluation mode: Not yet implemented.")


def run_benchmark(paths: dict, config, verbose: bool = False) -> None:
    """Benchmark FLOPs and parameters for each exit point.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with benchmark parameters
        verbose: If True, print detailed progress information
    """
    if verbose:
        print("Running benchmark profiling...")
    
    # TODO: Implement benchmarking using weaver's flops_counter
    print("Benchmark mode: Not yet implemented. Will use utils/benchmark.py")


def run_plots(paths: dict, config, verbose: bool = False) -> None:
    """Generate visualization plots for early exit analysis.

    Args:
        paths: Dictionary with workspace_path, project_path, data_path, output_path
        config: Config dataclass with plotting parameters
        verbose: If True, print detailed progress information
    """
    if verbose:
        print("Generating plots...")
    
    # TODO: Implement plotting
    print("Plot mode: Not yet implemented. Will use utils/plotting.py")


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
