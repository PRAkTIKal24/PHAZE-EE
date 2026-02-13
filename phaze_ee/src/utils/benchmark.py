# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Benchmarking utilities for early exit models.

This module provides functions to profile FLOPs, parameters, and latency
for each exit point in early exit models.
"""

import json
from pathlib import Path
from typing import Dict, Optional  # noqa: F401

import torch
import torch.nn as nn


def benchmark_exit_points(
    paths: dict,
    config,
    verbose: bool = False,
) -> dict:
    """Benchmark FLOPs and parameters for each exit point.

    Combines evaluation results (accuracy/AUC) with complexity metrics (FLOPs/params)
    to generate comprehensive benchmark data.

    Args:
        paths: Dictionary with workspace_path, project_path, output_path
        config: Config dataclass with benchmark parameters
        verbose: If True, print detailed results

    Returns:
        dict: Benchmark results with FLOPs, parameters, accuracy, and AUC per exit
    """
    from weaver.utils.data.config import DataConfig

    from phaze_ee.src.models.early_exit.particle_net_ee import create_particle_net_ee
    from phaze_ee.src.trainers.training import load_checkpoint

    if verbose:
        print("Loading model for benchmarking...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data configuration to get input dimensions
    data_config = DataConfig.load(config.data_config)
    pf_features_dims = len(data_config.input_dicts["pf_features"])
    num_classes = len(data_config.label_value)

    # Create model
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
    # Load trained weights (optional, for consistency with evaluation)
    model_path = Path(paths["output_path"]) / "models" / "best_model.pt"
    if model_path.exists():
        load_checkpoint(model, model_path, device=device)
        if verbose:
            print(f"Loaded model from {model_path}")
    else:
        if verbose:
            print("Warning: No trained model found, using random initialization for FLOPs counting")

    model = model.to(device)
    model.eval()

    # Load evaluation results (accuracy/AUC metrics)
    eval_results_path = Path(paths["output_path"]) / "results" / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)
    else:
        eval_results = {}
        if verbose:
            print(
                "Warning: No evaluation results found. Run evaluation first for complete metrics."
            )

    # Benchmark results dictionary
    benchmark_results = {}

    # Create sample input for FLOPs counting
    # Input shapes: pf_points (B, 2, P), pf_features (B, C, P), pf_mask (B, 1, P)
    batch_size = 1
    num_points = 128  # From data config

    sample_points = torch.randn(batch_size, 2, num_points).to(device)
    sample_features = torch.randn(batch_size, pf_features_dims, num_points).to(device)
    sample_mask = torch.ones(batch_size, 1, num_points).to(device)

    # Benchmark each exit point
    for i in range(config.num_exit_points):
        if verbose:
            print(f"Profiling exit point {i}...")

        # Create partial model up to this exit
        class PartialModel(nn.Module):
            def __init__(self, full_model, exit_idx):
                super().__init__()
                self.full_model = full_model
                self.exit_idx = exit_idx

            def forward(self, points, features, mask):
                return self.full_model.forward_to_exit(points, features, self.exit_idx, mask)

        partial_model = PartialModel(model, i)

        # Profile FLOPs using weaver's counter
        try:
            from weaver.utils.flops_counter import get_model_complexity_info

            # Wrapper to match expected input format
            def input_constructor(input_res):
                return (sample_points, sample_features, sample_mask)

            macs_str, params_str = get_model_complexity_info(
                partial_model,
                input_res=(2, num_points),  # Dummy resolution
                input_constructor=input_constructor,
                as_strings=True,
                print_per_layer_stat=False,
            )

            # Parse strings for numeric values
            flops = parse_complexity_string(macs_str)

        except Exception as e:
            if verbose:
                print(f"Warning: FLOPs counting failed: {e}")
            flops = None
            params_str = "Unknown"  # noqa: F841

        # Count parameters
        params = model.get_num_parameters_to_exit(i)

        # Get evaluation metrics
        exit_key = f"exit_{i}"
        accuracy = eval_results.get(exit_key, {}).get("accuracy", None)
        roc_auc = eval_results.get(exit_key, {}).get("roc_auc", None)

        benchmark_results[exit_key] = {
            "flops": flops,
            "flops_str": macs_str if flops is not None else "Unknown",
            "params": params,
            "params_str": f"{params / 1e6:.2f}M",
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

    # Benchmark full model
    if verbose:
        print("Profiling full model...")

    try:
        from weaver.utils.flops_counter import get_model_complexity_info

        def input_constructor_full(input_res):
            return (sample_points, sample_features, sample_mask)

        # Use base model for full FLOPs counting
        macs_str_full, params_str_full = get_model_complexity_info(
            model.base_model,
            input_res=(2, num_points),
            input_constructor=input_constructor_full,
            as_strings=True,
            print_per_layer_stat=False,
        )

        flops_full = parse_complexity_string(macs_str_full)

    except Exception as e:
        if verbose:
            print(f"Warning: FLOPs counting for full model failed: {e}")
        flops_full = None
        macs_str_full = "Unknown"

    params_full = model.get_full_model_parameters()
    accuracy_full = eval_results.get("full_model", {}).get("accuracy", None)
    roc_auc_full = eval_results.get("full_model", {}).get("roc_auc", None)

    benchmark_results["full_model"] = {
        "flops": flops_full,
        "flops_str": macs_str_full,
        "params": params_full,
        "params_str": f"{params_full / 1e6:.2f}M",
        "accuracy": accuracy_full,
        "roc_auc": roc_auc_full,
    }

    # Save benchmark results
    benchmark_path = Path(paths["output_path"]) / "results" / "benchmark_results.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    if verbose:
        print(f"\nBenchmark results saved to: {benchmark_path}")

    return benchmark_results


def parse_complexity_string(complexity_str: str) -> Optional[float]:
    """Parse complexity string from weaver's flops_counter.

    Examples: "3.14 MMac" -> 3.14e6, "512.5 KMac" -> 512.5e3

    Args:
        complexity_str: String like "3.14 MMac" or "1.23 GMac"

    Returns:
        float: Numeric value in base units, or None if parsing fails
    """
    try:
        parts = complexity_str.strip().split()
        value = float(parts[0])
        unit = parts[1] if len(parts) > 1 else ""

        multipliers = {
            "Mac": 1,
            "KMac": 1e3,
            "MMac": 1e6,
            "GMac": 1e9,
        }

        multiplier = multipliers.get(unit, 1)
        return value * multiplier
    except Exception:
        return None


def profile_model_complexity(
    model: nn.Module,
    input_shape: tuple,
) -> tuple:
    """Profile overall model complexity.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)

    Returns:
        tuple: (total_flops, total_params)
    """
    try:
        from weaver.utils.flops_counter import get_model_complexity_info

        macs_str, params_str = get_model_complexity_info(
            model,
            input_res=input_shape,
            as_strings=True,
            print_per_layer_stat=False,
        )

        flops = parse_complexity_string(macs_str)
        params = sum(p.numel() for p in model.parameters())

        return flops, params
    except Exception as e:
        print(f"Warning: Model complexity profiling failed: {e}")
        return None, None
