# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Plotting utilities for early exit analysis.

This module provides functions to generate visualizations for early exit performance,
including parameter reduction vs accuracy curves, exit distribution plots, etc.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


# Set publication-quality plot style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100


def plot_flops_vs_accuracy(
    paths: dict,
    config,
    verbose: bool = False,
):
    """Plot FLOPs vs Accuracy for each exit point and full model.
    
    Args:
        paths: Dictionary with output_path
        config: Config dataclass with plotting parameters
        verbose: If True, print detailed progress
    """
    # Load benchmark results
    benchmark_path = Path(paths['output_path']) / 'results' / 'benchmark_results.json'
    if not benchmark_path.exists():
        if verbose:
            print(f"Error: Benchmark results not found at {benchmark_path}")
            print("Please run benchmark first using --mode benchmark")
        return
    
    with open(benchmark_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    exit_indices = []
    flops_list = []
    acc_list = []
    labels = []
    
    for i in range(config.num_exit_points):
        key = f'exit_{i}'
        if key in results and results[key]['flops'] is not None and results[key]['accuracy'] is not None:
            exit_indices.append(i)
            flops_list.append(results[key]['flops'])
            acc_list.append(results[key]['accuracy'])
            labels.append(f'Exit {i}')
    
    # Add full model
    if 'full_model' in results and results['full_model']['flops'] is not None:
        flops_full = results['full_model']['flops']
        acc_full = results['full_model']['accuracy']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot exit points
        ax.plot(flops_list, acc_list, 'o-', markersize=10, linewidth=2, 
                label='Early Exits', color='steelblue', alpha=0.7)
        
        # Plot full model
        ax.plot([flops_full], [acc_full], '*', markersize=20, 
                label='Full Model', color='crimson', markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Annotate points
        for i, (flops, acc, lbl) in enumerate(zip(flops_list, acc_list, labels)):
            ax.annotate(lbl, (flops, acc), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, alpha=0.8)
        
        ax.annotate('Full Model', (flops_full, acc_full), xytext=(5, -15), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('FLOPs (MACs)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'FLOPs vs Accuracy - {config.model_name} Early Exit', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Save plot
        plot_dir = Path(paths['output_path']) / 'plots' / 'performance'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in config.plot_formats:
            save_path = plot_dir / f'flops_vs_accuracy.{fmt}'
            fig.savefig(save_path, dpi=config.plot_dpi, bbox_inches='tight')
            if verbose:
                print(f"Saved: {save_path}")
        
        plt.close(fig)


def plot_params_vs_accuracy(
    paths: dict,
    config,
    verbose: bool = False,
):
    """Plot Parameters vs Accuracy for each exit point and full model.
    
    Args:
        paths: Dictionary with output_path
        config: Config dataclass with plotting parameters
        verbose: If True, print detailed progress
    """
    # Load benchmark results
    benchmark_path = Path(paths['output_path']) / 'results' / 'benchmark_results.json'
    if not benchmark_path.exists():
        if verbose:
            print(f"Error: Benchmark results not found at {benchmark_path}")
        return
    
    with open(benchmark_path, 'r') as f:
        results = json.load(f)
    
    # Extract data
    exit_indices = []
    params_list = []
    acc_list = []
    labels = []
    
    for i in range(config.num_exit_points):
        key = f'exit_{i}'
        if key in results and results[key]['params'] is not None and results[key]['accuracy'] is not None:
            exit_indices.append(i)
            params_list.append(results[key]['params'])
            acc_list.append(results[key]['accuracy'])
            labels.append(f'Exit {i}')
    
    # Add full model
    if 'full_model' in results:
        params_full = results['full_model']['params']
        acc_full = results['full_model']['accuracy']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot exit points
        ax.plot(params_list, acc_list, 'o-', markersize=10, linewidth=2, 
                label='Early Exits', color='steelblue', alpha=0.7)
        
        # Plot full model
        ax.plot([params_full], [acc_full], '*', markersize=20, 
                label='Full Model', color='crimson', markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Annotate points
        for i, (params, acc, lbl) in enumerate(zip(params_list, acc_list, labels)):
            ax.annotate(lbl, (params, acc), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, alpha=0.8)
        
        ax.annotate('Full Model', (params_full, acc_full), xytext=(5, -15), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Parameters', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'Parameters vs Accuracy - {config.model_name} Early Exit', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.9)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Save plot
        plot_dir = Path(paths['output_path']) / 'plots' / 'performance'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in config.plot_formats:
            save_path = plot_dir / f'params_vs_accuracy.{fmt}'
            fig.savefig(save_path, dpi=config.plot_dpi, bbox_inches='tight')
            if verbose:
                print(f"Saved: {save_path}")
        
        plt.close(fig)


def plot_param_reduction_curve(
    exit_points: list,
    param_counts: list,
    accuracies: list,
    save_path: str,
    config,
):
    """Plot parameter reduction vs performance curve.

    Args:
        exit_points: List of exit point names (e.g., ['exit_1', 'exit_2', 'final'])
        param_counts: Number of parameters used up to each exit
        accuracies: Accuracy at each exit point
        save_path: Path to save the plot
        config: Config dataclass with plotting parameters
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points
    ax.plot(param_counts, accuracies, 'o-', markersize=8, linewidth=2)
    
    # Annotate points
    for ep, pc, acc in zip(exit_points, param_counts, accuracies):
        ax.annotate(ep, (pc, acc), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Parameter Reduction vs Accuracy', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in config.plot_formats:
        fig.savefig(save_path.replace('.png', f'.{fmt}'), dpi=config.plot_dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_exit_distribution(
    exit_decisions: list,
    save_path: str,
    config,
):
    """Plot distribution of samples exiting at each point.

    Args:
        exit_decisions: List of exit point indices for each sample
        save_path: Path to save the plot
        config: Config dataclass with plotting parameters
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    unique, counts = np.unique(exit_decisions, return_counts=True)
    ax.bar(unique, counts, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Exit Point', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Exit Point Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in config.plot_formats:
        fig.savefig(save_path.replace('.png', f'.{fmt}'), dpi=config.plot_dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_loss_curves(
    history: dict,
    save_path: str,
    config,
):
    """Plot training and validation loss curves for each exit point.

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        config: Config dataclass with plotting parameters
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    ax1.plot(epochs, history['train_loss'], label='Total Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], label='Full Model', linewidth=2)
        
        # Plot exit accuracies if available
        if 'val_exit_acc' in history and len(history['val_exit_acc']) > 0:
            for i in range(len(history['val_exit_acc'][0])):
                exit_accs = [epoch_accs[i] for epoch_accs in history['val_exit_acc']]
                ax2.plot(epochs, exit_accs, label=f'Exit {i}', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in config.plot_formats:
        fig.savefig(save_path.replace('.png', f'.{fmt}'), dpi=config.plot_dpi, bbox_inches='tight')
    
    plt.close(fig)


def plot_comparison_flops_vs_accuracy(
    workspace_path: str,
    project_names: list,
    save_path: str,
    plot_formats: list = ['png', 'pdf'],
    plot_dpi: int = 300,
    verbose: bool = False,
):
    """Compare FLOPs vs Accuracy across multiple projects.
    
    Overlays results from multiple projects (e.g., different exit loss strategies)
    onto a single plot for easy comparison.
    
    Args:
        workspace_path: Path to workspace directory
        project_names: List of project names to compare
        save_path: Path to save the comparison plot
        plot_formats: List of output formats ['png', 'pdf']
        plot_dpi: DPI for saved plots
        verbose: If True, print progress information
    """
    # Define color palette for different projects
    colors = plt.cm.tab10(np.linspace(0, 1, len(project_names)))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, project_name in enumerate(project_names):
        # Load benchmark and evaluation results
        results_dir = Path(workspace_path) / project_name / 'output' / 'results'
        benchmark_path = results_dir / 'benchmark_results.json'
        
        if not benchmark_path.exists():
            if verbose:
                print(f"Warning: Benchmark results not found for project '{project_name}', skipping...")
            continue
        
        with open(benchmark_path, 'r') as f:
            results = json.load(f)
        
        # Extract exit point data
        flops_list = []
        acc_list = []
        exit_labels = []
        
        # Get number of exits from the results
        num_exits = sum(1 for key in results.keys() if key.startswith('exit_'))
        
        for i in range(num_exits):
            key = f'exit_{i}'
            if key in results and results[key].get('flops') is not None and results[key].get('accuracy') is not None:
                flops_list.append(results[key]['flops'])
                acc_list.append(results[key]['accuracy'])
                exit_labels.append(f'E{i}')
        
        # Add full model
        if 'full_model' in results and results['full_model'].get('flops') is not None:
            flops_full = results['full_model']['flops']
            acc_full = results['full_model']['accuracy']
            
            # Plot exit points as connected line
            ax.plot(flops_list, acc_list, 'o-', 
                   color=colors[idx], markersize=8, linewidth=2, 
                   alpha=0.7, label=f'{project_name} (exits)')
            
            # Plot full model as star
            ax.plot([flops_full], [acc_full], '*', 
                   color=colors[idx], markersize=16, 
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=f'{project_name} (full)')
            
            # Optionally annotate exit points for the first project only
            if idx == 0:
                for i, (flops, acc, lbl) in enumerate(zip(flops_list, acc_list, exit_labels)):
                    ax.annotate(lbl, (flops, acc), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.6)
    
    # Formatting
    ax.set_xlabel('FLOPs (MACs)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('FLOPs vs Accuracy - Multi-Project Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in plot_formats:
        output_path = save_path.replace('.png', f'.{fmt}')
        fig.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        if verbose:
            print(f"Saved comparison plot: {output_path}")
    
    plt.close(fig)


def plot_comparison_params_vs_accuracy(
    workspace_path: str,
    project_names: list,
    save_path: str,
    plot_formats: list = ['png', 'pdf'],
    plot_dpi: int = 300,
    verbose: bool = False,
):
    """Compare Parameters vs Accuracy across multiple projects.
    
    Overlays results from multiple projects (e.g., different exit loss strategies)
    onto a single plot for easy comparison.
    
    Args:
        workspace_path: Path to workspace directory
        project_names: List of project names to compare
        save_path: Path to save the comparison plot
        plot_formats: List of output formats ['png', 'pdf']
        plot_dpi: DPI for saved plots
        verbose: If True, print progress information
    """
    # Define color palette for different projects
    colors = plt.cm.tab10(np.linspace(0, 1, len(project_names)))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, project_name in enumerate(project_names):
        # Load benchmark and evaluation results
        results_dir = Path(workspace_path) / project_name / 'output' / 'results'
        benchmark_path = results_dir / 'benchmark_results.json'
        
        if not benchmark_path.exists():
            if verbose:
                print(f"Warning: Benchmark results not found for project '{project_name}', skipping...")
            continue
        
        with open(benchmark_path, 'r') as f:
            results = json.load(f)
        
        # Extract exit point data
        params_list = []
        acc_list = []
        exit_labels = []
        
        # Get number of exits from the results
        num_exits = sum(1 for key in results.keys() if key.startswith('exit_'))
        
        for i in range(num_exits):
            key = f'exit_{i}'
            if key in results and results[key].get('params') is not None and results[key].get('accuracy') is not None:
                params_list.append(results[key]['params'])
                acc_list.append(results[key]['accuracy'])
                exit_labels.append(f'E{i}')
        
        # Add full model
        if 'full_model' in results:
            params_full = results['full_model']['params']
            acc_full = results['full_model']['accuracy']
            
            # Plot exit points as connected line
            ax.plot(params_list, acc_list, 'o-', 
                   color=colors[idx], markersize=8, linewidth=2, 
                   alpha=0.7, label=f'{project_name} (exits)')
            
            # Plot full model as star
            ax.plot([params_full], [acc_full], '*', 
                   color=colors[idx], markersize=16, 
                   markeredgecolor='black', markeredgewidth=0.5,
                   label=f'{project_name} (full)')
            
            # Optionally annotate exit points for the first project only
            if idx == 0:
                for i, (params, acc, lbl) in enumerate(zip(params_list, acc_list, exit_labels)):
                    ax.annotate(lbl, (params, acc), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.6)
    
    # Formatting
    ax.set_xlabel('Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Parameters vs Accuracy - Multi-Project Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    for fmt in plot_formats:
        output_path = save_path.replace('.png', f'.{fmt}')
        fig.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
        if verbose:
            print(f"Saved comparison plot: {output_path}")
    
    plt.close(fig)
