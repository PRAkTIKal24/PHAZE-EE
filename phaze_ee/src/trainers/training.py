# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
Training module for early exit models.

This module implements the training loop for early exit models, handling
multi-exit loss computation and distributed training. Mirrors weaver-core's
training pipeline but with early exit functionality.
"""

import os  # noqa: F401
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np  # noqa: F401
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from phaze_ee.src.trainers.exit_loss import (
    BetaScheduler,
    ExitLossStrategy,
    compute_multi_exit_loss,
)


def train_early_exit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    paths: dict,
    verbose: bool = False,
):
    """Train an early exit model.

    Implements training loop mirroring weaver-core's train_classification but
    with early exit loss computation and beta scheduling.

    Args:
        model: Early exit model (e.g., ParticleNetEE)
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Config dataclass with training parameters
        paths: Dictionary with output paths (workspace_path, project_path, output_path)
        verbose: If True, print detailed progress

    Returns:
        dict: Training history (losses, metrics per epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)

    # Setup DDP if enabled
    if config.use_ddp and config.is_ddp_active:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )
        is_main_process = config.local_rank == 0
    else:
        is_main_process = True

    # Setup optimizer (following weaver-core defaults)
    if config.optimizer == "ranger":
        try:
            from weaver.utils.nn.optimizer.ranger import Ranger

            optimizer = Ranger(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        except ImportError:
            print("Warning: Ranger optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Setup learning rate scheduler (following weaver-core)
    if config.scheduler == "flat+decay":
        # 70% flat, 30% exponential decay to 1% of initial LR
        decay_start_epoch = int(0.7 * config.epochs)
        decay_epochs = config.epochs - decay_start_epoch
        gamma = (0.01) ** (1.0 / decay_epochs) if decay_epochs > 0 else 1.0

        def lr_lambda(epoch):
            if epoch < decay_start_epoch:
                return 1.0
            else:
                return gamma ** (epoch - decay_start_epoch)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # Setup AMP
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    # Setup loss criterion
    criterion = nn.CrossEntropyLoss()

    # Setup beta scheduler
    beta_scheduler = BetaScheduler(
        beta_max=config.beta_max,
        total_epochs=config.epochs,
        zero_epochs=config.beta_zero_epochs,
        ramp_type=config.beta_ramp_type,
        per_exit=True,
    )

    # Setup TensorBoard
    if is_main_process:
        log_dir = Path(paths["output_path"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
    else:
        writer = None

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_exit_acc": [],
        "learning_rate": [],
        "betas": [],
    }

    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Get current betas
        betas = beta_scheduler.get_betas(epoch, config.num_exit_points)

        # Train one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            betas=betas,
            config=config,
            device=device,
            scaler=scaler,
            epoch=epoch,
            verbose=verbose and is_main_process,
        )

        # Validate
        if (epoch + 1) % config.val_interval == 0:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                betas=betas,
                config=config,
                device=device,
                verbose=verbose and is_main_process,
            )
        else:
            val_metrics = {}

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log to history
        history["train_loss"].append(train_metrics["total_loss"])
        history["learning_rate"].append(current_lr)
        history["betas"].append(betas.copy())

        if val_metrics:
            history["val_loss"].append(val_metrics["total_loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_exit_acc"].append(val_metrics["exit_accuracies"])

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_metrics["total_loss"], epoch)
            writer.add_scalar("Loss/train_base", train_metrics["base_loss"], epoch)
            writer.add_scalar("Loss/train_exit", train_metrics["exit_loss"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

            for i, beta in enumerate(betas):
                writer.add_scalar(f"Beta/exit_{i}", beta, epoch)

            if val_metrics:
                writer.add_scalar("Loss/val", val_metrics["total_loss"], epoch)
                writer.add_scalar("Accuracy/val_full", val_metrics["accuracy"], epoch)
                for i, acc in enumerate(val_metrics["exit_accuracies"]):
                    writer.add_scalar(f"Accuracy/val_exit_{i}", acc, epoch)

        # Print progress
        if is_main_process and verbose:
            epoch_time = time.time() - epoch_start
            log_str = f"Epoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s) - "
            log_str += f"train_loss: {train_metrics['total_loss']:.4f} - "
            log_str += f"lr: {current_lr:.6f} - "
            log_str += f"betas: [{', '.join([f'{b:.3f}' for b in betas])}]"

            if val_metrics:
                log_str += f" - val_loss: {val_metrics['total_loss']:.4f} - "
                log_str += f"val_acc: {val_metrics['accuracy']:.4f}"

            print(log_str)

        # Save checkpoints
        if is_main_process:
            # Save best model
            if val_metrics and val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={"val_acc": best_val_acc},
                    path=Path(paths["output_path"]) / "models" / "best_model.pt",
                    config=config,
                )

            # Save periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=val_metrics if val_metrics else {},
                    path=Path(paths["output_path"]) / "models" / f"checkpoint_epoch_{epoch + 1}.pt",
                    config=config,
                )

    # Save final model
    if is_main_process:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.epochs - 1,
            metrics=val_metrics if val_metrics else {},
            path=Path(paths["output_path"]) / "models" / "final_model.pt",
            config=config,
        )

        if writer is not None:
            writer.close()

        if verbose:
            print("\nTraining complete!")
            print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")

    return history


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    betas: list,
    config,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_base_loss = 0.0
    total_exit_loss = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not verbose)

    for batch_idx, (X, y, _) in enumerate(pbar):
        # Prepare inputs (following weaver-core data loading pattern)
        # X is dict of tensors: {input_name: tensor}
        # y is dict: {label_name: tensor}

        # Get input tensors (assuming data_config.input_names order: pf_points, pf_features, pf_vectors, pf_mask)
        points = X.get("pf_points", X.get("points")).to(device)
        features = X.get("pf_features", X.get("features")).to(device)
        mask = X.get("pf_mask", X.get("mask", None))
        if mask is not None:
            mask = mask.to(device)

        # Get labels
        labels = y.get("_label_", y.get("label")).long().to(device)

        # Forward pass with AMP
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Forward through early exit model
            full_output, exit_outputs, exit_features = model(
                points, features, mask, return_exit_outputs=True
            )

            # Compute multi-exit loss
            loss, loss_dict = compute_multi_exit_loss(
                full_output=full_output,
                exit_outputs=exit_outputs,
                exit_features=exit_features,
                labels=labels,
                betas=betas,
                strategy=ExitLossStrategy(config.exit_loss_strategy),
                base_criterion=criterion,
            )

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        # Accumulate losses
        total_loss += loss_dict["total"]
        total_base_loss += loss_dict["base"]
        total_exit_loss += loss_dict["exit_total"]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": loss_dict["total"],
                "base": loss_dict["base"],
                "exit": loss_dict["exit_total"],
            }
        )

        # Limit samples per epoch if configured
        if config.samples_per_epoch is not None:
            samples_seen = (batch_idx + 1) * config.batch_size
            if samples_seen >= config.samples_per_epoch:
                break

    return {
        "total_loss": total_loss / num_batches,
        "base_loss": total_base_loss / num_batches,
        "exit_loss": total_exit_loss / num_batches,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    betas: list,
    config,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    exit_correct = [0] * config.num_exit_points

    with torch.no_grad():
        for X, y, _ in val_loader:
            # Prepare inputs
            points = X.get("pf_points", X.get("points")).to(device)
            features = X.get("pf_features", X.get("features")).to(device)
            mask = X.get("pf_mask", X.get("mask", None))
            if mask is not None:
                mask = mask.to(device)

            labels = y.get("_label_", y.get("label")).long().to(device)

            # Forward pass
            full_output, exit_outputs, exit_features = model(
                points, features, mask, return_exit_outputs=True
            )

            # Compute loss
            loss, _ = compute_multi_exit_loss(
                full_output=full_output,
                exit_outputs=exit_outputs,
                exit_features=exit_features,
                labels=labels,
                betas=betas,
                strategy=ExitLossStrategy(config.exit_loss_strategy),
                base_criterion=criterion,
            )

            total_loss += loss.item() * labels.size(0)

            # Compute accuracy for full model
            preds = full_output.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Compute accuracy for each exit
            for i, exit_out in enumerate(exit_outputs):
                exit_preds = exit_out.argmax(dim=1)
                exit_correct[i] += (exit_preds == labels).sum().item()

    accuracy = total_correct / total_samples
    exit_accuracies = [ec / total_samples for ec in exit_correct]

    return {
        "total_loss": total_loss / total_samples,
        "accuracy": accuracy,
        "exit_accuracies": exit_accuracies,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: dict,
    path: Path,
    config,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Extract model state (handle DDP wrapper)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {
            "model_name": config.model_name,
            "num_classes": config.num_classes,
            "num_exit_points": config.num_exit_points,
            "conv_params": config.conv_params,
            "fc_params": config.fc_params,
            "exit_loss_strategy": config.exit_loss_strategy,
        },
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None,
) -> dict:
    """Load model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    # Load model state (handle DDP wrapper)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
