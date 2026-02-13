# Copyright 2026 PHAZE-EE Contributors
# Licensed under the Apache License, Version 2.0

"""
PHAZE-EE (Early Exit for High Energy Physics) main module.

This module serves as the entry point for the PHAZE-EE framework, providing command-line
interface functionality for early exit model workflows. It orchestrates the complete
pipeline from project setup to model training, evaluation, and benchmarking.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401

from .src.utils import ggl


def main():
    """Process command-line arguments to execute PHAZE-EE functionality.

    Parses command-line arguments and executes the appropriate functionality based on the
    specified mode. The available modes are:

    Modes:
        new_project: Create a new project with default configuration.
        train: Train an early exit model using the prepared data.
        evaluate: Evaluate early exit performance across different exit points.
        benchmark: Profile FLOPs and parameters at each exit point.
        plot: Generate visualization plots for parameter reduction vs performance.
        compare: Generate comparison plots across multiple projects in a workspace.
        chain: Execute the full pipeline from project creation to visualization.

    Raises:
        NameError: If the specified mode is not recognized.
    """
    (
        config,
        mode,
        options,
        workspace_name,
        project_name,
        verbose,
        projects,
    ) = ggl.get_arguments()

    # Initialize DDP if configured and multiple GPUs are available
    local_rank = 0
    world_size = 1
    is_ddp_active = False

    if (
        config
        and hasattr(config, "use_ddp")
        and config.use_ddp
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            if world_size > 1:
                print(
                    f"Initializing DDP: RANK {os.environ.get('RANK')}, LOCAL_RANK {local_rank}, WORLD_SIZE {world_size}"
                )
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")
                is_ddp_active = True
                if local_rank == 0 and verbose:
                    print(
                        f"DDP initialized. World size: {world_size}. Running on {torch.cuda.device_count()} GPUs."
                    )
            else:
                if verbose:
                    print("DDP use_ddp is True, but world_size is 1. Running in non-DDP mode.")
                config.use_ddp = False

        except KeyError:
            print("DDP environment variables not set. Running in non-DDP mode.")
            config.use_ddp = False
        except Exception as e:
            print(f"Error initializing DDP: {e}. Running in non-DDP mode.")
            config.use_ddp = False

    # Pass DDP status to config
    if config:
        config.is_ddp_active = is_ddp_active
        config.local_rank = local_rank
        config.world_size = world_size

    # Set CUDNN benchmark for potential speedup
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Define paths dict for the different paths used frequently in the pipeline
    paths = {
        "workspace_path": os.path.join("phaze_ee/workspaces", workspace_name),
        "project_path": os.path.join("phaze_ee/workspaces", workspace_name, project_name),
        "data_path": os.path.join("phaze_ee/workspaces", workspace_name, "data"),
        "output_path": os.path.join(
            "phaze_ee/workspaces", workspace_name, project_name, "output"
        ),
    }

    # MODE DISPATCHER - Call appropriate ggl function based on mode
    if mode == "new_project":
        ggl.create_new_project(workspace_name, project_name, verbose)
    elif mode == "train":
        ggl.run_training(paths, config, verbose)
    elif mode == "evaluate":
        ggl.run_evaluation(paths, config, verbose)
    elif mode == "benchmark":
        ggl.run_benchmark(paths, config, verbose)
    elif mode == "plot":
        ggl.run_plots(paths, config, verbose)
    elif mode == "compare":
        # Validate projects argument
        if not projects or len(projects) < 2:
            raise ValueError(
                "Compare mode requires at least 2 projects. "
                "Use --projects project1 project2 ..."
            )
        ggl.run_comparison(
            workspace_path=paths["workspace_path"],
            project_names=projects,
            output_path=None,  # Will default to workspace/comparisons/
            verbose=verbose,
        )
    elif mode == "chain":
        ggl.run_full_chain(
            workspace_name, project_name, paths, config, options, verbose
        )
    else:
        raise NameError(
            "PHAZE-EE mode "
            + mode
            + " not recognised. Use < phaze_ee --help > to see available modes."
        )

    # Cleanup DDP
    if is_ddp_active:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
