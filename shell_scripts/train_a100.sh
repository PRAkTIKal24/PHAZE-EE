#!/bin/bash
#SBATCH --job-name=phaze_ee_train
#SBATCH --output=logs/phaze_ee_%j.out
#SBATCH --error=logs/phaze_ee_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Configuration
WORKSPACE_NAME="JetClass"
PROJECT_NAME="my_experiment"
MODE="train"  # Options: train, evaluate, benchmark, plot, chain

# Environment setup
module purge
module load cuda/12.6.0
module load python/3.11

# Activate virtual environment (if using venv instead of uv directly)
# source .venv/bin/activate

# Print GPU info
echo "Running on host: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Set distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with torchrun for DDP
echo "Starting PHAZE-EE training with DDP..."
echo "Workspace: $WORKSPACE_NAME"
echo "Project: $PROJECT_NAME"
echo "Mode: $MODE"
echo "World Size: $WORLD_SIZE"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m phaze_ee \
    -m $MODE \
    -p $WORKSPACE_NAME $PROJECT_NAME \
    -v

echo "Job completed!"
