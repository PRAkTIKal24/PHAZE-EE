# PHAZE-EE: Early Exit for High Energy Physics

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-managed-orange)](https://github.com/astral-sh/uv)

**PHAZE-EE** implements early exit mechanisms for ParticleNet on the JetClass dataset, enabling computational efficiency analysis through FLOPs and parameter benchmarking. This provides a foundation for exploring early exit strategies in particle physics deep learning.

## Overview

Early exit networks add intermediate prediction points (exit branches) within a model, allowing predictions at different computational budgets:
- **Computational efficiency**: Trade accuracy for speed by exiting early
- **Parameter efficiency**: Find optimal accuracy vs model size trade-offs
- **Benchmarking**: Compare FLOPs/parameters across exit points

**Current Implementation**: ParticleNet with naive early exits after each EdgeConv block, trained on JetClass 10-class jet classification.

## What's Implemented

✅ **ParticleNet Early Exit Model**
- Linear exit branches after each of 3 EdgeConv blocks
- Wrapper pattern preserves original weaver-core ParticleNet code
- Forward pass returns full model output + all exit outputs

✅ **4 Exit Loss Strategies**
- `mimic_detached`: MSE between exit and full model outputs (detached target)
- `mimic_flow`: MSE with gradient flow to both exit and full model
- `target_detached`: Cross-entropy on exit with detached features
- `target_flow`: Cross-entropy on exit with full gradient flow

✅ **Beta Scheduling**
- Configurable zero epochs (no exit loss) followed by linear/cosine ramp
- Per-exit beta values: `[beta_0, beta_1, beta_2]` for 3 exits
- Total loss: `CE(full_model, labels) + Σ beta_i * exit_loss_i`

✅ **Training Pipeline**
- Ranger optimizer (weaver default)
- Flat + exponential decay scheduler (70% flat, 30% decay)
- AMP (fp16) and DDP multi-GPU support
- Checkpointing and TensorBoard logging

✅ **Evaluation & Benchmarking**
- Accuracy and ROC-AUC per exit point
- FLOPs counting via `weaver.utils.flops_counter`
- Parameter counting per exit point

✅ **Visualization**
- FLOPs vs Accuracy plots
- Parameters vs Accuracy plots
- **NEW**: Comparison mode to overlay multiple projects

✅ **JetClass Dataset Integration**
- YAML data config for 10-class jet classification
- 128 particles, 19 features (pt, energy, eta, phi, etc.)
- Download script for Zenodo dataset

## Installation

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU support)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/PRAkTIKal24/PHAZE-EE.git
cd PHAZE-EE

# Install dependencies
uv sync

# Verify installation
uv run phaze_ee --help
```

### Get JetClass Dataset

Download the JetClass dataset using the provided script:

```bash
bash shell_scripts/download_jetclass.sh
```

Choose from 3 download options:
1. **Full dataset** (2.3 TB): All train/val/test splits
2. **Quick start** (10 GB): Small sample for testing
3. **Custom**: Specify your own file list

The script downloads from [Zenodo (record 6619768)](https://zenodo.org/record/6619768) and organizes files into:
```
phaze_ee/workspaces/test_workspace/data/
├── root/          # Original ROOT files
├── h5/            # Converted to HDF5 (if applicable)
└── parquet/       # Converted to Parquet (if applicable)
```

## Quick Start

This guide walks through training a ParticleNet early exit model on JetClass.

### 1. Create a New Project

```bash
uv run phaze_ee -m new_project -p JetClass my_first_experiment
```

This creates a project structure:
```
phaze_ee/workspaces/JetClass/my_first_experiment/
├── config/
│   └── my_first_experiment_config.py  # Edit this
├── data/                               # Symlink or copy data here
└── output/
    ├── logs/                           # TensorBoard logs
    ├── models/                         # Checkpoints
    ├── plots/                          # Performance plots
    └── results/                        # JSON results
```

### 2. Configure Your Experiment

Edit the config file: `phaze_ee/workspaces/JetClass/my_first_experiment/config/my_first_experiment_config.py`

**Key Configuration Parameters:**

```python
def set_config(c):
    # ===== Data Configuration =====
    c.data_config = 'data_configs/JetClass_full.yaml'
    c.train_files = ['path/to/train/*.root']  # Update with your data paths
    c.val_files = ['path/to/val/*.root']
    
    # ===== Network Architecture =====
    c.network_config = 'network_configs/ParticleNet_JetClass.py'
    c.num_classes = 10  # JetClass has 10 jet types
    c.num_exit_points = 3  # One exit after each EdgeConv block
    
    # ParticleNet architecture params
    c.conv_params = [(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]
    c.fc_params = [(256, 0.1)]
    c.use_fusion = False
    c.use_fts_bn = False
    c.use_counts = True
    
    # ===== Exit Loss Strategy =====
    # Choose from: 'mimic_detached', 'mimic_flow', 'target_detached', 'target_flow'
    c.exit_loss_strategy = 'mimic_detached'
    
    # ===== Beta Scheduling =====
    c.beta_max = [0.3, 0.5, 1.0]  # Max beta for each exit
    c.beta_schedule = 'linear'     # or 'cosine'
    c.zero_epochs = 10             # Epochs before exit loss kicks in
    
    # ===== Training Hyperparameters =====
    c.epochs = 50
    c.lr = 0.002  # Ranger default
    c.batch_size = 512
    c.optimizer = 'ranger'
    c.lr_schedule = 'flat_decay'  # 70% flat, 30% exponential decay
    
    # ===== Hardware Configuration =====
    c.num_workers = 4
    c.use_amp = True   # Mixed precision training
    c.use_ddp = False  # Set True for multi-GPU
```

**Exit Loss Strategies Explained:**

| Strategy | Loss Type | Target | Gradient Flow |
|----------|-----------|--------|---------------|
| `mimic_detached` | MSE | Full model output | Exit only |
| `mimic_flow` | MSE | Full model output | Exit + Full |
| `target_detached` | Cross-Entropy | Ground truth | Exit only |
| `target_flow` | Cross-Entropy | Ground truth | Exit + Full |

**Beta Scheduling:**
- First `zero_epochs`: beta = 0 (no exit loss, train base model)
- Remaining epochs: beta ramps from 0 to `beta_max` (linear or cosine)
- Total loss: `L = CE(full, y) + Σ beta_i * L_exit_i`

### 3. Train the Model

```bash
uv run phaze_ee -m train -p JetClass my_first_experiment -v
```

Output:
- Model checkpoints: `output/models/best_model.pt`, `latest_model.pt`
- TensorBoard logs: `output/logs/train/`
- Training curves: Loss, accuracy per exit point

**Monitor training:**
```bash
tensorboard --logdir phaze_ee/workspaces/JetClass/my_first_experiment/output/logs
```

### 4. Evaluate Performance

```bash
uv run phaze_ee -m evaluate -p JetClass my_first_experiment -v
```

Computes:
- Accuracy per exit point and full model
- ROC-AUC per exit point and full model

Output: `output/results/evaluation_results.json`

### 5. Benchmark FLOPs and Parameters

```bash
uv run phaze_ee -m benchmark -p JetClass my_first_experiment -v
```

Profiles computational complexity using `weaver.utils.flops_counter`:
- FLOPs (MACs) per exit point
- Number of parameters per exit point

Output: `output/results/benchmark_results.json`

### 6. Generate Plots

```bash
uv run phaze_ee -m plot -p JetClass my_first_experiment -v
```

Creates publication-quality plots:
- **FLOPs vs Accuracy**: Shows computational efficiency
- **Parameters vs Accuracy**: Shows model size efficiency

Output: `output/plots/performance/*.{png,pdf}`

### 7. Run Full Pipeline (Recommended)

Instead of running each step separately, chain them:

```bash
uv run phaze_ee -m chain -p JetClass my_first_experiment \
    -o train_evaluate_benchmark_plot -v
```

This runs: train → evaluate → benchmark → plot in sequence.

## Comparing Multiple Projects

One of the key features is the ability to compare different exit loss strategies or hyperparameter configurations side-by-side.

### Manual Comparison Workflow

**Step 1: Create and train multiple projects**

```bash
# Create 4 projects with different exit loss strategies
for strategy in mimic_detached mimic_flow target_detached target_flow; do
    uv run phaze_ee -m new_project -p JetClass "strategy_${strategy}"
    
    # Edit config: Set c.exit_loss_strategy = "${strategy}"
    # Then train, evaluate, benchmark
    uv run phaze_ee -m chain -p JetClass "strategy_${strategy}" \
        -o train_evaluate_benchmark_plot -v
done
```

**Step 2: Generate comparison plots**

```bash
uv run phaze_ee -m compare -p JetClass dummy \
    --projects strategy_mimic_detached strategy_mimic_flow \
               strategy_target_detached strategy_target_flow \
    -v
```

**Output:** Multi-project overlay plots in `phaze_ee/workspaces/JetClass/comparisons/`:
- `comparison_flops_vs_accuracy.{png,pdf}`: All strategies on one FLOPs plot
- `comparison_params_vs_accuracy.{png,pdf}`: All strategies on one params plot

Each project shown in different color, full models marked with stars, exits connected by lines.

### Automated Comparison Script

For convenience, use the provided automation script:

```bash
bash shell_scripts/compare_strategies.sh
```

This script:
1. Creates 4 projects (one per exit loss strategy)
2. Prompts you to configure each project
3. Trains all strategies sequentially
4. Benchmarks all strategies
5. Generates comparison plots
6. Displays summary table

**Use this to quickly answer:** "Which exit loss strategy works best for my dataset?"

## Architecture

```
PHAZE-EE/
├── phaze_ee/
│   ├── __main__.py                          # Entry point
│   ├── phaze_ee.py                          # CLI dispatcher (modes: train, evaluate, benchmark, plot, compare, chain)
│   ├── src/
│   │   ├── models/
│   │   │   ├── weaver_models/               # Original weaver-core models (copied locally)
│   │   │   │   ├── ParticleNet.py           # Unchanged base model
│   │   │   │   ├── ParticleTransformer.py   # (future)
│   │   │   │   └── ParticleNeXt.py          # (future)
│   │   │   └── early_exit/
│   │   │       ├── exit_branches.py         # LinearExitBranch class
│   │   │       ├── particle_net_ee.py       # ParticleNetEE wrapper
│   │   │       └── confidence.py            # (placeholder for dynamic exits)
│   │   ├── trainers/
│   │   │   ├── training.py                  # Full training loop (Ranger, AMP, DDP)
│   │   │   └── exit_loss.py                 # 4 exit loss strategies + BetaScheduler
│   │   └── utils/
│   │       ├── ggl.py                       # Mode orchestration (train, eval, benchmark, plot, compare)
│   │       ├── config.py                    # Configuration dataclass
│   │       ├── plotting.py                  # Single & multi-project plots
│   │       └── benchmark.py                 # FLOPs/params profiling
│   └── workspaces/                          # Experiment workspace root
│       └── <WORKSPACE>/
│           ├── <PROJECT>/
│           │   ├── config/
│           │   │   └── <PROJECT>_config.py
│           │   ├── data/                    # Symlink to dataset
│           │   └── output/
│           │       ├── logs/                # TensorBoard
│           │       ├── models/              # Checkpoints
│           │       ├── plots/               # Performance plots
│           │       └── results/             # JSON results
│           └── comparisons/                 # Multi-project comparison plots
├── network_configs/
│   └── ParticleNet_JetClass.py              # Network config (weaver format)
├── data_configs/
│   └── JetClass_full.yaml                   # Data config (weaver format)
└── shell_scripts/
    ├── download_jetclass.sh                 # Dataset download
    ├── example_workflow.sh                  # Demo workflow
    ├── compare_strategies.sh                # Automated strategy comparison
    └── train_a100.sh                        # SLURM script for HPC
```

### Key Components

**ParticleNetEE Wrapper** ([phaze_ee/src/models/early_exit/particle_net_ee.py](phaze_ee/src/models/early_exit/particle_net_ee.py))
- Wraps original ParticleNet from weaver-core
- Intercepts hidden activations after each EdgeConv block
- Applies LinearExitBranch (pooling + fully connected layer)
- Returns: `(full_output, exit_outputs, exit_features)`

**Exit Loss Strategies** ([phaze_ee/src/trainers/exit_loss.py](phaze_ee/src/trainers/exit_loss.py))
- `ExitLossComputer`: Strategy-based loss computation
- `BetaScheduler`: Linear/cosine ramp scheduling
- Total loss: `L = CE(full, labels) + Σ beta_i(epoch) * L_exit_i`

**Training Loop** ([phaze_ee/src/trainers/training.py](phaze_ee/src/trainers/training.py))
- Mirrors weaver-core's `train_classification` structure
- Ranger optimizer, flat+decay scheduler
- AMP (fp16), DDP, gradient clipping
- Validation every N epochs, checkpointing

**Benchmarking** ([phaze_ee/src/utils/benchmark.py](phaze_ee/src/utils/benchmark.py))
- Uses `weaver.utils.flops_counter.get_model_complexity_info()`
- Profiles each exit point separately via `forward_to_exit()`
- Combines with evaluation metrics (accuracy, AUC)

**Plotting** ([phaze_ee/src/utils/plotting.py](phaze_ee/src/utils/plotting.py))
- Single-project: FLOPs vs accuracy, params vs accuracy
- Multi-project: Overlay comparison plots with color-coded strategies

## Multi-GPU Training

PHAZE-EE supports Distributed Data Parallel (DDP) training for multi-GPU setups.

### Enable DDP in Config

In your project config file:

```python
def set_config(c):
    c.use_ddp = True
    c.use_amp = True  # Recommended with DDP
    c.batch_size = 512  # Per GPU batch size
```

### Launch with torchrun

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 -m phaze_ee -m train -p JetClass my_experiment -v

# Multi-node (2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=<RANK> \
    --master_addr=<MASTER_IP> --master_port=29500 \
    -m phaze_ee -m train -p JetClass my_experiment -v
```

### SLURM Integration

Use the provided SLURM script for HPC clusters:

```bash
# Edit shell_scripts/train_a100.sh with your settings
sbatch shell_scripts/train_a100.sh
```

**Key SLURM Variables:**
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

srun torchrun --nproc_per_node=4 -m phaze_ee -m train ...
```

## CLI Reference

### Modes

```bash
# Create new project
phaze_ee -m new_project -p <WORKSPACE> <PROJECT>

# Train model
phaze_ee -m train -p <WORKSPACE> <PROJECT> [-v]

# Evaluate model
phaze_ee -m evaluate -p <WORKSPACE> <PROJECT> [-v]

# Benchmark FLOPs/params
phaze_ee -m benchmark -p <WORKSPACE> <PROJECT> [-v]

# Generate plots
phaze_ee -m plot -p <WORKSPACE> <PROJECT> [-v]

# Compare multiple projects
phaze_ee -m compare -p <WORKSPACE> dummy --projects proj1 proj2 proj3 [-v]

# Chain multiple operations
phaze_ee -m chain -p <WORKSPACE> <PROJECT> -o train_evaluate_benchmark_plot [-v]
```

### Arguments

- `-m, --mode`: Operation mode (`new_project`, `train`, `evaluate`, `benchmark`, `plot`, `compare`, `chain`)
- `-p, --project`: Workspace and project names (2 arguments)
- `--projects`: List of project names for comparison (compare mode only)
- `-o, --operations`: Operations to chain (chain mode only)
- `-v, --verbose`: Verbose output
- `-h, --help`: Show help message

### Example Workflows

**Basic single experiment:**
```bash
uv run phaze_ee -m new_project -p JetClass exp1
# Edit config...
uv run phaze_ee -m chain -p JetClass exp1 -o train_evaluate_benchmark_plot -v
```

**Compare beta schedules:**
```bash
for beta in linear cosine; do
    uv run phaze_ee -m new_project -p JetClass beta_${beta}
    # Edit config: c.beta_schedule = '${beta}'
    uv run phaze_ee -m chain -p JetClass beta_${beta} -o train_evaluate_benchmark_plot -v
done

uv run phaze_ee -m compare -p JetClass dummy --projects beta_linear beta_cosine -v
```

**Hyperparameter sweep:**
```bash
for lr in 1e-3 2e-3 5e-3; do
    project="lr_${lr//[.-]/_}"  # lr_1e_3, lr_2e_3, etc.
    uv run phaze_ee -m new_project -p JetClass ${project}
    # Edit config: c.lr = ${lr}
    uv run phaze_ee -m chain -p JetClass ${project} -o train_evaluate_benchmark_plot -v
done

uv run phaze_ee -m compare -p JetClass dummy --projects lr_1e_3 lr_2e_3 lr_5e_3 -v
```

## Configuration Reference

All configuration is done via Python config files in `<workspace>/<project>/config/<project>_config.py`.

### Configuration Sections

#### Data Configuration
```python
c.data_config = 'data_configs/JetClass_full.yaml'  # Path to YAML data config
c.train_files = ['path/to/train/*.root']            # Training files
c.val_files = ['path/to/val/*.root']                # Validation files
```

#### Network Architecture
```python
c.network_config = 'network_configs/ParticleNet_JetClass.py'
c.num_classes = 10
c.num_exit_points = 3

# ParticleNet specific
c.conv_params = [(k, (c1, c2, c3)), ...]  # [(16, (64,64,64)), (16, (128,128,128)), (16, (256,256,256))]
c.fc_params = [(hidden_dim, dropout), ...]  # [(256, 0.1)]
c.use_fusion = False
c.use_fts_bn = False
c.use_counts = True
```

#### Exit Loss Configuration
```python
c.exit_loss_strategy = 'mimic_detached'  # 'mimic_detached', 'mimic_flow', 'target_detached', 'target_flow'
c.beta_max = [0.3, 0.5, 1.0]              # Max beta per exit
c.beta_schedule = 'linear'                # 'linear' or 'cosine'
c.zero_epochs = 10                        # Epochs before exit loss starts
```

#### Training Hyperparameters
```python
c.epochs = 50
c.lr = 0.002
c.batch_size = 512
c.optimizer = 'ranger'       # Currently only 'ranger' supported
c.lr_schedule = 'flat_decay' # 70% flat, 30% exponential decay
c.weight_decay = 0.01
c.gradient_clip = 1.0
```

#### Hardware & Performance
```python
c.num_workers = 4
c.use_amp = True      # Mixed precision (fp16)
c.use_ddp = False     # Distributed training
c.device = 'cuda'     # 'cuda' or 'cpu'
```

#### Logging & Checkpointing
```python
c.log_interval = 100        # Log every N batches
c.val_interval = 1          # Validate every N epochs
c.save_interval = 5         # Save checkpoint every N epochs
c.tensorboard = True        # Enable TensorBoard logging
```

## Integration with weaver-core

PHAZE-EE uses [weaver-core](https://github.com/hqucms/weaver-core) (v0.4.17+) for:

1. **Data Pipeline**
   - `weaver.utils.data.config.DataConfig`: YAML-based data configuration
   - `weaver.utils.dataset.SimpleIterDataset`: Efficient data loading
   - Preprocessing, augmentation, batching

2. **Base Models**
   - ParticleNet architecture copied from weaver-core
   - Preserved in `phaze_ee/src/models/weaver_models/ParticleNet.py`
   - Wrapped (not modified) by `ParticleNetEE`

3. **Training Utilities**
   - Ranger optimizer from `weaver.utils.nn.optimizer_factory`
   - Learning rate schedulers

4. **Benchmarking**
   - `weaver.utils.flops_counter.get_model_complexity_info()` for FLOPs counting
   - Returns MACs (multiply-accumulate operations) and parameter counts

### Why Copy weaver Models?

Models are **copied** into `phaze_ee/src/models/weaver_models/` rather than imported to:
- Preserve original weaver-core functionality
- Enable local modifications if needed (though currently unchanged)
- Maintain version stability
- Allow the wrapper pattern to intercept intermediate activations

## Troubleshooting

### Common Issues

**1. ImportError: weaver not found**
```bash
# Install weaver-core
uv pip install weaver-core>=0.4.17
```

**2. Data not found errors**
```python
# Update paths in config file
c.train_files = ['/absolute/path/to/train/*.root']
c.val_files = ['/absolute/path/to/val/*.root']
```

**3. CUDA out of memory**
```python
# Reduce batch size in config
c.batch_size = 256  # or 128
```

**4. Comparison plots missing projects**
```bash
# Ensure all projects have been benchmarked
uv run phaze_ee -m benchmark -p <WORKSPACE> <PROJECT> -v
```

**5. TensorBoard not showing logs**
```bash
# Check logs directory exists
ls phaze_ee/workspaces/<WORKSPACE>/<PROJECT>/output/logs/

# Launch TensorBoard with correct path
tensorboard --logdir phaze_ee/workspaces/<WORKSPACE>/<PROJECT>/output/logs
```

### Performance Tips

1. **Use AMP for faster training**: Set `c.use_amp = True`
2. **Increase batch size on large GPUs**: `c.batch_size = 1024` or higher
3. **Use DDP for multi-GPU**: Set `c.use_ddp = True`
4. **Adjust num_workers**: `c.num_workers = <num_cpu_cores>`
5. **Start with zero_epochs=10**: Stabilizes full model before exit training

## Extending PHAZE-EE

### Adding New Exit Branches

Edit `phaze_ee/src/models/early_exit/exit_branches.py`:

```python
class CustomExitBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return logits
```

### Adding New Exit Loss Strategies

Edit `phaze_ee/src/trainers/exit_loss.py`:

```python
class ExitLossStrategy(Enum):
    # ... existing strategies ...
    CUSTOM_STRATEGY = "custom_strategy"

# In ExitLossComputer.compute_exit_loss():
elif strategy == ExitLossStrategy.CUSTOM_STRATEGY:
    # Your loss computation
    loss = ...
```

### Supporting New Architectures

1. Copy base model to `phaze_ee/src/models/weaver_models/`
2. Create wrapper in `phaze_ee/src/models/early_exit/`
3. Implement `forward_to_exit()` and `get_num_parameters_to_exit()` methods
4. Update `create_model()` factory in `ggl.py`

## Results & Benchmarks

After running the full pipeline, you'll have comprehensive benchmarking data for your early exit model.

### Example Output Structure

```
output/
├── models/
│   ├── best_model.pt          # Best validation accuracy
│   └── latest_model.pt        # Latest checkpoint
├── logs/
│   └── train/                 # TensorBoard logs
├── plots/
│   ├── performance/
│   │   ├── flops_vs_accuracy.png
│   │   ├── flops_vs_accuracy.pdf
│   │   ├── params_vs_accuracy.png
│   │   └── params_vs_accuracy.pdf
│   ├── loss/                  # (if implemented)
│   ├── exit_distribution/     # (if implemented)
│   └── param_reduction/       # (if implemented)
└── results/
    ├── evaluation_results.json
    └── benchmark_results.json
```

### Benchmark Results Format

`benchmark_results.json`:
```json
{
  "exit_0": {
    "flops": 1234567,
    "flops_str": "1.23 MMac",
    "params": 50000,
    "params_str": "0.05M",
    "accuracy": 0.75,
    "roc_auc": 0.95
  },
  "exit_1": { ... },
  "exit_2": { ... },
  "full_model": {
    "flops": 9876543,
    "flops_str": "9.88 MMac",
    "params": 500000,
    "params_str": "0.50M",
    "accuracy": 0.92,
    "roc_auc": 0.99
  }
}
```

### Interpreting Results

**FLOPs vs Accuracy Plot:**
- X-axis: FLOPs (log scale) - computational cost
- Y-axis: Accuracy (%) - model performance
- Points closer to top-left are better (high accuracy, low cost)
- Look for "knee" in curve - best accuracy/cost trade-off

**Parameters vs Accuracy Plot:**
- X-axis: Parameters (log scale) - model size
- Y-axis: Accuracy (%) - model performance
- Useful for deployment scenarios with memory constraints

**Comparison Plots:**
- Multiple strategies overlaid on same plot
- Identify Pareto frontier: best strategies for each budget
- Different colors = different strategies
- Stars = full model, circles = exit points

## Project Status

**Current Implementation (v0.1):**
- ✅ ParticleNet with 3 early exit points
- ✅ 4 exit loss strategies (mimic/target × detached/flow)
- ✅ Beta scheduling (linear/cosine ramp)
- ✅ Full training pipeline (Ranger, AMP, DDP)
- ✅ Evaluation and benchmarking
- ✅ Single & multi-project plotting
- ✅ JetClass dataset integration

**Planned Features:**
- ⏳ Dynamic early exit at inference (confidence/entropy thresholds)
- ⏳ ParticleTransformer early exit implementation
- ⏳ ParticleNeXt early exit implementation
- ⏳ Learned exit decision modules
- ⏳ Exit point optimization (where to place exits)
- ⏳ Distillation-based training
- ⏳ Unit and integration tests

**Research Questions to Explore:**
1. Which exit loss strategy generalizes best?
2. How does beta scheduling affect exit performance?
3. What's the optimal number of exit points?
4. Do exits help with out-of-distribution detection?
5. Can we predict which samples will exit early?

## Citation

If you use PHAZE-EE in your research, please cite:

```bibtex
@software{phaze_ee,
  title = {PHAZE-EE: Early Exit Mechanisms for High Energy Physics},
  author = {PHAZE-EE Contributors},
  year = {2026},
  url = {https://github.com/PRAkTIKal24/PHAZE-EE}
}
```

Also cite the weaver-core framework:
```bibtex
@software{weaver,
  title = {Weaver: A machine learning framework for HEP},
  author = {Qu, Huilin and Li, Congqiao and Qian, Sitian},
  url = {https://github.com/hqucms/weaver-core}
}
```

For JetClass dataset:
```bibtex
@dataset{jetclass,
  author = {Qu, Huilin and Li, Congqiao and Qian, Sitian},
  title = {JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics},
  year = {2022},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.6619768},
  url = {https://doi.org/10.5281/zenodo.6619768}
}
```

## Related Projects

- [PHAZE](https://github.com/PRAkTIKal24/PHAZE) - Fast inference framework for HEP ML
- [BEAD](https://github.com/PRAkTIKal24/BEAD) - Anomaly detection with VAEs for HEP
- [weaver-core](https://github.com/hqucms/weaver-core) - ML framework for particle physics
- [BranchyNet](https://arxiv.org/abs/1709.01686) - Early exit networks (original paper)

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas of interest:
- New early exit strategies
- Support for additional architectures
- Improved benchmarking tools
- Documentation improvements
- Bug fixes

**Contribution Workflow:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

This project builds upon:
- [weaver-core](https://github.com/hqucms/weaver-core) for data pipeline, base models, and FLOPs counting
- [BEAD](https://github.com/PRAkTIKal24/BEAD) for repository structure and uv setup patterns
- JetClass dataset from Zenodo (record 6619768)
- PyTorch ecosystem for deep learning infrastructure

## Contact

For questions, issues, or discussions:
- **GitHub Issues**: [PHAZE-EE Issues](https://github.com/PRAkTIKal24/PHAZE-EE/issues)
- **Discussions**: [PHAZE-EE Discussions](https://github.com/PRAkTIKal24/PHAZE-EE/discussions)

---

**Happy benchmarking! 🚀**
