# PHAZE-EE: Early Exit for High Energy Physics

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![uv](https://img.shields.io/badge/uv-managed-orange)](https://github.com/astral-sh/uv)

**PHAZE-EE** implements early exit mechanisms for deep learning models used in high energy physics, enabling fast inference through strategic early exits at multiple decision points within neural networks. This work is part of the [PHAZE framework](https://github.com/PRAkTIKal24/PHAZE) for optimized particle physics ML inference.

## Overview

Early exit networks allow models to make predictions at intermediate layers, enabling:
- **Faster inference**: Simple samples exit early, avoiding full network computation
- **Parameter efficiency**: Identify optimal accuracy-parameter trade-offs
- **Adaptive computation**: Allocate resources based on sample difficulty

This repository explores early exit strategies for models like ParticleNet, ParticleTransformer, and ParticleNeXt used in particle physics applications (e.g., jet tagging).

## Features

- 🚀 Early exit implementations for weaver-core models
- 📊 Automatic benchmarking of FLOPs and parameters per exit point
- 📈 Parameter reduction vs performance analysis
- 🔧 Multiple exit strategies: confidence-based, entropy-based, learned
- 🖥️ Multi-GPU training with DDP and AMP support
- 📦 uv-based dependency management (mirrors [BEAD](https://github.com/PRAkTIKal24/BEAD))

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

# Install dependencies with uv
uv sync

# Optional: Install GPU acceleration extras
uv pip install -e ".[gpu]"

# Optional: Install visualization extras
uv pip install -e ".[viz]"
```

## Quick Start

### 1. Create a New Project

```bash
uv run phaze_ee -m new_project -p JetClass my_first_test
```

This creates:
```
phaze_ee/workspaces/JetClass/my_first_test/
├── config/
│   └── my_first_test_config.py
├── data/
└── output/
    ├── models/
    ├── plots/
    └── results/
```

### 2. Configure Your Experiment

Edit `phaze_ee/workspaces/JetClass/my_first_test/config/my_first_test_config.py`:

```python
def set_config(c):
    # Model configuration
    c.model_name = "ParticleNet"
    c.num_classes = 10
    c.num_exit_points = 3
    
    # Early exit configuration
    c.exit_loss_weights = [0.3, 0.5, 1.0]
    c.exit_threshold = 0.9
    c.exit_strategy = "confidence"
    
    # Training configuration
    c.epochs = 100
    c.lr = 1e-3
    c.batch_size = 512
    c.use_amp = True
```

### 3. Train Model

```bash
uv run phaze_ee -m train -p JetClass my_first_test -v
```

### 4. Evaluate and Benchmark

```bash
# Evaluate early exit performance
uv run phaze_ee -m evaluate -p JetClass my_first_test

# Benchmark FLOPs/parameters
uv run phaze_ee -m benchmark -p JetClass my_first_test

# Generate plots
uv run phaze_ee -m plot -p JetClass my_first_test
```

### 5. Run Full Pipeline

```bash
uv run phaze_ee -m chain -p JetClass my_first_test -o train_evaluate_benchmark_plot -v
```

## Architecture

```
PHAZE-EE/
├── phaze_ee/
│   ├── __main__.py              # Entry point
│   ├── phaze_ee.py              # Main CLI dispatcher
│   ├── src/
│   │   ├── models/
│   │   │   ├── weaver_models/   # Modified weaver models
│   │   │   │   ├── ParticleNet.py
│   │   │   │   ├── ParticleTransformer.py
│   │   │   │   └── ParticleNeXt.py
│   │   │   └── early_exit/      # Early exit utilities
│   │   │       ├── exit_branches.py
│   │   │       └── confidence.py
│   │   ├── trainers/
│   │   │   └── training.py
│   │   └── utils/
│   │       ├── ggl.py           # Orchestration layer
│   │       ├── config.py        # Configuration dataclass
│   │       ├── plotting.py
│   │       └── benchmark.py
│   └── workspaces/              # Experiment organization
├── network_configs/             # weaver-compatible network configs
├── data_configs/                # YAML data configurations
└── shell_scripts/               # SLURM scripts
```

## Multi-GPU Training

For distributed training with DDP:

```bash
# In your config file
c.use_ddp = True
c.use_amp = True

# Launch with torchrun
torchrun --nproc_per_node=4 -m phaze_ee -m train -p JetClass my_test
```

Or use the provided SLURM script:

```bash
sbatch shell_scripts/train_a100.sh
```

## Early Exit Strategies

### Confidence-Based
Exit when max softmax probability exceeds threshold:
```python
c.exit_strategy = "confidence"
c.exit_threshold = 0.9
```

### Entropy-Based
Exit when prediction entropy is below threshold:
```python
c.exit_strategy = "entropy"
c.exit_threshold = 0.5
```

### Learned
Train a learned exit decision module:
```python
c.exit_strategy = "learned"
```

## Integration with weaver-core

PHAZE-EE uses [weaver-core](https://github.com/hqucms/weaver-core) for:
- Data pipeline (`weaver.utils.dataset`)
- Performance benchmarking (`weaver.utils.flops_counter`)
- Base model architectures (copied and modified locally)

Models from weaver-core are copied into `phaze_ee/src/models/weaver_models/` to enable direct modification for early exit insertion.

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

## Related Projects

- [PHAZE](https://github.com/PRAkTIKal24/PHAZE) - Fast inference framework for HEP ML
- [BEAD](https://github.com/PRAkTIKal24/BEAD) - Anomaly detection with VAEs
- [weaver-core](https://github.com/hqucms/weaver-core) - ML framework for particle physics

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project builds upon:
- [weaver-core](https://github.com/hqucms/weaver-core) for data pipeline and base models
- [BEAD](https://github.com/PRAkTIKal24/BEAD) for repository structure and uv setup patterns
