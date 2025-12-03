# SAE Music Interpretability

Sparse Autoencoder (SAE) interpretability tools for music generation models. Extract, analyze, and label interpretable features from MusicGen transformer layers.

## Overview

This project implements Sparse Autoencoders to discover monosemantic features in music generation models. It supports multiple SAE architectures optimized for interpretability research on MusicGen.

**Key Features:**
- Extract activations from any MusicGen layer
- Train multiple SAE variants (Vanilla, TopK, BatchTopK, Matryoshka)
- Automatic feature discovery and labeling (with optional LLM)
- Memory-efficient processing for large datasets
- Handles variable-length sequences

## Installation

```bash
# Clone repository
git clone https://github.com/zhangsiyu1103/sae-music-interp.git
cd sae-music-interp

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)
- 16GB+ RAM for training
- Optional: Anthropic/OpenAI API key for LLM labeling

## Quick Start

### 1. Download Dataset (Optional)

Download MusicCaps prompts for training:

```bash
python -m src.data.download
```

Outputs: `data/musiccaps_prompts_{quick_test,medium,full}.txt`

### 2. Extract Activations

Extract activations from MusicGen layers:

```bash
# Single layer
python -m src.data.extract \
    --model facebook/musicgen-medium \
    --layers 12 \
    --prompt_file data/musiccaps_prompts_medium.txt \
    --batch_size 64 \
    --duration 10.0

# Multiple layers
python -m src.data.extract \
    --layers 12,16,18 \
    --prompt_file data/musiccaps_prompts_medium.txt
```

Outputs: `data/activations/activations_layer{N}.pt`

### 3. Train SAE

Train a Sparse Autoencoder on extracted activations:

```bash
# BatchTopK SAE (recommended)
python -m src.training.train \
    --sae_type batch_topk \
    --layer_idx 12 \
    --hidden_dim 6144 \
    --top_k 64 \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4

# Vanilla SAE with L1 sparsity
python -m src.training.train \
    --sae_type vanilla \
    --layer_idx 12 \
    --hidden_dim 6144 \
    --sparsity_coef 0.05
```

Outputs: `models/{sae_type}/layer{N}/dim{D}/sae_best.pt`

### 4. Discover Features

Discover and label the top 10 most prominent features:

```bash
# Without LLM (heuristic labeling)
python -m src.analysis.discover \
    --sae_path models/batch_topk/layer12/dim6144/sae_best.pt \
    --activation_file data/activations/activations_layer12.pt \
    --output results/features_layer12.json \
    --min_activation_count 10 \
    --max_activation_count 10000 \
    --max_features 10

# With LLM labeling (Anthropic)
python -m src.analysis.discover \
    --sae_path models/batch_topk/layer12/dim6144/sae_best.pt \
    --activation_file data/activations/activations_layer12.pt \
    --output results/features_layer12.json \
    --use_llm \
    --llm_api anthropic
```

**Features are ranked by maximum activation value. By default, saves top 10 features with 20 examples each.**

Outputs: `results/features_layer{N}.json`

## SAE Architectures

### Vanilla SAE
Standard autoencoder with L1 sparsity penalty.
- **Pros**: Simple, interpretable
- **Cons**: Requires careful tuning of sparsity coefficient

### TopKSAE
Per-sample TopK activation selection.
- **Pros**: Explicit sparsity control, no L1 tuning
- **Cons**: Fixed k per sample

### BatchTopKSAE (Recommended)
Global batch-wise TopK with learned inference threshold.
- **Pros**: Better feature utilization, adaptive threshold
- **Cons**: Slightly more complex

### Matryoshka SAE
Hierarchical feature groups with nested structure.
- **Pros**: Multi-scale feature learning
- **Cons**: More memory, experimental

## Project Structure

```
sae-music-interp/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
│
├── scripts/                       # Batch operation scripts
│   ├── run_discover.sh           # Batch feature discovery
│   └── train_sae_layers_10_16.sh # Batch SAE training
│
├── src/                           # Source code (organized by function)
│   │
│   ├── data/                     # Data operations
│   │   ├── download.py           # Download MusicCaps dataset
│   │   └── extract.py            # Extract activations from MusicGen
│   │
│   ├── training/                 # Model training
│   │   └── train.py              # Train SAE models
│   │
│   ├── analysis/                 # Analysis and interpretation
│   │   ├── discover.py           # Feature discovery CLI
│   │   └── feature_discovery.py  # Feature discovery library
│   │
│   └── models/                   # Model implementations
│       ├── sae.py                # SAE architectures (Vanilla, TopK, BatchTopK, Matryoshka)
│       └── musicgen_wrapper.py   # MusicGen wrapper for activation extraction
│
├── data/                          # Data files
│   ├── musiccaps_prompts_*.txt   # Prompt datasets (quick_test, medium, full)
│   └── activations/              # Extracted activations (.pt)
│       └── activations_layer{N}.pt
│
├── models/                        # Trained SAE checkpoints
│   └── {sae_type}/layer{N}/dim{D}/
│       ├── sae_best.pt           # Best validation checkpoint
│       ├── sae_final.pt          # Final epoch checkpoint
│       ├── training_curves.png   # Loss curves
│       └── training_stats.json   # Training metrics
│
├── results/                       # Analysis outputs
│   └── features_layer{N}.json    # Top 10 discovered features with 20 examples each
│
└── tests/                         # Unit tests
```

## Training Configuration

### Memory Optimization
- **Large files auto-sampled** (>2GB activations)
- **GPU memory cleared** after each epoch
- **Gradient clipping** for stability
- **Batch processing** for feature discovery

### Training Features
- **Warmup + cosine annealing** learning rate schedule
- **Early stopping** with validation monitoring
- **Dead feature resurrection** via auxiliary loss
- **Decoder weight normalization** for TopK variants

### Recommended Settings

| Dataset Size | Batch Size | Hidden Dim | Top-K | Epochs |
|-------------|-----------|-----------|-------|--------|
| Small (<1K) | 128 | 2048 | 32 | 50 |
| Medium (1-10K) | 256 | 4096 | 64 | 30 |
| Large (>10K) | 512 | 6144 | 64 | 20 |

## Feature Discovery

### Filtering
Features are filtered by activation count range:
- **min_activation_count**: Minimum activations (default: 10)
- **max_activation_count**: Maximum activations (default: 10,000)
- Removes both dead features and overly-generic features

### Ranking
Features are **ranked by maximum activation value** before labeling. This ensures you analyze the most strongly-activating features first.

### Selection
- **max_features**: Number of top features to save (default: 10)
- **Top examples per feature**: 20 (or fewer if less than 20 non-zero activations exist)

### Labeling Options
1. **Heuristic**: Fast, pattern-based labeling using common words
2. **LLM (Anthropic/OpenAI)**: Intelligent semantic labeling (requires API key)

### Output Format
```json
{
  "42": {
    "label": "Heavy metal guitar distortion",
    "top_examples": [
      {
        "activation": 45.2314,
        "prompt": "aggressive distorted electric guitar riff...",
        "type": "sample-level"
      },
      ...  // up to 20 examples
    ]
  },
  ...  // up to 10 features by default
}
```

## Common Workflows

### Training Multiple Layers
```bash
# Extract multiple layers
python -m src.data.extract --layers 6,12,18,24

# Train SAE for each layer
for layer in 6 12 18 24; do
    python -m src.training.train \
        --layer_idx $layer \
        --sae_type batch_topk \
        --hidden_dim 6144 \
        --top_k 64
done

# Discover features for each layer
for layer in 6 12 18 24; do
    python -m src.analysis.discover \
        --sae_path models/batch_topk/layer${layer}/dim6144/sae_best.pt \
        --activation_file data/activations/activations_layer${layer}.pt \
        --output results/features_layer${layer}.json
done
```

### Using Batch Scripts
```bash
# Train multiple layers (modify layers in script)
./scripts/train_sae_layers_10_16.sh

# Discover features for multiple layers
./scripts/run_discover.sh
```

### Memory-Constrained Training
```bash
# Use smaller batch size and hidden dimension
python -m src.training.train \
    --batch_size 128 \
    --hidden_dim 4096 \
    --num_samples 5000  # Subsample activations
```

### Comparing SAE Variants
```bash
# Train all variants on same layer
for sae_type in vanilla topk batch_topk; do
    python -m src.training.train \
        --sae_type $sae_type \
        --layer_idx 12 \
        --hidden_dim 6144
done
```

## API Usage

### Load and Use Trained SAE

```python
from src.models.sae import BatchTopKSAE
import torch

# Load checkpoint
checkpoint = torch.load('models/batch_topk/layer12/dim6144/sae_best.pt')
config = checkpoint['config']

# Initialize model
sae = BatchTopKSAE(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    top_k=config['top_k'],
    aux_penalty=config['aux_penalty'],
    device='cuda'
)
sae.load_state_dict(checkpoint['state_dict'])
sae.eval()

# Encode activations
activations = torch.randn(100, config['input_dim']).cuda()
features = sae.encode(activations)  # Shape: [100, 6144]

# Decode features
reconstructed = sae.decode(features)  # Shape: [100, input_dim]
```

### Feature Discovery API

```python
from src.analysis.feature_discovery import FeatureDiscovery

# Initialize discovery
discovery = FeatureDiscovery(sae, device='cuda')

# Discover features (top 10 by default)
feature_info = discovery.discover_and_label(
    activations=activations,
    music_metadata=metadata,
    use_llm=False,
    filter_features=True,
    min_activation_count=10,
    max_activation_count=10000,
    max_features=10,  # Top 10 most prominent features
    aggregation='max'  # or 'mean', 'moments' for token-level
)

# Save results (saves top 20 examples per feature)
discovery.save_features(feature_info, 'features.json')

# Load saved features
loaded_features = FeatureDiscovery.load_features('features.json')
```

## Running as Python Modules

All scripts are now organized in `src/` and should be run as Python modules:

```bash
# Use this format
python -m src.{module}.{script}

# NOT this (won't work anymore)
python scripts/script_name.py
```

**Examples:**
- `python -m src.data.download` - Download dataset
- `python -m src.data.extract` - Extract activations
- `python -m src.training.train` - Train SAE
- `python -m src.analysis.discover` - Discover features

This ensures proper Python path handling and imports work correctly.

## Tips and Best Practices

1. **Start with medium dataset** (`musiccaps_prompts_medium.txt`) for testing
2. **Use BatchTopKSAE** for most tasks (best balance)
3. **Run from project root** using `python -m` for correct imports
4. **Default to top 10 features** with 20 examples - increase `--max_features` if needed
5. **Monitor dead features** during training (should decrease)
6. **Adjust top_k** based on desired sparsity (32-128 typical)
7. **Enable LLM labeling** for final analysis only (API costs)
8. **Check training curves** for convergence (validation loss)
9. **Compare multiple layers** to find best interpretability
10. **Use batch scripts** in `scripts/` for processing multiple layers

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 128 or 64)
- Reduce `--hidden_dim` (try 4096 or 2048)
- Enable `--num_samples` to subsample activations
- Use CPU: `--device cpu` (slower but more memory)

### Poor Feature Quality
- Increase `--min_activation_count` (try 20-50)
- Decrease `--max_activation_count` (try 5000)
- Train longer: `--epochs 30`
- Increase `--aux_penalty` for TopK variants

### Training Instability
- Lower learning rate: `--lr 5e-5`
- Increase warmup: automatic in current implementation
- Check for NaN/Inf in training curves

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sae_music_interp,
  title = {SAE Music Interpretability},
  author = {Zhang, Siyu},
  year = {2025},
  url = {https://github.com/zhangsiyu1103/sae-music-interp}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- MusicGen model by Meta AI
- MusicCaps dataset for text prompts
- Inspired by SAE interpretability research
