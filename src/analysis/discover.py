#!/usr/bin/env python3
"""
Discover and label SAE features automatically.

Usage:
    # Without LLM (heuristic labeling)
    python scripts/discover_features.py \
        --sae_path models/sae_layer12.pt \
        --activation_file data/activations/layer12.pt \
        --output results/features_layer12.json

    # With LLM labeling (Anthropic)
    python scripts/discover_features.py \
        --sae_path models/sae_layer12.pt \
        --activation_file data/activations/layer12.pt \
        --output results/features_layer12.json \
        --use_llm \
        --llm_api anthropic

    # Memory-efficient mode (auto-sampling)
    python scripts/discover_features.py \
        --sae_path models/sae_layer12.pt \
        --activation_file data/activations/layer12.pt \
        --output results/features_layer12.json \
        --num_samples 3000 \
        --max_features 50

    # Token-level localization (find exact token positions with timing)
    python scripts/discover_features.py \
        --sae_path models/sae_layer12.pt \
        --activation_file data/activations/layer12.pt \
        --output results/features_layer12.json \
        --token_level

Notes:
    - Files > 2GB are automatically sampled (use --num_samples -1 to disable)
    - Use --max_features to control how many features to label (default: 10)
    - Features are filtered by activation count (default: 10-10000 activations)
    - Features are ranked by maximum activation value before labeling
    - Use --token_level to get token positions and timing info (not just sample-level)
    - Outputs JSON file with feature labels and top 20 examples per feature
"""

import argparse
import torch
from pathlib import Path
import sys
import os
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import (
    SparseAutoencoder,
    TopKSAE,
    BatchTopKSAE,
    GlobalBatchTopKMatryoshkaSAE
)
from src.analysis.feature_discovery import FeatureDiscovery


def _clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_sae(sae_path: str, device: str):
    """Load SAE model from checkpoint, supporting all variants."""
    checkpoint = torch.load(sae_path, map_location=device)

    # Check if it's a vanilla SAE or TopK variant
    if 'config' in checkpoint and 'sae_type' in checkpoint['config']:
        # TopK variant
        config = checkpoint['config']
        sae_type = config['sae_type']

        if sae_type == 'topk':
            sae = TopKSAE(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                top_k=config['top_k'],
                aux_penalty=config['aux_penalty'],
                device=device
            )
        elif sae_type == 'batch_topk':
            sae = BatchTopKSAE(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                top_k=config['top_k'],
                aux_penalty=config['aux_penalty'],
                device=device
            )
        elif sae_type == 'matryoshka':
            sae = GlobalBatchTopKMatryoshkaSAE(
                input_dim=config['input_dim'],
                group_sizes=config['group_sizes'],
                top_k=config['top_k'],
                aux_penalty=config['aux_penalty'],
                device=device
            )
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}")

        sae.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded {sae_type} SAE from {sae_path}")
    else:
        # Vanilla SparseAutoencoder
        config = checkpoint['config']
        sae = SparseAutoencoder(**config)
        sae.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded vanilla SAE from {sae_path}")

    sae.to(device)
    sae.eval()
    return sae


def main():
    parser = argparse.ArgumentParser(
        description='Discover and label SAE features'
    )

    # Input arguments
    parser.add_argument(
        '--sae_path',
        type=str,
        required=True,
        help='Path to trained SAE checkpoint (.pt file)'
    )
    parser.add_argument(
        '--activation_file',
        type=str,
        required=True,
        help='Path to activation data (.pt file)'
    )

    # Feature discovery arguments
    parser.add_argument(
        '--filter_features',
        action='store_true',
        default=True,
        help='Filter features before labeling (default: True)'
    )
    parser.add_argument(
        '--no_filter',
        action='store_true',
        help='Disable feature filtering'
    )
    parser.add_argument(
        '--min_activation_count',
        type=int,
        default=10,
        help='Minimum number of times a feature must activate (default: 10)'
    )
    parser.add_argument(
        '--max_activation_count',
        type=int,
        default=10000,
        help='Maximum number of times a feature can activate (default: 10000)'
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=10,
        help='Maximum number of features to label (default: 10)'
    )

    # LLM labeling arguments
    parser.add_argument(
        '--use_llm',
        action='store_true',
        help='Use LLM for intelligent feature labeling'
    )
    parser.add_argument(
        '--llm_api',
        type=str,
        choices=['openai', 'anthropic'],
        default='anthropic',
        help='LLM API to use (default: anthropic)'
    )
    parser.add_argument(
        '--token_level',
        action='store_true',
        help='Enable token-level localization (shows which specific tokens activate features with timing info)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save feature labels (.json file)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of activation samples to use (default: auto-sample if file > 2GB)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load SAE model
    print(f"\nLoading SAE from {args.sae_path}...")
    sae = load_sae(args.sae_path, args.device)

    # Check file size and auto-sample if needed
    file_size_gb = os.path.getsize(args.activation_file) / (1024**3)
    print(f"\nActivation file size: {file_size_gb:.2f} GB")

    # Load activation data
    print(f"Loading activations from {args.activation_file}...")
    data = torch.load(args.activation_file, map_location='cpu')  # Load to CPU first
    activations = data['activations']
    metadata = data.get('metadata', [])
    config = data.get('config', {})

    # Extract seq_len from metadata (stored per batch)
    print(f"Activation shape: {activations.shape}")
    print(f"Activation memory: {activations.element_size() * activations.nelement() / (1024**3):.2f} GB")

    # Build sample boundaries from metadata (handles variable seq_len per batch)
    sample_boundaries = []
    flat_metadata = []
    num_samples = 0

    if isinstance(metadata, list) and len(metadata) > 0:
        if isinstance(metadata[0], dict) and 'prompts' in metadata[0]:
            # Batched metadata format
            token_offset = 0
            for batch in metadata:
                batch_seq_len = batch.get('seq_len', 1)
                batch_prompts = batch['prompts']

                for prompt in batch_prompts:
                    sample_boundaries.append((token_offset, batch_seq_len))
                    flat_metadata.append({'prompt': prompt, 'seq_len': batch_seq_len})
                    token_offset += batch_seq_len
                    num_samples += 1

            print(f"  Samples: {num_samples}")
            print(f"  Seq lengths: variable (per-batch)")
            seq_lens = [batch['seq_len'] for batch in metadata]
            print(f"  Seq length range: {min(seq_lens)} - {max(seq_lens)}")
        else:
            print("  Warning: Metadata format not recognized")
            num_samples = activations.shape[0]
    else:
        print("  Warning: No metadata found")
        num_samples = activations.shape[0]

    # Subsample if requested or auto-enabled
    did_subsample = False
    if args.num_samples and args.num_samples > 0 and num_samples > 0 and args.num_samples < num_samples and len(sample_boundaries) > 0:
        print(f"\nSubsampling to {args.num_samples} samples (from {num_samples})...")
        sample_indices = torch.randperm(num_samples)[:args.num_samples]
        sample_indices_sorted = sorted(sample_indices.tolist())

        # Select tokens for sampled sequences (handles variable seq_len)
        token_indices = []
        new_metadata = []
        for sample_idx in sample_indices_sorted:
            start_token, seq_len = sample_boundaries[sample_idx]
            token_indices.extend(range(start_token, start_token + seq_len))
            new_metadata.append(flat_metadata[sample_idx])

        activations = activations[token_indices].contiguous()

        # Rebuild batched metadata with variable seq_len support
        # Group by seq_len to create batches
        seq_len_groups = {}
        for meta in new_metadata:
            seq_len = meta['seq_len']
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append(meta['prompt'])

        # Create batched metadata
        metadata = []
        for seq_len, prompts in seq_len_groups.items():
            metadata.append({'prompts': prompts, 'seq_len': seq_len})

        print(f"  New activation shape: {activations.shape}")
        print(f"  New activation memory: {activations.element_size() * activations.nelement() / (1024**3):.2f} GB")
        print(f"  New batches: {len(metadata)}")

        # Clean up
        del token_indices, new_metadata, seq_len_groups, flat_metadata, sample_boundaries
        _clear_gpu_memory()
        did_subsample = True

    # Clean up temporary structures if not already done
    if not did_subsample and 'flat_metadata' in locals():
        del flat_metadata, sample_boundaries
        _clear_gpu_memory()

    # Load metadata - keep batched format for FeatureDiscovery
    # Each batch dict should have: {'prompts': [...], 'seq_len': int, ...}
    if isinstance(metadata, list) and len(metadata) > 0:
        # Check if it's batched format (has 'prompts' key)
        if isinstance(metadata[0], dict) and 'prompts' in metadata[0]:
            music_metadata = metadata  # Keep batched format
            print(f"Loaded batched metadata from activation file ({len(metadata)} batches)")
        else:
            # Already flat or different format - use as is
            music_metadata = metadata
            print(f"Loaded metadata from activation file ({len(metadata)} entries)")
    else:
        # Fallback: load from musiccaps prompts
        try:
            with open("data/musiccaps_prompts_medium.txt", 'r') as f:
                musiccaps_prompts = [line.strip() for line in f if line.strip()]
            # Create batched format with default seq_len
            music_metadata = [{'prompts': musiccaps_prompts, 'seq_len': seq_len or 1}]
            print(f"Loaded prompts from musiccaps file")
        except FileNotFoundError:
            # Create dummy metadata
            print("Warning: No metadata found, using dummy prompts")
            num_samples = activations.shape[0] // (seq_len or 1)
            music_metadata = [{'prompts': [f'sample_{i}' for i in range(num_samples)], 'seq_len': seq_len or 1}]

    # Initialize feature discovery
    print("\nInitializing feature discovery...")
    discovery = FeatureDiscovery(sae, device=args.device)

    # Discover and label features
    filter_features = args.filter_features and not args.no_filter

    if filter_features:
        print("\nFeature filtering is ENABLED")
        print(f"  Min activation count: {args.min_activation_count}")
        print(f"  Max activation count: {args.max_activation_count}")
        print(f"  (Features must activate between {args.min_activation_count} and {args.max_activation_count} times)")
    else:
        print("\nFeature filtering is DISABLED (--no_filter)")
        print("  All features will be analyzed (may be slow and noisy)")

    if args.use_llm:
        print(f"\nLLM labeling enabled using {args.llm_api}")
        if args.llm_api == 'anthropic':
            print("Make sure ANTHROPIC_API_KEY is set in environment")
        elif args.llm_api == 'openai':
            print("Make sure OPENAI_API_KEY is set in environment")

    # Ensure activations are on CPU (will be batched to GPU during processing)
    activations = activations.cpu()

    # Clear memory before discovery
    _clear_gpu_memory()

    print(f"\n{'='*70}")
    print("RUNNING FEATURE DISCOVERY")
    print(f"{'='*70}")
    print(f"  Activations: {activations.shape} ({activations.element_size() * activations.nelement() / (1024**2):.1f} MB)")
    print(f"  Device: {args.device}")
    print(f"  Filter: {filter_features} (min={args.min_activation_count}, max={args.max_activation_count})")
    print(f"  Max features: {args.max_features}")
    print(f"  LLM: {args.use_llm}")
    print(f"  Token-level: {args.token_level}")
    print(f"{'='*70}\n")

    # Determine aggregation mode
    aggregation = 'moments' if args.token_level else 'max'

    feature_info = discovery.discover_and_label(
        activations=activations,
        music_metadata=music_metadata,
        use_llm=args.use_llm,
        llm_api=args.llm_api if args.use_llm else None,
        filter_features=filter_features,
        min_activation_count=args.min_activation_count,
        max_activation_count=args.max_activation_count,
        max_features=args.max_features,
        aggregation=aggregation
    )

    # Save results
    print(f"\nSaving feature labels...")

    # Save JSON
    discovery.save_features(feature_info, str(output_path))

    # Print summary
    print("\n" + "="*60)
    print("Feature Discovery Summary")
    print("="*60)
    print(f"Total features discovered: {len(feature_info)}")
    print(f"Aggregation mode: {'token-level' if args.token_level else 'sample-level'}")
    print(f"\nTop 10 features:")
    for i, (feat_idx, info) in enumerate(list(feature_info.items())[:10]):
        print(f"  {i+1}. Feature {feat_idx}: {info['label']}")
        if info['top_examples']:
            top_ex = info['top_examples'][0]
            print(f"     Top example: '{top_ex['metadata'].get('prompt', 'N/A')}'")
            print(f"     Activation: {top_ex['activation']:.3f}")

            # Show token-level info if available
            if top_ex.get('type') == 'moment':
                print(f"     Token position: {top_ex.get('token_index')} (time: {top_ex.get('time_seconds', 0):.2f}s)")

    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"  JSON: {output_path}")
    print("\nDone!")
    print(f"\nNext steps:")
    print(f"  1. Analyze features: cat {output_path}")
    print(f"  2. Visualize features in notebooks/feature_analysis.ipynb")
    print(f"  3. Use features for steering in generation")


if __name__ == "__main__":
    main()
