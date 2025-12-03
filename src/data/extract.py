#!/usr/bin/env python3
"""
Extract activations from MusicGen model.

Usage:
    # Single layer
    python scripts/extract_activations.py \
        --model facebook/musicgen-medium \
        --layers 12 \
        --prompt_file data/prompts.txt \
        --batch_size 4 \
        --duration 10.0

    # Multiple layers
    python scripts/extract_activations.py \
        --layers 6,12,18 \
        --prompt_file data/prompts.txt

    # Layer range
    python scripts/extract_activations.py \
        --layers 0-23 \
        --prompt_file data/prompts.txt
"""

import argparse
import torch
from pathlib import Path
import sys
from tqdm import tqdm
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from src.models.musicgen_wrapper import ActivationExtractor


def parse_layers(layers_str: str) -> List[int]:
    """
    Parse layer specification string.

    Formats:
        "12"        -> [12]
        "6,12,18"   -> [6, 12, 18]
        "0-5"       -> [0, 1, 2, 3, 4, 5]
        "0-5,10,15" -> [0, 1, 2, 3, 4, 5, 10, 15]
    """
    layers = []
    for part in layers_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def main():
    parser = argparse.ArgumentParser(description='Extract activations from MusicGen')

    parser.add_argument('--model', type=str, default='facebook/musicgen-melody',
                        help='MusicGen model name')
    parser.add_argument('--layers', type=str, default='12',
                        help='Layers to extract from. Formats: "12", "6,12,18", "0-5", "0-5,10,15"')
    parser.add_argument('--prompt_file', type=str,
                        default='data/musiccaps_prompts_medium.txt',
                        help='Path to file with prompts (one per line)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Duration of each sample in seconds')
    parser.add_argument('--output_dir', type=str, default='data/activations',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no_filter_special_tokens', action='store_true',
                        help='Do not filter special/conditioning tokens (keep all tokens)')

    args = parser.parse_args()

    # Parse layers
    layer_idxs = parse_layers(args.layers)
    print(f"Will extract from layers: {layer_idxs}")

    # Load prompts
    print(f"Loading prompts from {args.prompt_file}...")
    with open(args.prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    print(f"\nInitializing MusicGen: {args.model}")
    extractor = ActivationExtractor(model_name=args.model, device=args.device)

    # Extract activations in batches
    print(f"\nExtracting activations from layers {layer_idxs}...")
    print(f"Duration: {args.duration}s, Batch size: {args.batch_size}")
    print(f"Filter special tokens: {not args.no_filter_special_tokens}")

    # Storage for each layer
    all_activations = {idx: [] for idx in layer_idxs}
    metadata = []

    for batch_start in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[batch_start:batch_start + args.batch_size]

        # Extract activations from all layers at once
        batch_acts, seq_len = extractor.extract_from_prompts(
            prompts=batch_prompts,
            layer_idxs=layer_idxs,
            duration=args.duration,
            filter_special_tokens=not args.no_filter_special_tokens
        )

        # Store activations for each layer
        for idx in layer_idxs:
            all_activations[idx].append(batch_acts[idx])

        metadata.append({
            'prompts': batch_prompts,
            'seq_len': seq_len,
            'batch_idx': batch_start
        })

    # Save each layer to its own file
    for idx in layer_idxs:
        activations = torch.cat(all_activations[idx], dim=0)
        print(f"\nLayer {idx} activations shape: {activations.shape}")

        output_path = output_dir / f"activations_layer{idx}.pt"
        torch.save({
            'activations': activations,
            'metadata': metadata,
            'config': {
                'model': args.model,
                'layer': idx,
                'duration': args.duration,
                'num_samples': len(prompts),
                'filter_special_tokens': not args.no_filter_special_tokens
            }
        }, output_path)
        print(f"Saved to {output_path}")

        # Validate that metadata matches activations
        total_samples = sum(len(batch['prompts']) for batch in metadata)
        expected_tokens = sum(len(batch['prompts']) * batch['seq_len'] for batch in metadata)
        actual_tokens = activations.shape[0]

        print(f"\nValidation for layer {idx}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Expected tokens (from metadata): {expected_tokens}")
        print(f"  Actual tokens (in tensor): {actual_tokens}")

        if expected_tokens == actual_tokens:
            print(f"  ✓ Metadata matches activations!")
        else:
            print(f"  ⚠️  MISMATCH: {abs(expected_tokens - actual_tokens)} tokens difference")
            print(f"     This may cause bugs in feature discovery!")

    print(f"\nDone! Extracted activations for {len(layer_idxs)} layers.")
    print(f"Next: python scripts/train_sae.py --activation_file {output_dir}/activations_layer<N>.pt")


if __name__ == "__main__":
    main()
