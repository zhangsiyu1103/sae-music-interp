#!/usr/bin/env python3
"""Train Sparse Autoencoder on MusicGen activations."""

import argparse
import torch
from pathlib import Path
import sys
import gc
import json
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.sae import (
    SparseAutoencoder,
    TopKSAE,
    BatchTopKSAE,
    GlobalBatchTopKMatryoshkaSAE
)
import matplotlib.pyplot as plt


def _clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class HighErrorBuffer:
    """Keeps a rolling set of high-error tokens for neuron resampling."""

    def __init__(self, max_size: int, per_batch_topk: int):
        self.max_size = max(0, max_size)
        self.per_batch_topk = max(0, per_batch_topk)
        self.entries = []

    def __len__(self):
        return len(self.entries)

    def add_batch(self, errors: torch.Tensor, inputs: torch.Tensor, residuals: torch.Tensor):
        if self.max_size == 0 or self.per_batch_topk == 0:
            return

        errors = errors.detach()
        topk = min(self.per_batch_topk, errors.shape[0])
        if topk == 0:
            return

        top_errors, top_indices = torch.topk(errors, topk)
        for err, idx in zip(top_errors.tolist(), top_indices.tolist()):
            self.entries.append({
                'error': float(err),
                'input': inputs[idx].detach().cpu(),
                'residual': residuals[idx].detach().cpu(),
            })

        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda entry: entry['error'], reverse=True)
            self.entries = self.entries[:self.max_size]

    def sample(self, num_samples: int):
        if self.max_size == 0 or len(self.entries) == 0:
            return []

        num_samples = max(0, min(num_samples, len(self.entries)))
        if num_samples == 0:
            return []

        errors = torch.tensor([entry['error'] for entry in self.entries], dtype=torch.float32)
        if errors.sum() <= 0:
            probs = torch.ones_like(errors) / len(self.entries)
        else:
            probs = errors / errors.sum()

        indices = torch.multinomial(probs, num_samples, replacement=False)
        return [self.entries[i] for i in indices.tolist()]


def resample_dead_features(sae, error_buffer: HighErrorBuffer, args, device: torch.device):
    """Resample dead neurons using high-error residuals."""
    if getattr(args, 'max_resample_per_step', 0) <= 0:
        return 0

    required_attrs = all(hasattr(sae, attr) for attr in ['num_batches_not_active', 'W_enc', 'W_dec'])
    if not required_attrs or error_buffer is None or len(error_buffer) == 0:
        return 0

    dead_mask = sae.num_batches_not_active >= args.dead_feature_resample_steps
    dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten()

    if dead_indices.numel() == 0:
        return 0

    num_to_resample = min(dead_indices.numel(), args.max_resample_per_step)
    samples = error_buffer.sample(num_to_resample)

    if len(samples) == 0:
        return 0

    dead_indices = dead_indices[:len(samples)]

    with torch.no_grad():
        decoder_weights = sae.W_dec.data
        decoder_norm = decoder_weights / (decoder_weights.norm(dim=-1, keepdim=True) + 1e-6)

        for dead_idx_tensor, sample in zip(dead_indices, samples):
            feature_idx = int(dead_idx_tensor.item())
            direction = sample['residual'].to(device)

            if torch.linalg.norm(direction) < 1e-6:
                direction = sample['input'].to(device)
            if torch.linalg.norm(direction) < 1e-6:
                direction = torch.randn_like(direction)

            direction = direction / (torch.linalg.norm(direction) + 1e-6)

            sae.W_enc.data[:, feature_idx] = direction

            decoder_norm[feature_idx] = 0.0
            projection = (decoder_norm @ direction).unsqueeze(1) * decoder_norm
            projection = projection.sum(dim=0)
            decoder_direction = direction - projection
            if torch.linalg.norm(decoder_direction) < 1e-6:
                decoder_direction = direction
            decoder_direction = decoder_direction / (torch.linalg.norm(decoder_direction) + 1e-6)
            decoder_weights[feature_idx] = decoder_direction
            decoder_norm[feature_idx] = decoder_direction

            if hasattr(sae, 'b_enc'):
                sae.b_enc.data[feature_idx] = 0.0

            sae.num_batches_not_active[feature_idx] = 0.0

        if hasattr(sae, 'make_decoder_weights_and_grad_unit_norm'):
            sae.make_decoder_weights_and_grad_unit_norm()

    return len(dead_indices)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def validate(sae, val_data, device, batch_size=256):
    sae.eval()
    val_losses = []
    val_l0s = []

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            x_batch = val_data[i:i+batch_size].to(device)
            output = sae(x_batch)
            val_losses.append(output['l2_loss'].item())
            val_l0s.append(output['l0_norm'].item())

    return {'val_loss': np.mean(val_losses), 'val_l0': np.mean(val_l0s)}


def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(history['loss'], label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['l2_loss'], label='Train')
    if 'val_l2' in history:
        axes[0, 1].plot(history['val_l2'], label='Val')
    axes[0, 1].set_title('L2 Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history['aux_loss'])
    axes[0, 2].set_title('Auxiliary Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True)

    axes[1, 0].plot(history['l0_norm'], label='Train')
    if 'val_l0' in history:
        axes[1, 0].plot(history['val_l0'], label='Val')
    axes[1, 0].set_title('Active Features (L0)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['num_dead_features'])
    axes[1, 1].set_title('Dead Features')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True)

    has_lr = 'lr' in history and len(history['lr']) > 0
    has_resample = 'resampled_features' in history and len(history['resampled_features']) > 0

    if has_lr or has_resample:
        if has_lr:
            axes[1, 2].plot(history['lr'], label='LR')
            axes[1, 2].set_yscale('log')
        axes[1, 2].set_title('LR & Resampled Features')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].grid(True)

        if has_resample:
            twin_ax = axes[1, 2].twinx()
            twin_ax.bar(range(len(history['resampled_features'])), history['resampled_features'],
                        alpha=0.3, color='tab:red', label='Resampled')
            twin_ax.set_ylabel('Resampled Features')

        if has_lr:
            axes[1, 2].set_ylabel('Learning Rate')
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_topk_sae(sae, train_data, val_data, args, output_dir):
    device = args.device
    sae.to(device)

    if val_data is None:
        val_size = int(0.1 * len(train_data))
        train_size = len(train_data) - val_size
        train_data, val_data = torch.split(train_data, [train_size, val_size])

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr, weight_decay=1e-4)

    num_batches = len(train_data) // args.batch_size
    total_steps = args.epochs * num_batches
    warmup_steps = min(1000, total_steps // 10)

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=args.lr * 0.01)

    history = {
        'loss': [], 'l2_loss': [], 'aux_loss': [], 'l0_norm': [],
        'num_dead_features': [], 'lr': [], 'val_loss': [], 'val_l0': [],
        'resampled_features': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    max_grad_norm = 1.0
    global_step = 0
    high_error_buffer = HighErrorBuffer(args.resample_buffer_size, args.resample_candidates_per_batch)

    for epoch in range(args.epochs):
        sae.train()
        perm = torch.randperm(len(train_data))
        train_data_shuffled = train_data[perm]
        epoch_losses = {k: [] for k in history.keys() if k not in ['val_loss', 'val_l0', 'resampled_features']}
        resampled_this_epoch = 0

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            start_idx = batch_idx * args.batch_size
            x_batch = train_data_shuffled[start_idx:start_idx + args.batch_size].to(device)

            output = sae(x_batch)
            loss = output['loss']

            optimizer.zero_grad()
            loss.backward()

            if hasattr(sae, 'W_dec'):
                torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)

            if hasattr(sae, 'make_decoder_weights_and_grad_unit_norm'):
                with torch.no_grad():
                    sae.make_decoder_weights_and_grad_unit_norm()

            optimizer.step()

            if hasattr(sae, 'make_decoder_weights_and_grad_unit_norm'):
                with torch.no_grad():
                    sae.make_decoder_weights_and_grad_unit_norm()

            global_step += 1

            if args.resample_buffer_size > 0 and 'sae_out' in output:
                with torch.no_grad():
                    recon_batch = output['sae_out']
                    residual = (x_batch - recon_batch).detach()
                    sample_errors = residual.pow(2).mean(dim=1)
                    high_error_buffer.add_batch(sample_errors, x_batch, residual)

            resampled_now = 0
            if args.resample_interval > 0 and global_step % args.resample_interval == 0:
                resampled_now = resample_dead_features(sae, high_error_buffer, args, device)
                if resampled_now > 0:
                    resampled_this_epoch += resampled_now

            current_lr = scheduler.step()

            epoch_losses['loss'].append(loss.item())
            epoch_losses['l2_loss'].append(output['l2_loss'].item())
            epoch_losses['aux_loss'].append(output['aux_loss'].item())
            epoch_losses['l0_norm'].append(output['l0_norm'].item())
            epoch_losses['num_dead_features'].append(
                output.get('num_dead_features', torch.tensor(0)).item()
                if torch.is_tensor(output.get('num_dead_features', 0))
                else output.get('num_dead_features', 0)
            )
            epoch_losses['lr'].append(current_lr)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l0': f"{output['l0_norm'].item():.1f}",
                'dead': f"{epoch_losses['num_dead_features'][-1]:.0f}",
                'lr': f"{current_lr:.2e}",
                'rsmp': resampled_now
            })

        del train_data_shuffled, x_batch, output, loss
        _clear_gpu_memory()

        for k in epoch_losses.keys():
            if epoch_losses[k]:
                history[k].append(np.mean(epoch_losses[k]))
        history['resampled_features'].append(resampled_this_epoch)

        val_metrics = validate(sae, val_data, device, args.batch_size)
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_l0'].append(val_metrics['val_l0'])

        print(f"\nEpoch {epoch+1}: Train={history['loss'][-1]:.6f}, Val={history['val_loss'][-1]:.6f}, "
              f"L0={history['l0_norm'][-1]:.1f}, Dead={history['num_dead_features'][-1]:.0f}, "
              f"Resampled={resampled_this_epoch}")

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'state_dict': sae.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'sae_type': args.sae_type,
                    'input_dim': sae.input_dim,
                    'hidden_dim': sae.hidden_dim,
                    'top_k': sae.top_k,
                    'aux_penalty': sae.aux_penalty,
                    'layer_idx': args.layer_idx,
                }
            }, output_dir / "sae_best.pt")
            print(f"  New best: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': sae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, output_dir / f"checkpoint_epoch{epoch+1}.pt")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train SAE')

    parser.add_argument('--sae_type', type=str, default='batch_topk',
                       choices=['vanilla', 'topk', 'batch_topk', 'matryoshka'])
    parser.add_argument('--hidden_dim', type=int, default=6144)
    parser.add_argument('--top_k', type=int, default=64)
    parser.add_argument('--aux_penalty', type=float, default=0.1)
    parser.add_argument('--n_batches_to_dead', type=int, default=100)
    parser.add_argument('--input_unit_norm', action='store_true')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resample_interval', type=int, default=1000,
                        help='Steps between neuron resampling (0 disables resampling)')
    parser.add_argument('--dead_feature_resample_steps', type=int, default=1000,
                        help='Minimum inactive steps before a neuron is eligible for resampling')
    parser.add_argument('--resample_buffer_size', type=int, default=4096,
                        help='Number of high-error tokens to keep for resampling')
    parser.add_argument('--resample_candidates_per_batch', type=int, default=32,
                        help='How many high-error samples to stash from each batch')
    parser.add_argument('--max_resample_per_step', type=int, default=64,
                        help='Maximum neurons to resample at each resampling interval')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--layer_idx', type=int, default=12)
    parser.add_argument('--activation_dir', type=str, default="data/activations/")
    parser.add_argument('--output', type=str, default='models/')

    args = parser.parse_args()

    activation_file = Path(args.activation_dir) / f"activations_layer{args.layer_idx}.pt"
    activations = torch.load(activation_file)

    if isinstance(activations, dict):
        activations = activations['activations']

    if len(activations.shape) == 3:
        num_samples, seq_len, feat_dim = activations.shape
        activations = activations.reshape(num_samples * seq_len, feat_dim)

    input_dim = activations.shape[1]
    output_dir = Path(args.output) / args.sae_type / f"layer{args.layer_idx}" / f"dim{args.hidden_dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.sae_type} SAE: layer {args.layer_idx}, "
          f"{input_dim} -> {args.hidden_dim} ({args.hidden_dim/input_dim:.1f}x), "
          f"k={args.top_k}, aux={args.aux_penalty}")
    if args.sae_type == 'vanilla':
        sae = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            aux_penalty=args.aux_penalty,
            n_batches_to_dead=args.n_batches_to_dead,
            input_unit_norm=args.input_unit_norm,
            device=args.device
        )
    elif args.sae_type == 'batch_topk':
        sae = BatchTopKSAE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            top_k=args.top_k,
            aux_penalty=args.aux_penalty,
            n_batches_to_dead=args.n_batches_to_dead,
            input_unit_norm=args.input_unit_norm,
            device=args.device
        )
    elif args.sae_type == 'topk':
        sae = TopKSAE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            top_k=args.top_k,
            aux_penalty=args.aux_penalty,
            n_batches_to_dead=args.n_batches_to_dead,
            input_unit_norm=args.input_unit_norm,
            device=args.device
        )
    elif args.sae_type == 'matryoshka':
        sae = GlobalBatchTopKMatryoshkaSAE(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            top_k=args.top_k,
            aux_penalty=args.aux_penalty,
            n_batches_to_dead=args.n_batches_to_dead,
            input_unit_norm=args.input_unit_norm,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")

    history = train_topk_sae(sae, activations, None, args, output_dir)

    torch.save({
        'state_dict': sae.state_dict(),
        'config': {
            'sae_type': args.sae_type,
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'top_k': args.top_k,
            'aux_penalty': args.aux_penalty,
            'layer_idx': args.layer_idx,
        },
        'history': history
    }, output_dir / "sae_final.pt")

    plot_training_history(history, str(output_dir / 'training_curves.png'))

    with open(output_dir / 'training_stats.json', 'w') as f:
        json.dump({
            'config': vars(args),
            'final_metrics': {
                'train_loss': history['loss'][-1],
                'val_loss': history['val_loss'][-1],
                'train_l0': history['l0_norm'][-1],
                'val_l0': history['val_l0'][-1],
                'dead_features': history['num_dead_features'][-1],
                'resampled_features_last_epoch': history['resampled_features'][-1],
                'resampled_features_total': sum(history['resampled_features']),
            },
            'best_val_loss': min(history['val_loss']),
        }, f, indent=2)

    print(f"\nBest val: {min(history['val_loss']):.6f}")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
