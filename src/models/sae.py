"""Sparse Autoencoder implementations for music generation interpretability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm
import gc


def _clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class SparseAutoencoder(nn.Module):
    """Vanilla SAE with L1 sparsity."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coef: float = 0.05, tied_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        self.tied_weights = tied_weights
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        if tied_weights:
            self.decoder_weight = None
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def loss(self, x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x)
        sparsity_loss = torch.abs(z).mean()
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'l0': (z > 0).float().sum(dim=1).mean()
        }
    
    def train_model(self, activations: torch.Tensor, epochs: int = 100, batch_size: int = 256,
                    lr: float = 1e-3, device: str = 'cuda', verbose: bool = True) -> Dict[str, list]:
        self.to(device)
        activations = activations.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        history = {'total_loss': [], 'recon_loss': [], 'sparsity_loss': [], 'l0': []}
        num_batches = len(activations) // batch_size
        
        for epoch in range(epochs):
            perm = torch.randperm(len(activations))
            activations_shuffled = activations[perm]
            epoch_losses = {k: [] for k in history.keys()}
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}") if verbose else range(num_batches)
            
            for batch_idx in pbar:
                x_batch = activations_shuffled[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                x_recon, z = self(x_batch)
                losses = self.loss(x_batch, x_recon, z)
                
                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
                
                for k in history.keys():
                    epoch_losses[k].append(losses[k.replace('_loss', '')].item() if 'loss' in k else losses[k].item())
                
                if verbose:
                    pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 'l0': f"{losses['l0'].item():.1f}"})

            del activations_shuffled
            for k in history.keys():
                history[k].append(np.mean(epoch_losses[k]))
            
            _clear_gpu_memory()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={history['total_loss'][-1]:.4f}, L0={history['l0'][-1]:.1f}")

        return history
    
    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'sparsity_coef': self.sparsity_coef,
                'tied_weights': self.tied_weights
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device).eval()
        return model


class TopKSAEBase(nn.Module):
    """Base class for TopK SAE variants with shared functionality."""
    
    def preprocess_input(self, x):
        if self.input_unit_norm:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.input_unit_norm:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    def update_inactive_features(self, acts):
        active_mask = (acts > 0).any(dim=0)
        self.num_batches_not_active[~active_mask] += 1
        self.num_batches_not_active[active_mask] = 0

    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / (self.W_dec.norm(dim=-1, keepdim=True) + 1e-8)
        if self.W_dec.grad is not None:
            W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(dim=-1, keepdim=True) * W_dec_normed
            self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed


class GlobalBatchTopKMatryoshkaSAE(TopKSAEBase):
    def __init__(self, input_dim: int, group_sizes: list, top_k: int = 128, aux_penalty: float = 0.01,
                 n_batches_to_dead: int = 100, top_k_aux: int = 64, input_unit_norm: bool = False,
                 device: str = 'cuda', dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.group_sizes = group_sizes
        self.top_k = top_k
        self.aux_penalty = aux_penalty
        self.n_batches_to_dead = n_batches_to_dead
        self.top_k_aux = top_k_aux
        self.input_unit_norm = input_unit_norm
        self.device_name = device
        self.dtype = dtype
        
        self.total_dict_size = sum(group_sizes)
        self.group_indices = [0] + list(np.cumsum(group_sizes))
        self.active_groups = len(group_sizes)
        
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.total_dict_size))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(input_dim, self.total_dict_size)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.total_dict_size, input_dim)))
        
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        self.num_batches_not_active = torch.zeros(self.total_dict_size, device=device)
        self.register_buffer('threshold', torch.tensor(0.0))
        
        self.to(dtype).to(device)

    def compute_activations(self, x_cent):
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(acts.flatten(), self.top_k * x_cent.shape[0], dim=-1)
            acts_topk = torch.zeros_like(acts.flatten()).scatter(-1, acts_topk.indices, acts_topk.values).reshape(acts.shape)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        x_cent = x - self.b_dec
        _, acts_topk = self.compute_activations(x_cent)
        return acts_topk

    def decode(self, acts_topk: torch.Tensor) -> torch.Tensor:
        out = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(out, self.x_mean, self.x_std)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if target is None:
            target = x
        x, x_mean, x_std = self.preprocess_input(x)
        
        x_cent = x - self.b_dec
        x_reconstruct = self.b_dec
        
        intermediate_reconstructs = []
        all_acts, all_acts_topk = self.compute_activations(x_cent)
        
        for i in range(self.active_groups):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
            acts_topk = all_acts_topk[:, start_idx:end_idx]
            x_reconstruct = acts_topk @ self.W_dec[start_idx:end_idx, :] + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)
        
        self.update_inactive_features(all_acts_topk)
        return self.get_loss_dict(target, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, intermediate_reconstructs)

    def get_loss_dict(self, x, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, intermediate_reconstructs):
        total_l2_loss = (self.b_dec - x.float()).pow(2).mean()
        l2_losses = torch.tensor([]).to(x.device)
        
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_loss = (intermediate_reconstruct.float() - x.float()).pow(2).mean()
            l2_losses = torch.cat([l2_losses, l2_loss.unsqueeze(0)])
            total_l2_loss += l2_loss
        
        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, all_acts)
        loss = mean_l2_loss + aux_loss
        
        num_dead_features = (self.num_batches_not_active > self.n_batches_to_dead).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        return {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": l2_losses.min(),
            "max_l2_loss": l2_losses.max(),
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }

    def get_auxiliary_loss(self, x, x_reconstruct, all_acts):
        residual = x.float() - x_reconstruct.float()
        dead_features = self.num_batches_not_active >= self.n_batches_to_dead
        
        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(all_acts[:, dead_features], min(self.top_k_aux, dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(all_acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = self.aux_penalty * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            return l2_loss_aux
        
        return torch.tensor(0.0, device=x.device)

    def update_threshold(self, acts_topk, lr=0.01):
        if self.training:
            threshold_estimate = acts_topk[acts_topk > 0].min() if (acts_topk > 0).any() else torch.tensor(0.0)
            self.threshold.data = (1 - lr) * self.threshold.data + lr * threshold_estimate


class BatchTopKSAE(TopKSAEBase):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int = 128, aux_penalty: float = 0.01,
                 n_batches_to_dead: int = 100, top_k_aux: int = 64, input_unit_norm: bool = False,
                 device: str = 'cuda', dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.aux_penalty = aux_penalty
        self.n_batches_to_dead = n_batches_to_dead
        self.top_k_aux = top_k_aux
        self.input_unit_norm = input_unit_norm
        
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(input_dim, hidden_dim)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_dim, input_dim)))
        
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        self.num_batches_not_active = torch.zeros(hidden_dim, device=device)
        self.register_buffer('threshold', torch.tensor(0.0))
        
        self.to(dtype).to(device)

    def compute_activations(self, x):
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(acts.flatten(), self.top_k * x.shape[0], dim=-1)
            acts_topk = torch.zeros_like(acts.flatten()).scatter(-1, acts_topk.indices, acts_topk.values).reshape(acts.shape)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk, pre_acts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        _, acts_topk, _ = self.compute_activations(x)
        return acts_topk

    def decode(self, acts_topk: torch.Tensor) -> torch.Tensor:
        out = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(out, self.x_mean, self.x_std)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)
        acts, acts_topk, pre_acts = self.compute_activations(x)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        self.update_threshold(acts_topk)
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std, pre_acts)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std, pre_acts):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, pre_acts)
        loss = l2_loss + aux_loss
        
        num_dead_features = (self.num_batches_not_active > self.n_batches_to_dead).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        return {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
        }

    def get_auxiliary_loss(self, x, x_reconstruct, pre_acts):
        residual = (x.float() - x_reconstruct.float()).detach()
        dead_mask = self.num_batches_not_active >= self.n_batches_to_dead
        
        if dead_mask.sum() > 0:
            dead_pre_acts = pre_acts[:, dead_mask]
            k_aux = min(self.top_k_aux, dead_mask.sum())
            topk_aux = torch.topk(dead_pre_acts, k_aux, dim=-1)
            acts_aux = torch.zeros_like(dead_pre_acts).scatter(-1, topk_aux.indices, F.relu(topk_aux.values))
            x_aux = acts_aux @ self.W_dec[dead_mask]
            return self.aux_penalty * (residual.float() - x_aux.float()).pow(2).mean()
        
        return torch.tensor(0.0, device=x.device)

    def update_threshold(self, acts_topk, lr=0.01):
        if self.training:
            threshold_estimate = acts_topk[acts_topk > 0].min() if (acts_topk > 0).any() else torch.tensor(0.0)
            self.threshold.data = (1 - lr) * self.threshold.data + lr * threshold_estimate


class TopKSAE(TopKSAEBase):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int = 128, aux_penalty: float = 0.01,
                 n_batches_to_dead: int = 100, top_k_aux: int = 64, input_unit_norm: bool = False,
                 device: str = 'cuda', dtype: torch.dtype = torch.float32, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.aux_penalty = aux_penalty
        self.n_batches_to_dead = n_batches_to_dead
        self.top_k_aux = top_k_aux
        self.input_unit_norm = input_unit_norm
        
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(input_dim, hidden_dim)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_dim, input_dim)))
        
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        self.num_batches_not_active = torch.zeros(hidden_dim, device=device)
        
        self.to(dtype).to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.top_k, dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        return acts_topk

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        out = acts @ self.W_dec + self.b_dec
        return self.postprocess_output(out, self.x_mean, self.x_std)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        acts_topk = torch.topk(acts, self.top_k, dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        
        self.update_inactive_features(acts_topk)
        return self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std, pre_acts)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std, pre_acts):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, pre_acts)
        loss = l2_loss + aux_loss
        
        num_dead_features = (self.num_batches_not_active > self.n_batches_to_dead).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        return {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "aux_loss": aux_loss,
        }

    def get_auxiliary_loss(self, x, x_reconstruct, pre_acts):
        residual = (x.float() - x_reconstruct.float()).detach()
        dead_features = self.num_batches_not_active >= self.n_batches_to_dead
        
        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(pre_acts[:, dead_features], min(self.top_k_aux, dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(pre_acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, F.relu(acts_topk_aux.values))
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            return self.aux_penalty * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        
        return torch.tensor(0.0, device=x.device)
