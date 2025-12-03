"""
MusicGen wrapper for extracting intermediate activations.

This module provides tools to:
1. Generate music with MusicGen
2. Extract activations from specific transformer layers
3. Apply SAE features to control generation
"""

import torch
from audiocraft.models import MusicGen
from typing import List, Optional, Dict, Tuple
import numpy as np
import torchaudio
import gc


def _clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class ActivationExtractor:
    """Extract activations from MusicGen transformer layers."""

    def __init__(
        self,
        model_name: str = 'facebook/musicgen-melody',
        device: str = 'cuda'
    ):
        self.device = device
        self.model_name = model_name

        print(f"Loading MusicGen model: {model_name}")
        self.model = MusicGen.get_pretrained(model_name, device=device)
        self.model.set_generation_params(use_sampling=True, top_k=250)
        
        self.activations = {}
        self.hooks = []
        
    def _get_transformer_layers(self):
        """Get transformer layer modules."""
        # Check for audiocraft structure
        if hasattr(self.model, 'lm'):
            return self.model.lm.transformer.layers
        # Fallback for potential HF structure or other versions
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
             return self.model.model.decoder.layers
        else:
            raise AttributeError("Model structure not recognized. Please inspect `self.model` manually.")
    
    def register_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # FIX 1: Handle Tuple Output
            # Transformer layers usually return (hidden_states, attentions, ...)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output

            # FIX: Ensure we are capturing what we expect (Batch, Seq, Hidden)
            # During generation with cache, Seq is usually 1
            
            key = f'layer_{layer_idx}'
            if key not in self.activations:
                self.activations[key] = []
            
            # Detach and move to CPU immediately
            act = out_tensor.detach().cpu()
            self.activations[key].append(act)

        layers = self._get_transformer_layers()
        if layer_idx >= len(layers):
            raise ValueError(f"Layer {layer_idx} out of range")

        hook = layers[layer_idx].register_forward_hook(hook_fn)
        self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        _clear_gpu_memory()
    
    def extract_from_prompts(
        self,
        prompts: List[str],
        layer_idxs: List[int],
        duration: float = 10.0,
        filter_special_tokens: bool = False # Changed default to False
    ) -> Tuple[Dict[int, torch.Tensor], int]:
        
        self.remove_hooks()
        for idx in layer_idxs:
            self.register_hook(idx)

        # Clear previous
        self.activations = {}

        self.model.set_generation_params(duration=duration)
        with torch.no_grad():
            self.model.generate(prompts)

        out_dict = {}
        seq_len = None
        
        # Calculate expected frames (50Hz for MusicGen)
        expected_frames = int(duration * 50)

        for idx in layer_idxs:
            act_list = self.activations.get(f'layer_{idx}')
            if not act_list:
                raise RuntimeError(f"No activation captured for layer {idx}")

            # Concatenate along time dimension
            acts_concat = torch.cat(act_list, dim=1) 
            
            batch_size, total_seq_len, hidden_dim = acts_concat.shape

            # FIX 2: Simplified Filtering Logic
            # MusicGen is Decoder-only for audio; Text is Cross-Attn.
            # We usually just want to ensure we don't exceed expected frames
            # or handle the BOS token if present.
            
            if filter_special_tokens and total_seq_len > expected_frames:
                # If we have extra tokens, it's likely just BOS or slight padding.
                # We prioritize keeping the *end* of the sequence (the generation).
                acts_concat = acts_concat[:, -expected_frames:, :]
                seq_len = expected_frames
            else:
                seq_len = total_seq_len

            # Flatten: [batch_size * seq_len, hidden_dim]
            out_dict[idx] = acts_concat.reshape(batch_size * seq_len, hidden_dim)

        self.remove_hooks()
        return out_dict, seq_len
    
   

class MusicGenController:
    """
    Control MusicGen generation using SAE features.
    """
    
    def __init__(
        self,
        model_name: str = 'facebook/musicgen-medium',
        sae_path: Optional[str] = None,
        feature_labels: Optional[Dict[int, str]] = None,
        device: str = 'cuda'
    ):
        """
        Initialize controller with MusicGen and optional SAE.
        
        Args:
            model_name: MusicGen model identifier
            sae_path: Path to trained SAE checkpoint
            feature_labels: Dictionary mapping feature indices to labels
            device: Device to run on
        """
        self.device = device
        
        # Load MusicGen
        print(f"Loading MusicGen: {model_name}")
        self.model = MusicGen.get_pretrained(model_name, device=device)
        self.model.set_generation_params(use_sampling=True, top_k=250)
        
        # Load SAE if provided
        self.sae = None
        self.sae_type = None
        if sae_path:
            self.sae = self._load_sae(sae_path, device)
            print(f"Loaded SAE ({self.sae_type}) from {sae_path}")
        
        # Feature labels
        self.feature_labels = feature_labels or {}

    def _load_sae(self, sae_path: str, device: str):
        """
        Load SAE model from checkpoint, supporting all variants.

        Args:
            sae_path: Path to SAE checkpoint
            device: Device to load on

        Returns:
            Loaded SAE model
        """
        from .sae import (
            SparseAutoencoder,
            TopKSAE,
            BatchTopKSAE,
            GlobalBatchTopKMatryoshkaSAE
        )

        checkpoint = torch.load(sae_path, map_location=device)

        # Check if it's a vanilla SAE or TopK variant
        if 'config' in checkpoint and 'sae_type' in checkpoint['config']:
            # TopK variant
            config = checkpoint['config']
            self.sae_type = config['sae_type']

            if self.sae_type == 'topk':
                sae = TopKSAE(
                    input_dim=config['input_dim'],
                    hidden_dim=config['hidden_dim'],
                    top_k=config['top_k'],
                    sparsity_coef=config['sparsity_coef'],
                    aux_penalty=config['aux_penalty'],
                    n_batches_to_dead=config['n_batches_to_dead'],
                    input_unit_norm=config['input_unit_norm'],
                    device=device
                )
            elif self.sae_type == 'batch_topk':
                sae = BatchTopKSAE(
                    input_dim=config['input_dim'],
                    hidden_dim=config['hidden_dim'],
                    top_k=config['top_k'],
                    sparsity_coef=config['sparsity_coef'],
                    aux_penalty=config['aux_penalty'],
                    n_batches_to_dead=config['n_batches_to_dead'],
                    input_unit_norm=config['input_unit_norm'],
                    device=device
                )
            elif self.sae_type == 'matryoshka':
                sae = GlobalBatchTopKMatryoshkaSAE(
                    input_dim=config['input_dim'],
                    group_sizes=config['group_sizes'],
                    top_k=config['top_k'],
                    sparsity_coef=config['sparsity_coef'],
                    aux_penalty=config['aux_penalty'],
                    n_batches_to_dead=config['n_batches_to_dead'],
                    input_unit_norm=config['input_unit_norm'],
                    device=device
                )
            else:
                raise ValueError(f"Unknown SAE type: {self.sae_type}")

            sae.load_state_dict(checkpoint['state_dict'])
        else:
            # Vanilla SparseAutoencoder
            self.sae_type = 'vanilla'
            config = checkpoint['config']
            sae = SparseAutoencoder(**config)
            sae.load_state_dict(checkpoint['state_dict'])

        sae.to(device)
        sae.eval()
        return sae

    def generate(
        self,
        prompt: str,
        duration: float = 10.0,
        feature_controls: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Generate music with optional feature control.
        
        Args:
            prompt: Text description
            duration: Audio duration in seconds
            feature_controls: Dict of {feature_name: strength} for control
            
        Returns:
            Generated audio as numpy array
        """
        self.model.set_generation_params(duration=duration)
        
        if feature_controls and self.sae:
            # TODO: Implement SAE-based steering
            # This requires modifying intermediate activations during generation
            print("Feature control not yet implemented")
            wav = self.model.generate([prompt])
        else:
            with torch.no_grad():
                wav = self.model.generate([prompt])
        
        # Convert to numpy
        audio = wav[0].cpu().numpy()
        return audio
    
    def generate_batch(
        self,
        prompts: List[str],
        duration: float = 10.0
    ) -> List[np.ndarray]:
        """
        Generate multiple audio samples.
        
        Args:
            prompts: List of text descriptions
            duration: Audio duration in seconds
            
        Returns:
            List of generated audio arrays
        """
        self.model.set_generation_params(duration=duration)
        
        with torch.no_grad():
            wavs = self.model.generate(prompts)
        
        # Convert to list of numpy arrays
        audios = [wav.cpu().numpy() for wav in wavs]
        return audios
    
    def save_audio(
        self,
        audio: np.ndarray,
        path: str,
        sample_rate: int = 32000
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio array
            path: Output file path
            sample_rate: Audio sample rate
        """
        # Ensure audio is in correct shape [channels, samples]
        if len(audio.shape) == 1:
            audio = audio[np.newaxis, :]
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Save
        torchaudio.save(path, audio_tensor, sample_rate)
        print(f"Audio saved to {path}")


