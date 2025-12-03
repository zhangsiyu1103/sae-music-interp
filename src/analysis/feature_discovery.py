"""
Automated feature discovery and labeling for SAE features.

This module provides tools to:
1. Filter features by interpretability metrics
2. Auto-label features using LLMs
4. Analyze feature composition
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from tqdm import tqdm
import gc


def _clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class FeatureDiscovery:
    """
    Automated discovery and labeling of SAE features.
    """

    def __init__(
        self,
        sae_model,
        device: str = 'cuda'
    ):
        """
        Initialize feature discovery pipeline.

        Args:
            sae_model: Trained SAE instance (any variant)
            device: Device to use
        """
        self.sae = sae_model
        self.device = device
        self.sae.to(device)
        self.sae.eval()

        # Determine SAE type and hidden dimension
        if hasattr(self.sae, 'total_dict_size'):
            # Matryoshka SAE
            self.hidden_dim = self.sae.total_dict_size
        elif hasattr(self.sae, 'hidden_dim'):
            # TopK variants
            self.hidden_dim = self.sae.hidden_dim
        else:
            # Vanilla SAE
            self.hidden_dim = self.sae.hidden_dim

        # Cached feature activations (computed once, reused)
        self._cached_feature_acts = None
        self._cached_activations_hash = None


    def _build_sample_boundaries(
        self,
        batched_metadata: List[Dict]
    ) -> Tuple[List[int], List[Dict]]:
        """
        Build token boundaries for each sample from batched metadata.

        Args:
            batched_metadata: List of batch dicts with 'prompts' and 'seq_len'

        Returns:
            Tuple of:
                - sample_starts: List of starting token indices for each sample
                - flat_metadata: Flattened list of {'prompt': ..., 'batch_idx': ..., 'sample_idx': ...}
        """
        sample_starts = []
        flat_metadata = []
        current_token_idx = 0

        for batch_idx, batch_meta in enumerate(batched_metadata):
            if not isinstance(batch_meta, dict):
                continue

            # Handle batched format
            if 'prompts' in batch_meta:
                prompts = batch_meta['prompts']
                seq_len = batch_meta.get('seq_len', 1)

                for sample_idx, prompt in enumerate(prompts):
                    sample_starts.append(current_token_idx)
                    flat_metadata.append({
                        'prompt': prompt,
                        'batch_idx': batch_idx,
                        'sample_idx_in_batch': sample_idx,
                        'seq_len': seq_len
                    })
                    current_token_idx += seq_len

            # Handle already flat format
            elif 'prompt' in batch_meta:
                seq_len = batch_meta.get('seq_len', 1)
                sample_starts.append(current_token_idx)
                flat_metadata.append({
                    'prompt': batch_meta['prompt'],
                    'batch_idx': batch_idx,
                    'sample_idx_in_batch': 0,
                    'seq_len': seq_len
                })
                current_token_idx += seq_len

        return sample_starts, flat_metadata

    def _token_idx_to_sample(
        self,
        token_idx: int,
        sample_starts: List[int],
        flat_metadata: List[Dict]
    ) -> Tuple[int, int, Dict]:
        """
        Map a flattened token index to sample index and token position.

        Args:
            token_idx: Flattened token index
            sample_starts: List of starting token indices for each sample
            flat_metadata: Flattened metadata list

        Returns:
            Tuple of (sample_idx, token_position_in_sample, metadata_dict)
        """
        # Binary search to find which sample this token belongs to
        sample_idx = 0
        for i, start in enumerate(sample_starts):
            if start > token_idx:
                break
            sample_idx = i

        token_position = token_idx - sample_starts[sample_idx]
        return sample_idx, token_position, flat_metadata[sample_idx]

    def get_feature_activations(
        self,
        activations: torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Get feature activations for input data.
        Works with all SAE variants.

        Args:
            activations: Input activations [num_samples, input_dim]
            batch_size: Batch size for processing
            use_cache: Whether to use/store cached results

        Returns:
            Feature activations [num_samples, hidden_dim]
        """
        # Check cache
        if self._cached_feature_acts is not None:
            print("  Using cached feature activations")
            return self._cached_feature_acts

        self.sae.eval()
        num_batches = (len(activations) + batch_size - 1) // batch_size

        print(f"  Processing {len(activations)} activations in {num_batches} batches...")

        # Get feature dimension from first batch to pre-allocate
        with torch.no_grad():
            x_sample = activations[0:1].to(self.device)

            # Handle different SAE types
            if hasattr(self.sae, 'encode'):
                output = self.sae.encode(x_sample)
                if isinstance(output, dict):
                    z_sample = output['z']
                else:
                    z_sample = output
            elif hasattr(self.sae, 'encoder'):
                result = self.sae(x_sample)
                if isinstance(result, tuple):
                    _, z_sample = result
                else:
                    z_sample = result['z'] if isinstance(result, dict) else result
            else:
                raise RuntimeError(f"Unknown SAE type: {type(self.sae)}")

            hidden_dim = z_sample.shape[-1]
            del z_sample, x_sample
            _clear_gpu_memory()

        # Pre-allocate result tensor in CPU memory
        print(f"  Pre-allocating result tensor: [{len(activations)}, {hidden_dim}]")
        result = torch.zeros(len(activations), hidden_dim, dtype=torch.float32)

        # Process in chunks and fill result directly
        print(f"  Computing features (memory-efficient mode)...")
        with torch.no_grad():
            for i in tqdm(range(0, len(activations), batch_size), desc="Computing features", disable=num_batches < 10):
                x_batch = activations[i:i+batch_size].to(self.device)

                # Handle different SAE types
                if hasattr(self.sae, 'encode'):
                    output = self.sae.encode(x_batch)
                    if isinstance(output, dict):
                        z = output['z']
                    else:
                        z = output
                elif hasattr(self.sae, 'encoder'):
                    sae_result = self.sae(x_batch)
                    if isinstance(sae_result, tuple):
                        _, z = sae_result
                    else:
                        z = sae_result['z'] if isinstance(sae_result, dict) else sae_result
                else:
                    raise RuntimeError(f"Unknown SAE type: {type(self.sae)}")

                # Copy directly to pre-allocated result
                end_idx = min(i + batch_size, len(activations))
                result[i:end_idx] = z.cpu()

                # Clear GPU memory periodically
                del z, x_batch
                if i % (batch_size * 5) == 0:
                    _clear_gpu_memory()

        _clear_gpu_memory()
        print(f"  Feature activations computed: {result.shape}")

        # Store in cache
        self._cached_feature_acts = result

        return result

    def clear_cache(self):
        """Clear cached feature activations."""
        self._cached_feature_acts = None
        self._cached_activations_hash = None

    def filter_important_features(
        self,
        activations: torch.Tensor,
        min_activation_count: int = 10,
        max_activation_count: int = 10000,
    ) -> List[int]:
        """
        Filter features based on minimum and maximum activation count.

        Args:
            activations: Input activations [num_samples, input_dim]
            min_activation_count: Minimum number of times a feature must activate
            max_activation_count: Maximum number of times a feature can activate

        Returns:
            List of important feature indices
        """
        print("Filtering important features...")

        # Get feature activations
        feature_acts = self.get_feature_activations(activations)
        num_samples = feature_acts.shape[0]
        num_features = feature_acts.shape[1]

        # Count how many times each feature activates (activation > 0)
        # Process in chunks to reduce memory usage
        print("  Computing activation counts...")
        chunk_size = 10000
        activation_counts = np.zeros(num_features, dtype=np.int64)

        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk = feature_acts[i:end_idx]
            activation_counts += (chunk > 0).sum(dim=0).cpu().numpy()
            del chunk

        _clear_gpu_memory()

        print("activation_counts", activation_counts)
        # Print statistics
        print("\n" + "="*70)
        print("FEATURE ACTIVATION STATISTICS")
        print("="*70)
        print(f"\nTotal samples: {num_samples}")
        print(f"Total features: {num_features}")
        print(f"\nActivation counts:")
        print(f"   Min:        {activation_counts.min()}")
        print(f"   Max:        {activation_counts.max()}")
        print(f"   Mean:       {activation_counts.mean():.1f}")
        print(f"   Median:     {np.median(activation_counts):.1f}")
        print(f"   Std:        {activation_counts.std():.1f}")
        print(f"\nDistribution:")
        for threshold in [1, 5, 10, 20, 50, 100, 500, 1000,10000, 100000]:
            count = np.sum(activation_counts >= threshold)
            print(f"   >= {threshold:4d} activations: {count:5d} features ({100*count/num_features:.1f}%)")

        print("\n" + "="*70)
        print("APPLYING FILTER")
        print("="*70)
        print(f"\nFilter: features with {min_activation_count} <= activations <= {max_activation_count}")

        # Apply filter
        important_mask = (activation_counts >= min_activation_count) & (activation_counts <= max_activation_count)
        important_indices = np.where(important_mask)[0].tolist()

        print(f"Result: {len(important_indices)} / {num_features} features selected ({100*len(important_indices)/num_features:.1f}%)")

        if len(important_indices) == 0:
            print("\nWARNING: No features passed the filter!")
            print(f"   Try adjusting --min_activation_count (current: {min_activation_count})")
            print(f"   or --max_activation_count (current: {max_activation_count})")
            print("   Or disable filtering with --no_filter")

        print("="*70 + "\n")

        return important_indices
    
    def get_top_activating_examples(
            self,
            feature_idx: int,
            activations: torch.Tensor,
            music_metadata: List[Dict],
            feature_acts: Optional[torch.Tensor] = None,
            k: int = 10,
            aggregation: str = 'max'  # 'max', 'mean', or 'moments'
        ) -> List[Dict]:
        """
        Find examples where feature activates strongly.

        Handles variable seq_len across batches by using metadata to map indices.

        Args:
            feature_idx: Index of the feature to analyze
            activations: Flattened activations [total_tokens, hidden_dim]
            music_metadata: Batched metadata list, each dict has:
                - 'prompts': List of prompts in this batch
                - 'seq_len': Sequence length for this batch
                OR already flat format with 'prompt' key
            feature_acts: Pre-computed feature activations (optional, for efficiency)
            k: Number of top examples to return
            aggregation:
                'max' - sample-level max pooling (default)
                'mean' - sample-level mean pooling
                'moments' - token-level with temporal info

        Returns:
            List of top activating examples with metadata
        """
        # Build sample boundaries from metadata (handles variable seq_len)
        sample_starts, flat_metadata = self._build_sample_boundaries(music_metadata)
        num_samples = len(flat_metadata)

        if num_samples == 0:
            return []

        # Get or compute feature activations (uses cache)
        if feature_acts is None:
            feature_acts = self.get_feature_activations(activations)

        # Extract activations for this specific feature
        feat_acts = feature_acts[:, feature_idx]  # [total_tokens]

        if aggregation in ['max', 'mean']:
            # Aggregate per sample without reshaping
            sample_acts = []
            for i in range(num_samples):
                start_idx = sample_starts[i]
                seq_len = flat_metadata[i]['seq_len']
                end_idx = start_idx + seq_len

                sample_feature_acts = feat_acts[start_idx:end_idx]

                if aggregation == 'max':
                    agg_val = sample_feature_acts.max().item()
                else:  # mean
                    agg_val = sample_feature_acts.mean().item()

                sample_acts.append(agg_val)

            # Find top k samples
            sample_acts_tensor = torch.tensor(sample_acts)
            k = min(k, (sample_acts_tensor>0).sum().item())
            top_indices = sample_acts_tensor.argsort(descending=True)[:k]

            examples = []
            for idx in top_indices:
                idx = int(idx)
                examples.append({
                    'index': idx,
                    'activation': sample_acts[idx],
                    'metadata': {'prompt': flat_metadata[idx]['prompt']},
                    'batch_idx': flat_metadata[idx]['batch_idx'],
                    'sample_idx_in_batch': flat_metadata[idx]['sample_idx_in_batch'],
                    'type': 'sample'
                })

        elif aggregation == 'moments':
            # Token-level: find top k tokens across all samples
            top_token_indices = feat_acts.argsort(descending=True)[:k]

            examples = []
            for token_idx in top_token_indices:
                token_idx = int(token_idx)

                # Map token index to sample
                sample_idx, token_position, meta = self._token_idx_to_sample(
                    token_idx, sample_starts, flat_metadata
                )

                # Estimate time (assuming ~50 tokens/sec for MusicGen)
                time_seconds = token_position / 50.0

                examples.append({
                    'sample_index': sample_idx,
                    'token_index': token_position,
                    'time_seconds': time_seconds,
                    'activation': float(feat_acts[token_idx]),
                    'metadata': {'prompt': meta['prompt']},
                    'batch_idx': meta['batch_idx'],
                    'sample_idx_in_batch': meta['sample_idx_in_batch'],
                    'type': 'moment'
                })

        return examples



    # def cluster_features(
    #     self,
    #     activations: torch.Tensor,
    #     feature_indices: Optional[List[int]] = None,
    #     n_clusters: int = 50
    # ) -> Tuple[Dict[int, int], List[int]]:
    #     """
    #     Cluster features by activation patterns.
        
    #     Args:
    #         activations: Input activations
    #         feature_indices: Subset of features to cluster (or None for all)
    #         n_clusters: Number of clusters
            
    #     Returns:
    #         Tuple of (feature_to_cluster mapping, representative indices)
    #     """
    #     print(f"Clustering features into {n_clusters} groups...")

    #     # Get feature activations
    #     feature_acts = self.get_feature_activations(activations)
        
    #     if feature_indices is not None:
    #         feature_acts = feature_acts[:, feature_indices]
    #         all_indices = feature_indices
    #     else:
    #         all_indices = list(range(feature_acts.shape[1]))
        
    #     # Compute correlation matrix
    #     print("Computing feature correlations...")
    #     corr_matrix = np.corrcoef(feature_acts.T.cpu().numpy())

    #     # Handle NaN values in correlation matrix
    #     corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    #     # Cluster based on correlations
    #     kmeans = KMeans(n_clusters=min(n_clusters, len(all_indices)), random_state=42, n_init=10)
    #     cluster_labels = kmeans.fit_predict(corr_matrix)
        
    #     # Map features to clusters
    #     feature_to_cluster = {
    #         feat_idx: cluster_id 
    #         for feat_idx, cluster_id in zip(all_indices, cluster_labels)
    #     }
        
    #     # Select representative from each cluster
    #     representatives = []
    #     for cluster_id in range(n_clusters):
    #         # Get features in this cluster
    #         cluster_mask = cluster_labels == cluster_id
    #         cluster_indices = np.where(cluster_mask)[0]
            
    #         if len(cluster_indices) == 0:
    #             continue
            
    #         # Pick the most "central" feature
    #         cluster_corr = corr_matrix[cluster_mask][:, cluster_mask]
    #         centrality = cluster_corr.mean(axis=1)
    #         representative_local_idx = np.argmax(centrality)
    #         representative_idx = all_indices[cluster_indices[representative_local_idx]]
            
    #         representatives.append(representative_idx)
        
    #     print(f"Selected {len(representatives)} representative features")
        
    #     return feature_to_cluster, representatives

    def auto_label_feature(
        self,
        feature_idx: int,
        activations: torch.Tensor,
        music_metadata: List[Dict],
        feature_acts: Optional[torch.Tensor] = None,
        use_llm: bool = False,
        llm_api: Optional[str] = None
    ) -> str:
        """
        Automatically generate label for a feature.

        Args:
            feature_idx: Feature index
            activations: Input activations
            music_metadata: Batched metadata list (with 'prompts' and 'seq_len')
            feature_acts: Pre-computed feature activations (optional)
            use_llm: Whether to use LLM for labeling
            llm_api: API to use ('openai' or 'anthropic')

        Returns:
            Feature label string
        """
        # Get top activating examples
        examples = self.get_top_activating_examples(
            feature_idx, activations, music_metadata, feature_acts=feature_acts, k=10
        )
        
        if not use_llm:
            # Simple heuristic labeling
            # Extract common words from prompts
            prompts = [ex['metadata'].get('prompt', '') for ex in examples]
            words = ' '.join(prompts).lower().split()
            ignore_words = ['music', 'sound', 'feature', 'something', 'this', 'playing', 'song', 'there', "with", "could", "recording", "male", "female", "voice"]
            # Find most common meaningful words
            common_words = {}
            for word in words:
                if len(word) > 3 and word not in ignore_words:  # Ignore short words
                    common_words[word] = common_words.get(word, 0) + 1

            if common_words:
                top_word = max(common_words, key=common_words.get)
                return f"feature_{feature_idx}_{top_word}"
            else:
                return f"feature_{feature_idx}_unknown"

        else:
            # Use LLM for intelligent labeling
            return self._llm_label_feature(examples, feature_idx, llm_api)
    
    def _llm_label_feature(
        self,
        examples: List[Dict],
        feature_idx: int,
        llm_api: str
    ) -> str:
        """
        Use LLM to label feature based on top activating examples.
        
        Args:
            examples: Top activating examples
            feature_idx: Feature index
            llm_api: 'openai' or 'anthropic'
            
        Returns:
            Feature label
        """
        # Format examples for LLM
        examples_text = "\n".join([
            f"{i+1}. {ex['metadata'].get('prompt', 'N/A')} "
            f"(activation: {ex['activation']:.3f})"
            for i, ex in enumerate(examples[:10])
        ])
        
        prompt = f"""I have a feature from a music generation model that activates strongly on these examples:

            {examples_text}

            Based on these examples, what musical concept does this feature represent?
            Provide a concise 2-4 word label.

            Good label examples: "fast tempo", "jazz saxophone", "minor key", "drum break", "vocal harmony"
            Bad label examples: "music", "sound", "feature_{feature_idx}", "something"

            Label:"""
        
        if llm_api == 'openai':
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=20
                )
                label = response.choices[0].message.content.strip()
                return label
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return f"feature_{feature_idx}_llm_error"
        
        elif llm_api == 'anthropic':
            try:
                import anthropic
                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}]
                )
                label = response.content[0].text.strip()
                return label
            except Exception as e:
                print(f"Anthropic API error: {e}")
                return f"feature_{feature_idx}_llm_error"
        
        else:
            return f"feature_{feature_idx}_unknown_api"
    
    def discover_and_label(
        self,
        activations: torch.Tensor,
        music_metadata: List[Dict],
        use_llm: bool = False,
        llm_api: Optional[str] = None,
        filter_features: bool = True,
        min_activation_count: int = 10,
        max_activation_count: int = 10000,
        max_features: int = 10,
        aggregation: str = 'max'
    ) -> Dict[int, Dict]:
        """
        Complete pipeline: filter and label features.

        Args:
            activations: Input activations [total_tokens, hidden_dim]
            music_metadata: Batched metadata list, each dict has:
                - 'prompts': List of prompts in this batch
                - 'seq_len': Sequence length for this batch
            use_llm: Whether to use LLM for labeling
            llm_api: LLM API to use
            filter_features: Whether to filter before labeling
            min_activation_count: Minimum number of times a feature must activate
            max_activation_count: Maximum number of times a feature can activate
            max_features: Maximum number of features to label (default: 10)
            aggregation: Aggregation mode for examples:
                - 'max': Sample-level max pooling (default)
                - 'mean': Sample-level mean pooling
                - 'moments': Token-level with position and timing info

        Returns:
            Dictionary of feature information
        """
        print("Starting feature discovery pipeline...")

        # Pre-compute feature activations once (will be cached and reused)
        print("Computing feature activations...")
        feature_acts = self.get_feature_activations(activations)
        print(f"  Feature activations shape: {feature_acts.shape}")
        _clear_gpu_memory()

        # Step 1: Filter important features (uses cached feature_acts)
        if filter_features:
            important_features = self.filter_important_features(
                activations,
                min_activation_count=min_activation_count,
                max_activation_count=max_activation_count
            )
        else:
            important_features = list(range(self.hidden_dim))

        _clear_gpu_memory()

        # Step 2: Rank features by maximum activation value
        print("\nRanking features by maximum activation value...")
        max_activations = {}
        chunk_size = 10000

        for feat_idx in tqdm(important_features, desc="Computing max activations"):
            max_val = 0.0
            for i in range(0, feature_acts.shape[0], chunk_size):
                end_idx = min(i + chunk_size, feature_acts.shape[0])
                chunk_max = feature_acts[i:end_idx, feat_idx].max().item()
                max_val = max(max_val, chunk_max)
            max_activations[feat_idx] = max_val

        # Sort features by max activation (descending)
        ranked_features = sorted(important_features, key=lambda f: max_activations[f], reverse=True)

        # Print top 10 features with largest activation values
        print("\n" + "="*70)
        print("TOP 10 FEATURES BY MAXIMUM ACTIVATION VALUE")
        print("="*70)
        for i, feat_idx in enumerate(ranked_features[:10], 1):
            print(f"{i:2d}. Feature {feat_idx:5d}: max activation = {max_activations[feat_idx]:.4f}")
        print("="*70 + "\n")

        _clear_gpu_memory()

        # Limit to max_features (from ranked list)
        features_to_label = ranked_features[:max_features]
        print(f"Labeling top {len(features_to_label)} features by activation (out of {len(important_features)} filtered)")

        # Step 3: Label features
        feature_info = {}

        print("Labeling features...")
        for feat_idx in tqdm(features_to_label, desc="Labeling"):
            # Pass pre-computed feature_acts
            label = self.auto_label_feature(
                feat_idx,
                activations,
                music_metadata,
                feature_acts=feature_acts,
                use_llm=use_llm,
                llm_api=llm_api
            )

            # Pass pre-computed feature_acts
            examples = self.get_top_activating_examples(
                feat_idx, activations, music_metadata, feature_acts=feature_acts, k=20, aggregation=aggregation
            )

            feature_info[feat_idx] = {
                'label': label,
                'top_examples': examples
            }

        print(f"Discovered and labeled {len(feature_info)} features")

        return feature_info
    
    def save_features(self, feature_info: Dict, path: str):
        """Save feature information to JSON."""
        # Convert to JSON-serializable format
        json_data = {}
        for k, v in feature_info.items():
            # Copy feature info
            feature_data = {
                'label': v['label'],
                'top_examples': []
            }

            # Preserve token-level information if available
            for ex in v.get('top_examples', []):
                example_data = {
                    'activation': float(ex['activation']),
                    'prompt': ex['metadata'].get('prompt', 'N/A')
                }

                # Add token-level information if this is a 'moments' example
                if ex.get('type') == 'moment':
                    example_data['sample_index'] = ex.get('sample_index')
                    example_data['token_index'] = ex.get('token_index')
                    example_data['time_seconds'] = ex.get('time_seconds')
                    example_data['type'] = 'token-level'
                else:
                    example_data['type'] = 'sample-level'

                feature_data['top_examples'].append(example_data)

            json_data[str(k)] = feature_data

        with open(path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Features saved to {path}")

    def _get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def load_features(path: str) -> Dict[int, Dict]:
        """Load feature information from JSON."""
        with open(path, 'r') as f:
            json_data = json.load(f)
        
        # Convert keys back to integers
        feature_info = {
            int(k): v for k, v in json_data.items()
        }
        
        print(f"Loaded {len(feature_info)} features from {path}")
        return feature_info


if __name__ == "__main__":
    # Example usage
    print("Feature discovery module loaded successfully")
    print("\nExample usage:")
    print("""
    from src.models.sae import SparseAutoencoder
    from src.analysis.feature_discovery import FeatureDiscovery
    
    # Load trained SAE
    sae = SparseAutoencoder.load('models/sae.pt')
    
    # Create discovery pipeline
    discovery = FeatureDiscovery(sae)
    
    # Discover and label features
    features = discovery.discover_and_label(
        activations=activation_data,
        music_metadata=metadata_list,
        use_llm=True,
        llm_api='anthropic'
    )
    
    # Save results
    discovery.save_features(features, 'results/features.json')
    """)
