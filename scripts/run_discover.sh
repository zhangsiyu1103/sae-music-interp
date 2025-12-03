#!/bin/bash
for layer in  12 13 14 15 16 17 18; do
 python -m src.analysis.discover    \
     --sae_path models/batch_topk/layer${layer}/dim6144/sae_final.pt \
     --activation_file data/activations/activations_layer${layer}.pt \
     --output results/features_layer${layer}.json \
     --min_activation_count 1000

done