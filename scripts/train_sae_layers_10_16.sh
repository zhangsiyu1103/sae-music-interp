#!/bin/bash
for layer in 12 13 14 15 16 17 18; do
    python -m src.training.train \
        --layer_idx ${layer} \
        --sae_type batch_topk \
        --top_k 64 \
        --epochs 5 \
        --hidden_dim 6144\
        --aux_penalty 0.1\
        --batch_size 256 \
        --resample_interval 0
done