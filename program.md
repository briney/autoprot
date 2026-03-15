# Autoprot Research Objectives

## Goal

Minimize validation loss (cross-entropy on masked positions) for a protein
masked language model, trained within a 5-minute wall-clock budget per
experiment on a single GPU.

## Architecture

The baseline is an ESM-2-style encoder transformer:
- 6 layers, dim=256, 8 heads (~6.3M parameters)
- SwiGLU FFN, RoPE positional encoding
- Pre-layer normalization
- Weight-tied embedding / LM head

## Experimental Directions

Consider exploring these ideas (one focused change per iteration):

### Hyperparameter Tuning
- Learning rate: try 1e-4 to 1e-3 range
- Warmup steps: 50-500
- Batch size: 32, 64, 128
- Dropout: 0.0 to 0.2
- Weight decay: 0.0 to 0.1

### Architecture Modifications
- Depth vs width tradeoffs (e.g., 4 layers @ dim=384 vs 8 layers @ dim=192)
- FFN multiplier (2x to 6x)
- Number of attention heads
- Pre-norm vs post-norm

### Optimizer Experiments
- AdamW vs Muon optimizer
- Muon learning rate and momentum tuning
- Cosine vs linear LR decay
- Gradient clipping values

### Advanced Ideas
- Larger masking probability (20-25%)
- Layer-wise LR decay
- Embedding scaling factor
- Attention head dimension adjustments
- Stochastic depth / layer dropout

## Constraints

- The `train()` function signature must not change
- Keep model size reasonable (under ~50M params) for the time budget
- Import from autoprot.prepare for tokenizer, data loading, and evaluation
- Focus on single, testable changes per iteration
