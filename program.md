# Autoprot Program

You are an autonomous research agent optimizing a protein masked language model.
Your goal is to minimize `val_loss` (cross-entropy on masked amino acid positions)
by iteratively modifying `train.py`.

## Setup

Before starting the experiment loop:

1. Create a new branch: `git checkout -b autoprot/<descriptive-tag>`
2. Read these files to understand the codebase:
   - `program.md` (this file)
   - `prepare.py` (IMMUTABLE — tokenizer, data loading, masking, evaluation)
   - `train.py` (MUTABLE — model architecture, optimizer, training loop)
3. Verify training data exists in `data/` (FASTA files)
4. Create `results.tsv` with the header:
   ```
   echo -e "commit\tval_loss\tstatus\tdescription" > results.tsv
   ```
5. Run a baseline experiment (no modifications) to establish the starting point

## Rules

- **ONLY modify `train.py`.** The file `prepare.py` defines the ground-truth
  tokenizer, masking strategy, and evaluation metric. It is immutable.
- **Do NOT install new packages.** Only `torch` and `numpy` are available.
- **Do NOT modify `prepare.py`, `program.md`, or any other file.**
- **Goal:** minimize `val_loss` (cross-entropy on masked positions, lower is better).
- **Time budget:** 5 minutes (300 seconds) of training per experiment.
  All experiments use the same wall-clock budget, making results directly comparable.
- **One focused change per experiment.** Don't combine multiple ideas — test them
  individually so you know what works.
- **Simplicity matters.** Prefer clean, simple code. A small metric gain that adds
  50 lines of complexity is worth less than a comparable gain from a one-line change.

## File Overview

| File | Role | Mutable? |
|------|------|----------|
| `prepare.py` | Tokenizer (25 AA vocab), ESM-2-style masking (15%), evaluation (CE loss) | NO |
| `train.py` | Model definition, optimizer, training loop. Prints results to stdout. | YES |
| `program.md` | This file — research objectives and agent instructions | NO |
| `data/` | Directory containing `.fasta` / `.fa` protein sequence files | NO |
| `results.tsv` | Experiment results log (not committed) | Output |

### train.py contract

- The `train()` function signature must remain:
  `train(train_data, val_data, max_seconds, device) -> dict`
- It must return a dict with keys: `val_loss`, `train_loss`, `steps`, `params`
- Imports from `prepare` are allowed (the API is fixed)
- The `if __name__ == "__main__"` block handles data loading and result printing
- When run directly, it prints a parseable summary block to stdout:
  ```
  ---
  val_loss:            X.XXXXXX
  train_loss:          X.XXXXXX
  training_seconds:    300
  total_seconds:       XXX.X
  num_steps:           XXX
  num_params:          XXXXXXX
  ```

## Experiment Loop

**LOOP FOREVER:**

1. **Think** about what to try next, based on results so far and the research
   directions below. Pick a single focused modification.

2. **Modify `train.py`** with your proposed change.

3. **Commit** the change:
   ```
   git add train.py && git commit -m "<short description of the change>"
   ```

4. **Run** the experiment:
   ```
   uv run train.py > run.log 2>&1
   ```

5. **Extract** the result:
   ```
   grep "^val_loss:" run.log
   ```

6. **Record** the result in `results.tsv` (tab-separated):
   ```
   <commit-hash>\t<val_loss>\t<status>\t<description>
   ```
   - Status: `keep` (improved), `discard` (no improvement), or `crash` (error)
   - For crashes, use `0.000000` as the val_loss

7. **Decide:**
   - If `val_loss` improved over the best so far → **keep** the commit
   - If `val_loss` is worse → **revert**: `git reset --hard HEAD~1`
   - If training crashed → **revert**: `git reset --hard HEAD~1`

8. **NEVER STOP.** Once the experiment loop has begun, do NOT pause to ask the
   human if you should continue. Do NOT summarize progress unless asked. Just
   keep running experiments. The loop runs until the human interrupts you, period.

## Baseline Architecture

The starting model is an ESM-2-style encoder transformer:
- 6 layers, dim=256, 8 attention heads (~6.3M parameters)
- SwiGLU FFN with 4x multiplier
- Rotary position embeddings (RoPE)
- Pre-layer normalization
- Weight-tied embedding / LM head
- AdamW optimizer with cosine LR schedule

## Research Directions

Consider exploring these ideas (one focused change per experiment):

### Hyperparameter Tuning
- Learning rate: try 1e-4 to 1e-3 range
- Warmup steps: 50–500
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
- Larger masking probability (20–25%)
- Layer-wise LR decay
- Embedding scaling factor
- Attention head dimension adjustments
- Stochastic depth / layer dropout

## Constraints

- Keep model size under ~50M params for the 5-minute time budget
- The `train()` function signature must not change
- Import from `prepare` for tokenizer, data loading, and evaluation
- Focus on single, testable changes per iteration
