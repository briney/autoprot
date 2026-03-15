"""MUTABLE: Model architecture, optimizer, and training loop.

This is the file the autonomous agent modifies between experiments.
It contains the full model definition and training loop.
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prepare import (
    PAD_ID,
    PAD_VOCAB_SIZE,
    create_dataloader,
    evaluate_loss,
)

logger = logging.getLogger(__name__)

# === HYPERPARAMETERS (agent-tunable) ===
DIM = 256
N_LAYERS = 6
N_HEADS = 8
FFN_MULT = 4
DROPOUT = 0.1
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
OPTIMIZER = "adamw"  # "adamw" or "muon"
NORM_TYPE = "pre"  # "pre" or "post"


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings)
# ---------------------------------------------------------------------------


def build_rope_cache(
    dim: int, seq_len: int, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Args:
        dim: Head dimension (must be even).
        seq_len: Maximum sequence length.
        device: Target device.

    Returns:
        (cos, sin) each of shape (seq_len, dim).
    """
    theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)  # (seq_len, dim//2)
    cos = freqs.cos().repeat(1, 2)  # (seq_len, dim) — duplicate for pairs
    sin = freqs.sin().repeat(1, 2)
    return cos, sin


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: (B, H, L, D_head)
        k: (B, H, L, D_head)
        cos: (L, D_head) or broadcastable
        sin: (L, D_head) or broadcastable

    Returns:
        Rotated (q, k) with same shapes.
    """
    seq_len = q.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU activation (LLaMA-style)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # value
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down-project
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} not divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: (B, L, D)
            cos, sin: RoPE tables
            attention_mask: (B, L) with 1 for real tokens, 0 for padding
        """
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, H, L, D_head)

        q, k = apply_rope(q, k, cos, sin)

        # Build causal-free attention mask for padding
        attn_mask = None
        if attention_mask is not None:
            # (B, L) -> (B, 1, 1, L) for broadcasting
            attn_mask = attention_mask[:, None, None, :].bool()
            # Convert: True where we attend, so invert for SDPA which masks where True
            attn_mask = ~attn_mask  # now True = ignore
            attn_mask = attn_mask.expand(B, self.n_heads, L, L)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # encoder, not decoder
        )
        # out: (B, H, L, D_head) -> (B, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer block with configurable pre/post-norm."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        norm_type: str = "pre",
    ) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.ffn = SwiGLU(dim, ffn_mult, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if self.norm_type == "pre":
            x = x + self.attn(self.norm1(x), cos, sin, attention_mask)
            x = x + self.ffn(self.norm2(x))
        else:  # post-norm
            x = self.norm1(x + self.attn(x, cos, sin, attention_mask))
            x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# ProteinLM Model
# ---------------------------------------------------------------------------


class ProteinLM(nn.Module):
    """Protein masked language model with RoPE and SwiGLU.

    Args:
        dim: Model dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        ffn_mult: FFN hidden dim multiplier.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE cache.
        norm_type: "pre" or "post" layer normalization.
    """

    def __init__(
        self,
        dim: int = DIM,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        ffn_mult: int = FFN_MULT,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
        norm_type: str = NORM_TYPE,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(PAD_VOCAB_SIZE, dim, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ffn_mult, dropout, norm_type) for _ in range(n_layers)]
        )

        # Final norm (needed for pre-norm architecture)
        self.final_norm = nn.LayerNorm(dim)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(dim, PAD_VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE cache
        head_dim = dim // n_heads
        self.register_buffer("rope_cos", torch.zeros(max_seq_len, head_dim))
        self.register_buffer("rope_sin", torch.zeros(max_seq_len, head_dim))
        self._init_rope(head_dim, max_seq_len)

    def _init_rope(self, head_dim: int, max_seq_len: int) -> None:
        cos, sin = build_rope_cache(head_dim, max_seq_len, device="cpu")
        self.rope_cos.copy_(cos)
        self.rope_sin.copy_(sin)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (B, L) token IDs.
            attention_mask: (B, L) with 1=real, 0=pad.

        Returns:
            Logits of shape (B, L, PAD_VOCAB_SIZE).
        """
        x = self.embed_dropout(self.embed(input_ids))

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, attention_mask)

        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        """Count trainable parameters (excluding tied LM head)."""
        seen = set()
        total = 0
        for name, p in self.named_parameters():
            if p.data_ptr() not in seen and p.requires_grad:
                seen.add(p.data_ptr())
                total += p.numel()
        return total


# ---------------------------------------------------------------------------
# Muon Optimizer
# ---------------------------------------------------------------------------


def _newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration for approximate matrix orthogonalization."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D weight matrices.

    Uses Newton-Schulz orthogonalization of gradients.
    """

    def __init__(
        self,
        params: list,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Apply weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Orthogonalize gradient via Newton-Schulz
                if g.ndim == 2:
                    g = _newton_schulz_5(g, steps=ns_steps)

                # Momentum
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                p.add_(buf, alpha=-lr)


def _build_optimizer(
    model: ProteinLM, optimizer_type: str, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build optimizer(s) for the model.

    When optimizer_type="muon", uses Muon for 2D weight matrices and
    AdamW for everything else (embeddings, LayerNorm, biases).
    """
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Muon + AdamW split
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "norm" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Use a combined optimizer wrapper
    class CombinedOptimizer:
        """Wraps Muon + AdamW into a single interface."""

        def __init__(self) -> None:
            self.muon = Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=weight_decay)
            self.adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
            self.param_groups = self.muon.param_groups + self.adamw.param_groups

        def zero_grad(self) -> None:
            self.muon.zero_grad()
            self.adamw.zero_grad()

        def step(self) -> None:
            self.muon.step()
            self.adamw.step()

        def state_dict(self) -> dict:
            return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    return CombinedOptimizer()


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------


def _get_lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train(
    train_data: list[list[int]],
    val_data: list[list[int]],
    max_seconds: int = 300,
    device: str = "cuda",
) -> dict[str, float | int]:
    """Train the protein language model.

    Args:
        train_data: List of encoded token ID sequences for training.
        val_data: List of encoded token ID sequences for validation.
        max_seconds: Wall-clock time budget in seconds.
        device: Device to train on.

    Returns:
        Dict with val_loss, train_loss, steps, and params.
    """
    # Build model
    model = ProteinLM(
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_mult=FFN_MULT,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
        norm_type=NORM_TYPE,
    ).to(device)

    n_params = model.num_parameters()
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = _build_optimizer(model, OPTIMIZER, LR, WEIGHT_DECAY)

    # Training
    loader = create_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    step = 0
    running_loss = 0.0
    log_interval = 50
    total_estimate = max_seconds * 3  # rough step estimate for LR schedule
    start_time = time.time()

    model.train()
    while True:
        for batch in loader:
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            opt_for_scaler = (
                optimizer if isinstance(optimizer, torch.optim.Optimizer) else optimizer.adamw
            )
            scaler.unscale_(opt_for_scaler)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt_for_scaler)
            if not isinstance(optimizer, torch.optim.Optimizer):
                optimizer.muon.step()
            scaler.update()

            # LR schedule
            lr_scale = _get_lr_scale(step, WARMUP_STEPS, total_estimate)
            for pg in optimizer.param_groups if hasattr(optimizer, "param_groups") else []:
                pg["lr"] = LR * lr_scale

            running_loss += loss.item()
            step += 1

            if step % log_interval == 0:
                avg = running_loss / log_interval
                logger.info(f"step={step}  train_loss={avg:.4f}  elapsed={elapsed:.0f}s")
                running_loss = 0.0

        if time.time() - start_time >= max_seconds:
            break

    # Final train loss
    train_loss = running_loss / max(1, step % log_interval) if step % log_interval != 0 else 0.0
    if step >= log_interval and train_loss == 0.0:
        train_loss = running_loss / log_interval if running_loss > 0 else 0.0

    # Evaluate
    val_loss = evaluate_loss(model, val_data, batch_size=BATCH_SIZE, device=device)
    logger.info(f"val_loss={val_loss:.4f} after {step} steps")

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "steps": step,
        "params": n_params,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from prepare import create_datasets

    # === RUN CONFIGURATION ===
    DATA_DIR = Path("data")
    VAL_FRACTION = 0.1
    TIME_BUDGET = 300  # seconds
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    train_data, val_data = create_datasets(
        DATA_DIR, val_fraction=VAL_FRACTION, max_length=MAX_SEQ_LEN
    )
    print(
        f"Train: {len(train_data)} sequences, Val: {len(val_data)} sequences",
        file=sys.stderr,
    )

    t0 = time.time()
    result = train(train_data, val_data, max_seconds=TIME_BUDGET, device=DEVICE)
    total = time.time() - t0

    # Print results in parseable format (agent greps for ^val_loss:)
    print("---")
    print(f"val_loss:            {result['val_loss']:.6f}")
    print(f"train_loss:          {result['train_loss']:.6f}")
    print(f"training_seconds:    {TIME_BUDGET}")
    print(f"total_seconds:       {total:.1f}")
    print(f"num_steps:           {result['steps']}")
    print(f"num_params:          {result['params']}")
