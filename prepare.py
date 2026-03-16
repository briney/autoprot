"""IMMUTABLE: Tokenizer, data loading, masking, and evaluation.

This file must never be modified by the autonomous agent. It defines the
ground-truth vocabulary, masking strategy, and evaluation metric that all
experiments are measured against.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Vocabulary & Tokenizer
# ---------------------------------------------------------------------------

VOCAB: dict[str, int] = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
}
VOCAB_SIZE = 25
PAD_VOCAB_SIZE = 32  # padded for tensor core alignment

ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}

# Token IDs for convenience
CLS_ID = VOCAB["<cls>"]
PAD_ID = VOCAB["<pad>"]
EOS_ID = VOCAB["<eos>"]
UNK_ID = VOCAB["<unk>"]
MASK_ID = VOCAB["<mask>"]

# Amino acid token ID range (inclusive)
AA_ID_MIN = 5
AA_ID_MAX = 24


def encode(sequence: str) -> list[int]:
    """Encode an amino acid sequence to token IDs with <cls> and <eos>."""
    ids = [CLS_ID]
    for ch in sequence.upper():
        ids.append(VOCAB.get(ch, UNK_ID))
    ids.append(EOS_ID)
    return ids


def decode(token_ids: list[int]) -> str:
    """Decode token IDs back to a string (special tokens included)."""
    return "".join(ID_TO_TOKEN.get(tid, "?") for tid in token_ids)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_fasta(path: Path) -> list[str]:
    """Parse a FASTA file and return a list of amino acid sequences.

    Supports plain (.fasta, .fa) and gzipped (.fasta.gz, .fa.gz) files.
    """
    sequences: list[str] = []
    current: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _glob_fasta(directory: Path) -> list[Path]:
    """Glob for all supported FASTA extensions in a directory."""
    files: list[Path] = []
    for ext in ("*.fasta", "*.fa", "*.fasta.gz", "*.fa.gz"):
        files.extend(directory.glob(ext))
    return sorted(files)


def create_datasets(
    data_dir: Path,
    max_length: int = 512,
) -> tuple[list[list[int]], list[list[int]]]:
    """Load FASTA files from train/ and val/ subdirectories, encode, and filter.

    Args:
        data_dir: Root data directory containing ``train/`` and ``val/`` subdirs.
        max_length: Maximum sequence length (including <cls> and <eos>).

    Returns:
        (train_data, val_data) — each a list of encoded token ID lists.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")

    train_files = _glob_fasta(train_dir)
    if not train_files:
        raise FileNotFoundError(f"No FASTA files found in {train_dir}")

    val_files = _glob_fasta(val_dir)
    if not val_files:
        raise FileNotFoundError(f"No FASTA files found in {val_dir}")

    def _load_and_encode(files: list[Path]) -> list[list[int]]:
        sequences: list[str] = []
        for fp in files:
            sequences.extend(load_fasta(fp))
        encoded = [encode(seq) for seq in sequences]
        return [ids for ids in encoded if len(ids) <= max_length]

    train_data = _load_and_encode(train_files)
    if not train_data:
        raise ValueError(
            f"No training sequences remaining after filtering to max_length={max_length}"
        )

    val_data = _load_and_encode(val_files)
    if not val_data:
        raise ValueError(
            f"No validation sequences remaining after filtering to max_length={max_length}"
        )

    return train_data, val_data


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def apply_mask(
    token_ids: Tensor,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply ESM-2 style masking to a batch of token IDs.

    Of the selected positions:
      - 80% are replaced with <mask>
      - 10% are replaced with a random amino acid
      - 10% are left unchanged

    Only amino acid positions are eligible for masking (not special tokens).

    Args:
        token_ids: (B, L) tensor of token IDs.
        mask_prob: Probability of selecting each AA position.
        generator: Optional RNG for deterministic masking.

    Returns:
        (masked_input_ids, labels) where labels = -100 at non-masked positions.
    """
    masked = token_ids.clone()
    labels = torch.full_like(token_ids, -100)

    # Only mask amino acid positions
    aa_mask = (token_ids >= AA_ID_MIN) & (token_ids <= AA_ID_MAX)

    # Select positions to mask
    prob_matrix = torch.full(token_ids.shape, mask_prob, device=token_ids.device)
    prob_matrix[~aa_mask] = 0.0
    selected = torch.bernoulli(prob_matrix, generator=generator).bool()

    # Set labels at selected positions
    labels[selected] = token_ids[selected]

    # 80% of selected → <mask>
    mask_replace = (
        torch.bernoulli(
            torch.full(token_ids.shape, 0.8, device=token_ids.device), generator=generator
        ).bool()
        & selected
    )
    masked[mask_replace] = MASK_ID

    # 10% of selected (and not already replaced) → random AA
    random_replace = (
        torch.bernoulli(
            torch.full(token_ids.shape, 0.5, device=token_ids.device),
            generator=generator,
        ).bool()
        & selected
        & ~mask_replace
    )
    random_tokens = torch.randint(
        AA_ID_MIN,
        AA_ID_MAX + 1,
        token_ids.shape,
        device=token_ids.device,
        generator=generator,
    )
    masked[random_replace] = random_tokens[random_replace]

    # Remaining 10% of selected → unchanged (already handled by clone)

    return masked, labels


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Simple dataset wrapping a list of encoded token ID sequences."""

    def __init__(self, data: list[list[int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[int]:
        return self.data[idx]


def collate_fn(batch: list[list[int]]) -> dict[str, Tensor]:
    """Pad sequences to max-in-batch, create attention mask, and apply masking.

    Returns:
        Dict with keys: input_ids, attention_mask, labels
    """
    max_len = max(len(seq) for seq in batch)

    padded = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, seq in enumerate(batch):
        length = len(seq)
        padded[i, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :length] = 1

    masked_input_ids, labels = apply_mask(padded)

    return {
        "input_ids": masked_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# DataLoader Helper
# ---------------------------------------------------------------------------


def create_dataloader(
    data: list[list[int]],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with masking collation."""
    dataset = SequenceDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    val_data: list[list[int]],
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """Compute average cross-entropy loss on masked positions.

    Uses a fixed random seed for masking so results are deterministic
    and comparable across experiments.

    Args:
        model: The protein language model to evaluate.
        val_data: List of encoded token ID sequences.
        batch_size: Batch size for evaluation.
        device: Device to run on.

    Returns:
        Average validation loss (float).
    """
    model.eval()

    # Fixed seed for deterministic masking during evaluation
    old_rng_state = torch.random.get_rng_state()
    torch.manual_seed(12345)

    total_loss = 0.0
    total_tokens = 0

    loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attention_mask)

        # Cross-entropy on masked positions only
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        n_masked = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_masked

    # Restore RNG state
    torch.random.set_rng_state(old_rng_state)

    model.train()

    if total_tokens == 0:
        return float("inf")

    return total_loss / total_tokens
