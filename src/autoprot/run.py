"""Autonomous experiment loop orchestrator.

Iteratively modifies train.py using an LLM, trains for a fixed time budget,
evaluates against an immutable metric, and keeps improvements or reverts failures.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from autoprot.prepare import create_datasets

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

logger = logging.getLogger("autoprot")

TRAIN_PY = Path(__file__).parent / "train.py"
RESULTS_FILE = Path("results.tsv")


# ---------------------------------------------------------------------------
# Git Integration
# ---------------------------------------------------------------------------


def git_commit(message: str) -> None:
    """Stage and commit train.py."""
    subprocess.run(
        ["git", "add", str(TRAIN_PY)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        check=True,
        capture_output=True,
    )


def git_revert() -> None:
    """Revert train.py to the last committed state."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--", str(TRAIN_PY)],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Results Logging
# ---------------------------------------------------------------------------


def log_result(
    iteration: int,
    val_loss: float | str,
    train_loss: float | str,
    steps: int | str,
    params: int | str,
    description: str,
) -> None:
    """Append a result row to results.tsv."""
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "iteration\ttimestamp\tval_loss\ttrain_loss\tsteps\tparams\tdescription\n"
        )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    row = f"{iteration}\t{timestamp}\t{val_loss}\t{train_loss}\t{steps}\t{params}\t{description}\n"
    with open(RESULTS_FILE, "a") as f:
        f.write(row)


def read_last_results(n: int = 20) -> str:
    """Read the last N rows from results.tsv."""
    if not RESULTS_FILE.exists():
        return "(no previous results)"
    lines = RESULTS_FILE.read_text().strip().split("\n")
    # Header + last N data rows
    if len(lines) <= n + 1:
        return "\n".join(lines)
    return "\n".join([lines[0]] + lines[-(n):])


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------


def call_llm(program: str, results: str, current_train: str) -> str:
    """Call Claude to generate a modified train.py.

    Args:
        program: Contents of program.md (research objectives).
        results: Recent results from results.tsv.
        current_train: Current source of train.py.

    Returns:
        Modified train.py source code.
    """
    import anthropic

    client = anthropic.Anthropic()

    system_prompt = """\
You are an autonomous ML research agent optimizing a protein language model.
You can ONLY modify train.py. The prepare.py file (tokenizer, data loading,
masking, evaluation) is immutable.

Rules:
1. Keep the `train()` function signature exactly: train(train_data, val_data, max_seconds, device)
2. It must return a dict with keys: val_loss, train_loss, steps, params
3. Import from autoprot.prepare as needed (the API is fixed)
4. You may change: architecture, hyperparameters, optimizer, schedule, etc.
5. Output ONLY the complete modified train.py file, no explanation outside the code
6. Start your response with the first line of code (no markdown fences)
7. Briefly explain your changes in a comment at the top of the file

Keep the model trainable in the time budget (5 min on a single GPU).
Focus on changes that reduce val_loss (cross-entropy on masked positions)."""

    user_prompt = f"""\
## Research Objectives (program.md)

{program}

## Recent Results

{results}

## Current train.py

```python
{current_train}
```

Based on the research objectives and previous results, propose a single focused
modification to train.py that you believe will reduce val_loss. Explain your
reasoning in a brief comment at the top of the file."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text

    # Strip markdown fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first line (```python) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    return response_text


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


def run_experiment(
    train_data: list[list[int]],
    val_data: list[list[int]],
    max_seconds: int,
    device: str,
) -> dict[str, float | int]:
    """Reload train module and run training."""
    # Force reimport to pick up modifications
    if "autoprot.train" in sys.modules:
        del sys.modules["autoprot.train"]
    train_module = importlib.import_module("autoprot.train")
    return train_module.train(train_data, val_data, max_seconds, device)


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="Directory containing .fasta/.fa files"),
    max_iterations: int = typer.Option(100, help="Number of experiment iterations"),
    train_seconds: int = typer.Option(300, help="Training time budget per iteration (seconds)"),
    device: str = typer.Option("cuda", help="Device to train on"),
    val_fraction: float = typer.Option(0.1, help="Fraction of data for validation"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
) -> None:
    """Run autonomous protein LM experiment loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    console.print("[bold]autoprot[/bold] — autonomous protein LM research", style="cyan")

    # Load data
    console.print(f"Loading data from {data_dir}...")
    train_data, val_data = create_datasets(
        data_dir,
        val_fraction=val_fraction,
        max_length=max_length,
    )
    console.print(f"  train: {len(train_data)} sequences, val: {len(val_data)} sequences")

    # Read program.md
    program_path = Path("program.md")
    if program_path.exists():
        program = program_path.read_text()
    else:
        console.print("[yellow]Warning: program.md not found, using empty objectives[/yellow]")
        program = ""

    best_val_loss = float("inf")

    for iteration in range(1, max_iterations + 1):
        console.rule(f"[bold]Iteration {iteration}/{max_iterations}[/bold]")

        # Read current state
        current_train = TRAIN_PY.read_text()
        results = read_last_results()

        # Call LLM to modify train.py
        try:
            console.print("Calling LLM for modifications...")
            new_train = call_llm(program, results, current_train)
            TRAIN_PY.write_text(new_train)
            console.print("  train.py updated")
        except Exception as e:
            console.print(f"[red]LLM call failed: {e}[/red]")
            log_result(iteration, "error", "error", "0", "0", f"LLM error: {e}")
            continue

        # Run experiment
        try:
            console.print(f"Training for {train_seconds}s...")
            result = run_experiment(train_data, val_data, train_seconds, device)
            val_loss = result["val_loss"]
            train_loss = result["train_loss"]
            steps = result["steps"]
            params = result["params"]

            console.print(
                f"  val_loss={val_loss:.4f}  train_loss={train_loss:.4f}"
                f"  steps={steps}  params={params:,}"
            )

        except Exception as e:
            console.print(f"[red]Training failed: {e}[/red]")
            console.print(traceback.format_exc())
            log_result(iteration, "error", "error", "0", "0", f"train error: {e}")
            git_revert()
            console.print("  reverted train.py")
            continue

        # Extract description from first comment in the file
        new_source = TRAIN_PY.read_text()
        description = ""
        for line in new_source.split("\n"):
            if line.startswith("# ") and "MUTABLE" not in line:
                description = line.lstrip("# ").strip()
                break
        if not description:
            description = f"iteration {iteration}"

        # Log result
        log_result(iteration, f"{val_loss:.4f}", f"{train_loss:.4f}", steps, params, description)

        # Keep or revert
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            git_commit(f"exp {iteration}: val_loss={val_loss:.4f} — {description}")
            console.print(f"  [green]improvement! committed (best={best_val_loss:.4f})[/green]")
        else:
            git_revert()
            console.print(f"  [yellow]no improvement (best={best_val_loss:.4f}), reverted[/yellow]")

        # Cleanup
        import gc

        gc.collect()
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()

    console.print(f"\n[bold green]Done! Best val_loss: {best_val_loss:.4f}[/bold green]")


if __name__ == "__main__":
    app()
