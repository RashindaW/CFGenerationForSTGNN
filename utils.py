"""Shared utility functions for training and main modules."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _format_training_command() -> str:
    """Format the current training command for logging."""
    python_exec = sys.executable or "python"
    try:
        arg_string = shlex.join(sys.argv)
    except AttributeError:
        arg_string = " ".join(shlex.quote(arg) for arg in sys.argv)
    return f"{python_exec} {arg_string}".strip()


def write_training_command_file(directory: Path, filename: str = "trainingCommand.txt") -> Path:
    """Write the training command to a file for reproducibility."""
    directory.mkdir(parents=True, exist_ok=True)
    command_path = directory / filename
    command_path.write_text(_format_training_command() + "\n")
    return command_path


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a PyTorch checkpoint with backwards compatibility."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
