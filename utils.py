"""Utility functions for CRL-Atari experiments."""

import torch
import numpy as np
import random
import os
import json
import yaml
from datetime import datetime
from typing import Any, Dict, Tuple


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve device from config preference."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class ResultsManager:
    """
    Manages experiment output files.

    Directory layout:
        output_dir/
        ├── csv/
        ├── json/
        ├── checkpoints/
        └── plots/
            ├── png/
            └── svg/
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, "csv")
        self.json_dir = os.path.join(output_dir, "json")
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.plots_png_dir = os.path.join(output_dir, "plots", "png")
        self.plots_svg_dir = os.path.join(output_dir, "plots", "svg")

        for d in [self.csv_dir, self.json_dir, self.checkpoints_dir,
                  self.plots_png_dir, self.plots_svg_dir]:
            os.makedirs(d, exist_ok=True)

    def save_json(self, data: Dict, filename: str) -> str:
        if not filename.endswith(".json"):
            filename += ".json"
        path = os.path.join(self.json_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_checkpoint(self, state_dict: Dict, name: str) -> str:
        path = os.path.join(self.checkpoints_dir, f"{name}.pt")
        torch.save(state_dict, path)
        return path

    def load_checkpoint(self, name: str) -> Dict:
        path = os.path.join(self.checkpoints_dir, f"{name}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

    def get_plot_paths(self, name: str) -> Tuple[str, str]:
        """Return (png_path, svg_path) for a plot name."""
        return (
            os.path.join(self.plots_png_dir, f"{name}.png"),
            os.path.join(self.plots_svg_dir, f"{name}.svg"),
        )

    def get_csv_path(self, name: str) -> str:
        if not name.endswith(".csv"):
            name += ".csv"
        return os.path.join(self.csv_dir, name)
