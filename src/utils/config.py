"""
Utility functions for configuration loading and management.
"""

import yaml
import os
import copy
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge in.

    Returns:
        Merged configuration dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_effective_config(
    base_config_path: str = "configs/base.yaml",
    override_config_path: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Load base config, optionally merge overrides, and apply debug mode.

    Debug mode replaces training hyperparameters with reduced values
    for fast local testing.

    Args:
        base_config_path: Path to the base YAML config.
        override_config_path: Optional path to override config.
        debug: Whether to enable debug mode (reduced training).

    Returns:
        Effective configuration dictionary.
    """
    config = load_config(base_config_path)

    if override_config_path and os.path.exists(override_config_path):
        override = load_config(override_config_path)
        config = deep_merge(config, override)

    # Force debug mode if requested
    if debug:
        config["debug"]["enabled"] = True

    # Apply debug overrides
    if config.get("debug", {}).get("enabled", False):
        debug_cfg = config["debug"]
        config["training"]["total_timesteps"] = debug_cfg.get(
            "total_timesteps", 5000
        )
        config["training"]["buffer_size"] = debug_cfg.get("buffer_size", 1000)
        config["training"]["min_buffer_size"] = debug_cfg.get("min_buffer_size", 500)
        config["training"]["eval_episodes"] = debug_cfg.get("eval_episodes", 2)
        config["training"]["eval_freq"] = debug_cfg.get("eval_freq", 1000)
        config["training"]["save_freq"] = debug_cfg.get("save_freq", 2500)
        config["ewc"]["fisher_samples"] = debug_cfg.get("fisher_samples", 100)
        config["htcl"]["fisher_samples"] = debug_cfg.get("fisher_samples", 100)
        config["distillation"]["distill_epochs"] = debug_cfg.get("distill_epochs", 5)
        config["distillation"]["buffer_size_per_task"] = debug_cfg.get(
            "distill_buffer_size", 500
        )
        # Also override evaluation episodes for compare.py
        config["evaluation"]["episodes"] = debug_cfg.get("eval_episodes", 2)

    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
