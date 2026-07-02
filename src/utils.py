"""
Shared utilities: logging setup, path helpers, config loading.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure root logger with console + optional file handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(LOGS_DIR / log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Defaults to configs/default_config.yaml."""
    path = Path(config_path) if config_path else CONFIGS_DIR / "default_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_results(results: dict[str, Any], name: str) -> Path:
    """Persist evaluation results as JSON to data/forecasts/ETT/."""
    import json

    out_dir = DATA_DIR / "forecasts" / "ETT"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return out_path
