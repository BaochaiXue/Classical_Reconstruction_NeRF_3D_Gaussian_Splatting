"""Configuration loader for the classical pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    """Load a YAML pipeline config."""
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
