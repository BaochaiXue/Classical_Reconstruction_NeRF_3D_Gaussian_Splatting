"""Config loader for the Nerf pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_pipeline_config(path: Path) -> Dict[str, Any]:
  """Load a YAML pipeline config."""
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f)
