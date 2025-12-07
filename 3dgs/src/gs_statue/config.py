"""Configuration helpers for the statue pipeline."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "scene_name": "statue_scene",
    "video_path": "data/raw/statue/1.mp4",
    "processing": {
        "num_frames_target": 250,
        "extra_args": [],
    },
    "training": {
        "method": "splatfacto",
        "vis": "viewer+tensorboard",
        "extra_args": [
            "--pipeline.datamanager.cache-images",
            "disk",
            "--pipeline.model.cull_alpha_thresh",
            "0.005",
            "--pipeline.model.continue_cull_post_densification=False",
        ],
    },
    "render": {
        "make_dataset_renders": True,
        "dataset_render_split": "train+test",
        "make_orbit_video": True,
        "camera_path_file": "data/processed/statue_scene/camera_paths/orbit.json",
        "make_interpolate_video": False,
        "interpolate_video_name": "interpolate.mp4",
    },
    "exports": {
        "figures": True,
        "cameras": True,
        "splats": True,
        "videos": True,
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    """Read YAML config file and merge with defaults."""
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    if data is None:
        data = {}
    return _deep_update(DEFAULT_CONFIG, data)
