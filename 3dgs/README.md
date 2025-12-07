# Statue → 3D Gaussian Splat Pipeline

This repository turns a single statue video into a Nerfstudio 3D Gaussian Splat (3DGS) reconstruction with clean engineering structure. It wraps Nerfstudio CLIs behind Python modules so the whole pipeline can run from one command while still exposing every intermediate artifact for papers or demos.

## Requirements

- Ubuntu (tested with 22.04) and Conda.
- CUDA 12.8 compatible GPU with ≥12 GB VRAM.
- Video recording at `data/raw/statue/1.mp4` (drop your own footage there).
- COLMAP and FFmpeg (installed via conda-forge/apt).

## Environment setup

```bash
# 1. Create the environment (PyTorch supports Python 3.10–3.14 per https://pytorch.org/get-started/locally/)
conda create -n 3dgs_statue python=3.12 -y
conda activate 3dgs_statue

# 2. Install PyTorch that matches your CUDA toolchain (example: CUDA 12.8)
pip install torch==2.7.0+cu128 torchvision==0.20.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

# 3. Extra CUDA dependencies for Nerfstudio
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 4. Nerfstudio
pip install nerfstudio
# or for local development:
# git clone https://github.com/nerfstudio-project/nerfstudio.git
# pip install -e nerfstudio

# 4.5. Satisfy gsplat's optional runtimes
pip install "rich>=13"

# 5. CLI tools
conda install -c conda-forge colmap
sudo apt update && sudo apt install -y ffmpeg

# 6. Project package (runs orchestration code)
pip install -e .
```

## Repository layout

```
configs/                 # YAML configs for scenes/pipelines
data/
  raw/statue/1.mp4       # your capture
  processed/             # ns-process-data output
exports/                 # figures, videos, splats, cameras, meshes, etc.
outputs/                 # Nerfstudio training runs (ns-train default)
scripts/run_full_pipeline.sh
src/gs_statue/           # Python package wrapping each stage
```

`idea.md` documents the system-level reasoning behind this layout and maps directly onto the implemented modules.

## Running the full pipeline (steps 1–7)

```bash
conda activate 3dgs_statue
python -m gs_statue.cli --config configs/pipeline.yaml
# or
scripts/run_full_pipeline.sh
```

What happens:

1. **Video → dataset** (`ns-process-data video`) extracts frames and runs COLMAP to estimate poses.
2. **Nerfstudio training** (`ns-train splatfacto`) optimizes a 3D Gaussian Splat model with viewer+TensorBoard monitoring and tuned culling.
3. **Viewer** can be launched at any time via `ns-viewer --load-config <latest config>`.
4. **Intermediate renders** (`ns-render dataset`) provide train/test re-renders for figures.
5. **Camera exports** (`ns-export cameras`) feed diagnostics plots.
6. **Gaussian splat export** (`ns-export gaussian-splat`) produces `.ply` files for SuGaR/CloudCompare.
7. **360° video** (`ns-render camera-path`) renders the saved orbit camera path.

Every sub-step is implemented in `src/gs_statue/pipeline/*` so you can mix/match or call them from notebooks.

## Configuring scenes

Edit `configs/pipeline.yaml`:

- `scene_name`: used for dataset folder names and export prefixes.
- `video_path`: source footage.
- `processing`: number of frames to keep and optional `ns-process-data` overrides. Defaults use `--matching-method sequential` and assume a GPU-capable COLMAP build; no CPU fallback is bundled.
- `training`: Nerfstudio method, viewer mode, extra command-line args.
- `render`: toggle dataset renders and orbit video, camera-path JSON location.

Add more YAML configs if you have multiple captures—pass `--config path/to/other.yaml` to the CLI.

## Useful individual commands

```bash
# Inspect newest run in the viewer
python - <<'PY'
from gs_statue.utils import find_latest_config
print(find_latest_config("statue_scene", "splatfacto"))
PY
ns-viewer --load-config <printed config>

# Export an interpolated flythrough without a camera path
CONFIG=$(python -c "from gs_statue.utils import find_latest_config; \
print(find_latest_config('statue_scene','splatfacto'))")
ns-render interpolate --load-config "$CONFIG" --output-path exports/videos/statue_interpolate.mp4

# Plot camera trajectory
python -m gs_statue.pipeline.diagnostics exports/cameras/statue_scene exports/figures/statue_scene/camera_path.png
```

## Orbit video workflow

1. Launch the viewer with the latest config.
2. Use the **RENDER** tab to create an orbit camera path (12–24 evenly spaced keyframes recommended).
3. Save it to `data/processed/<scene>/camera_paths/orbit.json`.
4. Re-run the CLI (or call `render_orbit_video`) to generate `exports/videos/<scene>_orbit.mp4`.

## Extending the pipeline

- Choose different Nerfstudio methods (e.g., `splatfacto-big`, `nerfacto`) by editing the YAML.
- Add `ns-export mesh` or SuGaR conversion steps in `export_assets.py`.
- Enable/disable diagnostics, adjust COLMAP sampling density, or add evaluation hooks with minimal code changes thanks to the modular package layout.

With this structure you can iterate on capture strategies, training settings, and rendering outputs while keeping everything reproducible from a single command.
