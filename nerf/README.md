# Video → Nerf (Nerfstudio) Pipeline

Goal: mirror the 3dgs layout so a single video becomes a Nerfstudio NeRF (default `nerfacto`) with clean, reproducible structure and one-command orchestration.

## Layout

```
nerf/
  configs/               # YAML configs
  data/
    raw/                 # drop raw videos here
    processed/           # ns-process-data outputs (frames + COLMAP)
  exports/               # rendered outputs: videos/figures/cameras/meshes
  outputs/               # Nerfstudio training runs
  scripts/               # helper scripts
  src/ns_nerf/           # Python package (pipeline orchestration)
```

## Environment setup (conda env: nerf)

```bash
cd nerf
bash install.sh
```

What the script does:
1) Create/activate conda env `nerf` (Python 3.12).
2) Install PyTorch with CUDA 12.1 wheels (edit `TORCH_INDEX` in `install.sh` if you need cu124/cu118, etc.).
3) Install Nerfstudio + helpers so `ns-process-data`, `ns-train`, `ns-render`, `ns-export` are available.
4) Install COLMAP + FFmpeg from conda-forge.
5) Install this project editable: `pip install -e .`.

## Run the full pipeline

Prepare a video at `data/raw/<scene>/input.mp4`, edit `configs/pipeline.yaml` for `scene_name` and `video_path`, then:

```bash
cd nerf
conda activate nerf
python -m ns_nerf.cli --config configs/pipeline.yaml
# or
scripts/run_full_pipeline.sh
```

Pipeline steps:
1. Frame extraction + COLMAP: `ns-process-data video --data <mp4> --output-dir data/processed/<scene>`.
2. Train NeRF: `ns-train nerfacto --data data/processed/<scene> --output-dir outputs --experiment-name <scene>`.
3. Render dataset views: `ns-render dataset --load-config <latest config>` for train/test reconstructions.
4. Render camera-path video (optional): `ns-render camera-path` using `data/processed/<scene>/camera_paths/orbit.json` to `exports/videos/<scene>_orbit.mp4`.
5. Export cameras/mesh (optional): `ns-export cameras|mesh`.

## Config quick reference (`configs/pipeline.yaml`)

- `scene_name`: used for output naming.
- `video_path`: raw video path under `data/raw/...`.
- `processing.num_frames_target`: downsample frames; comment out to use all frames.
- `training.method`: Nerfstudio method (default `nerfacto` for quality/speed balance).
- `render.make_dataset_renders` / `make_orbit_video`: toggle dataset/trajectory renders.
- `exports.cameras` / `mesh`: toggle camera/mesh exports.

## Single-step examples

```bash
# Only process data
python -m ns_nerf.pipeline.video_to_dataset \
  data/raw/demo/input.mp4 data/processed/demo --num-frames 300

# Only train
python -m ns_nerf.pipeline.train_nerf \
  data/processed/demo --scene demo --method nerfacto

# Find the latest config
python - <<'PY'
from ns_nerf.utils import find_latest_config
print(find_latest_config("demo", "nerfacto"))
PY

# Render an interpolation video (no camera path needed)
CONFIG=$(python - <<'PY'
from ns_nerf.utils import find_latest_config
print(find_latest_config("demo","nerfacto"))
PY)
ns-render interpolate --load-config "$CONFIG" --output-path exports/videos/demo_interpolate.mp4
```

## Notes for RTX 4090

- Use `--max-num-iterations` or `--target-num-steps` in `training.extra_args` to bound training time.
- If VRAM is tight, lower model size or dataparser scale (e.g., `--pipeline.model.hidden-dim 64 --pipeline.dataparser.downscale-factor 2`).
- Rendering time scales with resolution; add `--downscale-factor` in `render.extra_args` for quicker previews.

I have not run `ns-train` end-to-end here (time/compute heavy), but the layout matches 3dgs and is ready to execute. If any dependency or path errors appear, adjust `install.sh` or the YAML accordingly.
