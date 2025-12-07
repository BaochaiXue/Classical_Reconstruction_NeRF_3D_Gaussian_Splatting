# Video → Classical 3D Reconstruction (COLMAP SfM + MVS)

Goal: mirror the 3dgs layout for a reproducible classical pipeline—extract frames from a video, run COLMAP SfM/MVS, and export dense point clouds/meshes.

## Layout

```
classical/
  configs/               # YAML configs
  data/
    raw/                 # drop raw videos here
    processed/           # extracted frames
  exports/
    figures/             # visual diagnostics (future use)
    meshes/              # mesh exports
    pointclouds/         # fused point clouds
  outputs/               # COLMAP workspaces (db, sparse, dense)
  scripts/               # helper scripts
  src/classical_recon/   # Python package (pipeline orchestration)
```

## Environment setup (conda env: classical)

```bash
cd classical
bash install.sh
```

What the script does:
1) Create/activate conda env `classical` (Python 3.12).
2) Install COLMAP + FFmpeg from conda-forge.
3) Install PyYAML/Rich.
4) Install this project editable: `pip install -e .`.

## Run the full pipeline

Place your video at `data/raw/<scene>/input.mp4`, edit `configs/pipeline.yaml` (`scene_name`, `video_path`, frame rate, etc.), then:

```bash
cd classical
conda activate classical
python -m classical_recon.cli --config configs/pipeline.yaml
# or
scripts/run_full_pipeline.sh
```

Pipeline steps:
1. Extract frames with FFmpeg to `data/processed/<scene>/images`.
2. COLMAP feature extraction, sequential matching, mapping.
3. Image undistortion, PatchMatch stereo, and fusion to `outputs/<scene>/colmap/dense/fused.ply`.
4. Copy fused point cloud to `exports/pointclouds/<scene>/fused.ply`. Mesh export is stubbed (hook in OpenMVS/Poisson if needed).

## Config quick reference (`configs/pipeline.yaml`)

- `scene_name`: output naming.
- `video_path`: raw video path.
- `processing.frame_rate`: target FPS for extraction (None uses source FPS).
- `processing.max_frames`: cap number of extracted frames (None keeps all).
- `colmap.matcher`: `sequential` (default) or `exhaustive`.
- `colmap.num_threads`: threads for COLMAP.
- `colmap.make_mesh`: run Poisson mesher after fusion.
- `colmap.extra_args`: list forwarded to each COLMAP call (advanced).

## Single-step examples

```bash
# Only extract frames
python -m classical_recon.pipeline.video_to_frames \
  data/raw/demo/input.mp4 data/processed/demo --fps 8 --max-frames 400

# Only run COLMAP (frames already extracted)
python -m classical_recon.pipeline.run_colmap \
  data/processed/demo/images outputs/demo/colmap --matcher sequential
```

## Notes

- This repo does not bundle OpenMVS/mesh post-processing; add your preferred meshing step in `pipeline/export_assets.py`.
- I did not execute a full reconstruction here (no sample video in repo); the code paths are ready to run with your data. If you see dependency or path errors, adjust `install.sh` or the YAML config accordingly. 
