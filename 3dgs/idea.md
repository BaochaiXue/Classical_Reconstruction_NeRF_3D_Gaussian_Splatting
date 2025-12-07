Nice project, Xinjie. Let’s build the “statue-from-video → 3DGS” pipeline like a real system, not a pile of loose shell commands.

I’ll assume:

* **Ubuntu + conda**
* **One GPU with CUDA 12.8 ish**
* Your video is `data/raw/statue/1.mp4`

I’ll give you:

1. Env setup (Ubuntu + conda)
2. Repo structure
3. Step‑by‑step code modules matching your 1–7
4. Exact Nerfstudio commands (wrapped in Python) for Splatfacto / 3DGS
5. Where to grab intermediate artifacts for the paper (figures, point clouds, etc.)
6. How to render a 360° statue fly‑around

---

## 0. Environment on Ubuntu with conda

### 0.1. Create a conda env & install Nerfstudio

From the Nerfstudio README, they recommend conda, Python ≥3.8, then install torch + nerfstudio.([GitHub][1])

```bash
# Create env (PyTorch docs state 3.10–3.14 is supported)
conda create -n 3dgs_statue python=3.12 -y
conda activate 3dgs_statue

# (Recommended) install PyTorch with CUDA per official instructions:
# See: https://pytorch.org/get-started/locally/
# Example for CUDA 12.8:
pip install torch==2.7.0+cu128 torchvision==0.20.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Tiny-cuda-nn etc., as Nerfstudio suggests:
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install nerfstudio itself:
pip install nerfstudio
# or the dev version:
# git clone https://github.com/nerfstudio-project/nerfstudio.git
# cd nerfstudio && pip install -e .

# gsplat runtime dependency for viewer logging
pip install "rich>=13"
```

### 0.2. Install COLMAP + FFmpeg (via conda + apt)

Nerfstudio’s data pipeline uses **COLMAP + FFmpeg** when you run `ns-process-data {video,images}`.([Nerf Studio][2])

```bash
# COLMAP via conda-forge (Nerfstudio docs recommend this on Linux)
conda install -c conda-forge colmap

# FFmpeg via apt is fine
sudo apt update
sudo apt install -y ffmpeg

# Sanity‑check:
colmap -h
ffmpeg -h
```

That’s enough tooling for: video → images + camera poses.

---

## 1. Project structure

Let’s make a small, extensible repo:

```text
3dgs_statue/
  README.md
  pyproject.toml        # or requirements.txt
  configs/
    pipeline.yaml       # high-level configs
  data/
    raw/
      statue/
        1.mp4
    processed/
      statue_scene/     # ns-process-data output
  outputs/              # nerfstudio runs live here (ns-train default)
  exports/
    splats/             # Gaussian splat .ply
    meshes/             # (from SuGaR or mesh export)
    videos/             # 360° flythrough mp4
    figures/            # stills for paper
    cameras/            # camera JSONs
  src/gs_statue/
    __init__.py
    config.py
    cli.py
    pipeline/
      video_to_dataset.py
      train_splat.py
      render_views.py
      export_assets.py
      diagnostics.py
  scripts/
    run_full_pipeline.sh
```

We’ll treat **Nerfstudio commands as the “engine”**, and your Python just orchestrates them.

---

## 2. Step 1–2: video → images + camera poses

Your steps (1) & (2):

> 1. Transform the video frames into images and save it into data
> 2. Use the proper way to learn the camera info and other info from just the images

Nerfstudio has exactly this as `ns-process-data video`. It:

* uses **FFmpeg** to extract frames
* runs **COLMAP** to estimate camera intrinsics/extrinsics
* emits a Nerfstudio dataset (images + `transforms.json` etc.)([Nerf Studio][2])

### 2.1. Direct CLI

Core command:

```bash
ns-process-data video \
  --data data/raw/statue/1.mp4 \
  --output-dir data/processed/statue_scene
```

You can control sampling density with `--num-frames-target`, as seen in examples:([Gerry's World][3])

```bash
ns-process-data video \
  --data data/raw/statue/1.mp4 \
  --output-dir data/processed/statue_scene \
  --num-frames-target 200
```

This gives you:

* `data/processed/statue_scene/images/` – extracted frames (possibly multi-scale)
* `data/processed/statue_scene/transforms.json` – all poses
* `data/processed/statue_scene/colmap/` – COLMAP project & sparse model

### 2.2. Python wrapper: `video_to_dataset.py`

`src/gs_statue/pipeline/video_to_dataset.py`:

```python
import subprocess
from pathlib import Path

def process_video_to_dataset(
    video_path: Path,
    output_dir: Path,
    num_frames_target: int | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ns-process-data", "video",
        "--data", str(video_path),
        "--output-dir", str(output_dir),
    ]
    if num_frames_target is not None:
        cmd += ["--num-frames-target", str(num_frames_target)]
    if extra_args:
        cmd += extra_args

    print("[process_video_to_dataset] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return output_dir
```

That’s your **steps 1–2** done in a reproducible way.

---

## 3. Step 3: Train 3D Gaussian splats (Splatfacto)

Nerfstudio’s 3DGS method is **Splatfacto**, built on their `gsplat` rasterizer.([Nerf Studio][4])

Docs literally say:

> “To run splatfacto, run `ns-train splatfacto --data <data>`.”([Nerf Studio][4])

### 3.1. Baseline training command

```bash
ns-train splatfacto \
  --data data/processed/statue_scene
```

For real work, I’d tweak:

* Use the viewer + tensorboard for monitoring: `--vis viewer+tensorboard`([GitHub][1])
* Cache images to disk if GPU mem is tight / dataset is big (`full_images_datamanager` caching):([Nerf Studio][4])
* Slightly stricter culling for quality: docs suggest decreasing `cull_alpha_thresh` and stopping culling after densification.([Nerf Studio][4])

Example “statue-quality” run:

```bash
ns-train splatfacto \
  --data data/processed/statue_scene \
  --vis viewer+tensorboard \
  --pipeline.datamanager.cache-images disk \
  --pipeline.model.cull_alpha_thresh=0.005 \
  --pipeline.model.continue_cull_post_densification=False
```

### 3.2. Python wrapper: `train_splat.py`

```python
import subprocess
from pathlib import Path

def train_splat(
    data_dir: Path,
    method: str = "splatfacto",
    vis: str = "viewer+tensorboard",
    extra_args: list[str] | None = None,
):
    cmd = [
        "ns-train", method,
        "--data", str(data_dir),
        "--vis", vis,
    ]
    if extra_args:
        cmd += extra_args

    print("[train_splat] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

Nerfstudio will create:

```text
outputs/
  statue_scene/
    splatfacto/
      2025-xx-xx_xxxxxx/
        config.yml
        nerfstudio_models/
        ...
```

We’ll need `config.yml` later.

Add a utility in `src/gs_statue/utils.py`:

```python
from pathlib import Path

def find_latest_config(scene_name: str, method: str) -> Path:
    base = Path("outputs") / scene_name / method
    runs = sorted(base.glob("*/config.yml"))
    if not runs:
        raise FileNotFoundError(f"No runs under {base}")
    return runs[-1]
```

---

## 4. Step 4: Inspect reconstruction in the viewer

Two ways:

1. You already had the viewer during training (`--vis viewer+...`).
2. Or, after training, run **ns-viewer** with the saved config. The README explicitly supports this.([GitHub][1])

```bash
# After training:
CONFIG=$(python -c "from gs_statue.utils import find_latest_config; \
print(find_latest_config('statue_scene','splatfacto'))")

ns-viewer --load-config "$CONFIG"
```

The web viewer lets you:

* Fly around with WASD + mouse
* Toggle output types
* Use **RENDER** and **EXPORT** tabs to generate ready‑made `ns-render` / `ns-export` commands for videos and geometry([GitHub][1])

That’s your step (4).

---

## 5. Step 5: Intermediate results & visualizations (for paper/report)

You want:

> 5. Save important intermediate results and visualize them

I’d log three classes of artifacts:

1. Re‑rendered training/test views
2. Camera poses visualizations
3. Simple point cloud or mesh baseline

### 5.1. Re‑render dataset views

Nerfstudio has `ns-render dataset` as a subcommand (under `ns-render {camera-path,interpolate,spiral,dataset}`).([Nerf Studio][5])

Example:

```bash
ns-render dataset \
  --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
  --output-path exports/figures/statue_dataset \
  --split train+test \
  --rendered-output-names rgb
```

This gives you synthetic renders at **exactly the training/test camera poses**. Perfect for side‑by‑side comparisons.

Wrapper in `render_views.py`:

```python
def render_dataset_views(config_path, out_dir, split="train+test"):
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render", "dataset",
        "--load-config", str(config_path),
        "--output-path", str(out_dir),
        "--split", split,
        "--rendered-output-names", "rgb",
    ]
    subprocess.run(cmd, check=True)
```

### 5.2. Export cameras and plot trajectories

The docs show `ns-export` supports camera export; `ns-export cameras --help` gives details.([GitHub][1])

```bash
ns-export cameras \
  --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
  --output-dir exports/cameras/statue
```

Then `diagnostics.py` can:

* Load the camera JSONs
* Extract camera centers
* Plot them with matplotlib (3D scatter + frustum outlines)
* Save `exports/figures/statue_camera_poses.png`

Pseudo‑code:

```python
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def plot_camera_poses(cameras_dir: Path, out_png: Path):
    all_centers = []
    for f in cameras_dir.glob("*.json"):
        data = json.loads(f.read_text())
        # Depends on format; usually you get camera_to_world / intrinsics
        for cam in data["cameras"]:
            c2w = np.array(cam["camera_to_world"])  # 4x4
            center = c2w[:3, 3]
            all_centers.append(center)
    all_centers = np.stack(all_centers)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_centers[:, 0], all_centers[:, 1], all_centers[:, 2])
    ax.set_title("Statue camera trajectory")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
```

That figure is catnip for reviewers.

### 5.3. Optional: point cloud / mesh baseline

For comparison to “traditional” reconstructions, you can also export:

* **Point cloud:** `ns-export pointcloud`
* **Mesh:** `ns-export mesh` (for methods that support it; Splatfacto currently only guarantees splats, but other Nerfstudio methods do mesh export).([CSDN Blog][6])

Point cloud example for your statue:

```bash
ns-export pointcloud \
  --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
  --output-dir exports/pointclouds/statue \
  --num-points 1000000
```

You can visualize this in Meshlab / CloudCompare and include screenshots as baseline figures.

---

## 6. Step 6: 360° video of the statue

Goal:

> 6. Save a 360-degree video to show our reconstruction result

Nerfstudio’s `ns-render` has subcommands: `{camera-path, interpolate, spiral, dataset}`.([Nerf Studio][5])

Two good options:

### 6.1. Simple: `ns-render interpolate`

From community docs / examples: `ns-render interpolate` renders a video along a path that interpolates training cameras.([Mashaan14][7])

```bash
ns-render interpolate \
  --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
  --output-path exports/videos/statue_interpolate.mp4
```

This is one command and done. It tends to “orbit-ish” around the scene by following/interpolating your capture path.

### 6.2. Custom camera path (clean 360° orbit)

Official docs describe the workflow: use the viewer’s **RENDER tab** to define a camera path, then it shows you the `ns-render` command to run.([GitHub][1])

Workflow:

1. Run viewer:

   ```bash
   ns-viewer --load-config outputs/statue_scene/splatfacto/<date>/config.yml
   ```

2. In the web viewer:

   * Go to **RENDER** tab.
   * Set a keyframe at the front of the statue → “ADD CAMERA”.
   * Move around the statue in an orbit, add ~12–24 keyframes around it so you get a smooth 360.
   * Adjust duration, FPS, resolution.

3. Press **RENDER**. The viewer pops up the exact `ns-render` command, something like:

   ```bash
   ns-render camera-path \
     --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
     --camera-path-filename data/processed/statue_scene/camera_paths/orbit.json \
     --output-path exports/videos/statue_orbit.mp4
   ```

   `camera_paths/orbit.json` is a reusable artifact—check it into git so your paper references a reproducible camera path.

Python wrapper in `render_views.py`:

```python
def render_orbit_video(config_path, camera_path, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-render", "camera-path",
        "--load-config", str(config_path),
        "--camera-path-filename", str(camera_path),
        "--output-path", str(out_path),
    ]
    subprocess.run(cmd, check=True)
```

That’s your **360° statue promo video**.

---

## 7. From splats to mesh (SuGaR, optional but aligned with your bullets)

You explicitly mentioned SuGaR:

> Surface-aligned splats (SuGaR) produce editable meshes within minutes

Nerfstudio’s Splatfacto docs show how to export a `.ply` Gaussian splat:([Nerf Studio][4])

```bash
ns-export gaussian-splat \
  --load-config outputs/statue_scene/splatfacto/<date>/config.yml \
  --output-dir exports/splats/statue
```

This creates e.g. `exports/splats/statue/point_cloud.ply` which is compatible with various viewers & tools.([Nerf Studio][4])

Then:

1. Clone SuGaR (surface‑aligned Gaussians → mesh).([Nerf Studio][8])

2. Run their script on your `.ply`:

   ```bash
   python sugar_main.py \
     --input-3dgs exports/splats/statue/point_cloud.ply \
     --output-dir exports/meshes/statue \
     --mesh-resolution medium
   ```

3. Use the resulting mesh for:

   * close‑up stills in the paper,
   * comparison against NeRF/SDF meshes,
   * potentially editing/retargeting.

You can later wrap this in `export_assets.py` as a “third‑party” stage.

---

## 8. Step 7: One unified CLI & configuration

Tie everything together in `src/gs_statue/cli.py`.

### 8.1. High‑level YAML config

`configs/pipeline.yaml` (sketch):

```yaml
scene_name: statue_scene
video_path: data/raw/statue/1.mp4

processing:
  num_frames_target: 250
  extra_args:
    - --matching-method
    - sequential

training:
  method: splatfacto
  vis: viewer+tensorboard
  extra_args:
    - --pipeline.datamanager.cache-images
    - disk
    - --pipeline.model.cull_alpha_thresh
    - 0.005
    - --pipeline.model.continue_cull_post_densification=False

render:
  make_dataset_renders: true
  make_orbit_video: true
  camera_path_file: data/processed/statue_scene/camera_paths/orbit.json
```

You can parse it in `config.py` with `yaml.safe_load`.

### 8.2. Orchestration CLI

`src/gs_statue/cli.py`:

```python
import argparse
from pathlib import Path

from .config import load_pipeline_config
from .utils import find_latest_config
from .pipeline.video_to_dataset import process_video_to_dataset
from .pipeline.train_splat import train_splat
from .pipeline.render_views import render_dataset_views, render_orbit_video
from .pipeline.export_assets import export_gaussian_splat, export_cameras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    scene = cfg["scene_name"]

    # 1–2: video -> dataset
    data_dir = Path("data/processed") / scene
    process_video_to_dataset(
        Path(cfg["video_path"]),
        data_dir,
        num_frames_target=cfg["processing"]["num_frames_target"],
    )

    # 3: train splatfacto
    train_splat(
        data_dir,
        method=cfg["training"]["method"],
        vis=cfg["training"]["vis"],
        extra_args=cfg["training"]["extra_args"],
    )

    # Resolve config.yml of latest run
    config_path = find_latest_config(scene, cfg["training"]["method"])

    # 5: intermediate figures
    render_dataset_views(config_path, Path("exports/figures") / scene)
    export_cameras(config_path, Path("exports/cameras") / scene)

    # 6: export splats
    export_gaussian_splat(config_path, Path("exports/splats") / scene)

    # 6: 360° video (requires camera path json)
    cam_path = Path(cfg["render"]["camera_path_file"])
    if cfg["render"]["make_orbit_video"] and cam_path.exists():
        render_orbit_video(
            config_path,
            cam_path,
            Path("exports/videos") / f"{scene}_orbit.mp4",
        )

if __name__ == "__main__":
    main()
```

And a tiny `scripts/run_full_pipeline.sh`:

```bash
#!/usr/bin/env bash
set -e
conda activate 3dgs_statue
python -m gs_statue.cli --config configs/pipeline.yaml
```

---

## 9. Mapping back to your numbered goals

1. **Transform video frames into images**
   → `ns-process-data video` in `video_to_dataset.py` produces `data/processed/statue_scene/images`.([Nerf Studio][2])

2. **Learn camera info from just images**
   → Same command; COLMAP inside `ns-process-data` estimates camera poses, stored in `transforms.json` & COLMAP project.([Nerf Studio][2])

3. **Train the Gaussian with enough time**
   → `ns-train splatfacto --data data/processed/statue_scene` wrapped in `train_splat.py`.([Nerf Studio][4])

4. **See final reconstruction from viewer**
   → `ns-viewer --load-config outputs/statue_scene/splatfacto/<date>/config.yml`.([GitHub][1])

5. **Save intermediate results & visualizations**

   * `ns-render dataset` → image reconstructions (train/test).([Nerf Studio][5])
   * `ns-export cameras` → camera poses for 3D plots.([GitHub][1])
   * Optional `ns-export pointcloud` / `mesh`.([CSDN Blog][6])

6. **360° video**

   * Quick path: `ns-render interpolate --load-config ... --output-path exports/videos/statue_interpolate.mp4`.([Nerf Studio][5])
   * Clean orbit: viewer → camera path JSON → `ns-render camera-path ...`.([GitHub][1])

7. **Good engineering structure, extensible**

   * Modular pipeline scripts under `src/gs_statue/pipeline/`
   * YAML configs for “methods” (swap `splatfacto` ↔ `splatfacto-big` ↔ `nerf2gs2nerf` later).([Nerf Studio][4])
   * All Nerfstudio/GS specifics confined behind small wrappers so you can plug in SLAM‑style GS or SDF methods in future.

---

That’s the full coding plan: one conda env, a small orchestrator package, and a clean path from `1.mp4` of your statue to a 3D Gaussian Splat model, pretty figures, a mesh, and a 360° fly‑around video. Once this is in place, the fun part is ablations: COLMAP settings, densification/culling strategies, Splatfacto‑W vs Splatfacto, and how SuGaR’s mesh quality behaves on your statue capture.

[1]: https://github.com/nerfstudio-project/nerfstudio "GitHub - nerfstudio-project/nerfstudio: A collaboration friendly studio for NeRFs"
[2]: https://docs.nerf.studio/quickstart/custom_dataset.html "Using custom data - nerfstudio"
[3]: https://gerry-chen.com/blog/2023-05-02_nerf.html?utm_source=chatgpt.com "Playing with NeRF | Blog | Gerry's World"
[4]: https://docs.nerf.studio/nerfology/methods/splat.html "Splatfacto - nerfstudio"
[5]: https://docs.nerf.studio/reference/cli/ns_render.html?utm_source=chatgpt.com "ns-render"
[6]: https://blog.csdn.net/qq_41889538/article/details/147615528?utm_source=chatgpt.com "NeRFstudio export formats (pointcloud and mesh)"
[7]: https://mashaan14.github.io/YouTube-channel/nerf/2025_01_14_nerfstudio_lightning_ai?utm_source=chatgpt.com "Running nerfacto algorithm on lightning ai GPUs - GitHub Pages"
[8]: https://docs.nerf.studio/nerfology/methods/splat.html?utm_source=chatgpt.com "Splatfacto"
