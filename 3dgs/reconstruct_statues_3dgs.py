#!/usr/bin/env python3
"""
3D Gaussian Splatting pipeline from two RGB-only videos.

Stages:
1) Data loading / preprocessing: extract frames, resize/normalize, optional masks.
2) Camera pose estimation with COLMAP (intrinsics/extrinsics).
3) Gaussian initialization from COLMAP sparse points (positions+RGB).
4) Gaussian optimization with gsplat (photometric loss + regularization).
5) Novel-view rendering (spherical camera path).
6) Video export.
7) Interactive Open3D viewer (point approximation).

Dependencies: gsplat, colmap CLI, PyTorch, OpenCV, Open3D, imageio, numpy, tqdm, rembg (optional).

Author: You.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import open3d as o3d
except Exception:
    o3d = None

from gsplat import DefaultStrategy, rasterization


def run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """Run a subprocess command with live printing and error on failure."""
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def ensure_dir(path: Path) -> None:
    """Create directory path if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 2.0,
    max_edge: int = 1600,
    start_time_s: float = 0.0,
    duration_s: Optional[float] = None,
    mask_foreground: bool = False,
) -> List[Path]:
    """
    Extract frames from a video at a fixed FPS, resize to bound max_edge, optionally apply a foreground mask.

    Returns a list of saved frame paths.
    """
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stride = max(1, int(round(native_fps / target_fps)))
    start_frame = int(start_time_s * native_fps)
    end_frame = (
        total_frames
        if duration_s is None
        else min(total_frames, int((start_time_s + duration_s) * native_fps))
    )

    frame_paths: List[Path] = []
    saved = 0

    remover = None
    if mask_foreground:
        try:
            from rembg import remove

            remover = remove
        except Exception:
            print("[warn] rembg not available; continuing without masks.")
            remover = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pbar = tqdm(total=end_frame - start_frame, desc=f"Extract {video_path.name}")

    while True:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or pos >= end_frame:
            break

        if (pos - start_frame) % stride != 0:
            pbar.update(1)
            continue

        height, width = frame.shape[:2]
        scale = min(max_edge / float(max(height, width)), 1.0)
        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        if remover is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgba = remover(rgb)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            rgb_masked = (
                rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            ).astype(np.uint8)
            frame = cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR)

        out_path = out_dir / f"frame_{saved:06d}.png"
        cv2.imwrite(str(out_path), frame)
        frame_paths.append(out_path)
        saved += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if len(frame_paths) < 20:
        print("[warn] Very few frames extracted; COLMAP may fail to register.")

    return frame_paths


def _logit(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Stable logit transform applied element-wise."""
    x_clamped = np.clip(x, eps, 1.0 - eps)
    return np.log(x_clamped) - np.log(1.0 - x_clamped)

def colmap_reconstruct_from_images(
    images_dir: Path,
    work_dir: Path,
    use_gpu: bool = True,
    undistort: bool = True,
    max_image_size_undist: int = 2000,
    camera_model: str = "OPENCV",
) -> Dict[str, Path]:
    """
    Run COLMAP CLI to get SfM cameras and a sparse point cloud.

    Returns a dict with keys:
      model_txt: path to text model directory (cameras.txt, images.txt, points3D.txt)
      images_final: images directory to use downstream (undistorted if requested)
    """
    ensure_dir(work_dir)
    database = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    dense_dir = work_dir / "dense"
    ensure_dir(sparse_dir)

    run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            camera_model,
            "--SiftExtraction.use_gpu",
            "1" if use_gpu else "0",
        ],
        cwd=work_dir,
    )

    run(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            "1" if use_gpu else "0",
        ],
        cwd=work_dir,
    )

    run(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_dir),
        ],
        cwd=work_dir,
    )

    model0 = sparse_dir / "0"
    run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(model0),
            "--output_path",
            str(model0),
            "--output_type",
            "TXT",
        ],
        cwd=work_dir,
    )

    model_txt_dir = model0

    if undistort:
        ensure_dir(dense_dir)
        run(
            [
                "colmap",
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(model0),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(max_image_size_undist),
            ],
            cwd=work_dir,
        )

        model_txt_dir = dense_dir / "sparse"
        images_final = dense_dir / "images"

        if not (model_txt_dir / "cameras.txt").exists():
            run(
                [
                    "colmap",
                    "model_converter",
                    "--input_path",
                    str(model_txt_dir),
                    "--output_path",
                    str(model_txt_dir),
                    "--output_type",
                    "TXT",
                ],
                cwd=work_dir,
            )
    else:
        images_final = images_dir

    return {"model_txt": model_txt_dir, "images_final": images_final}

@dataclass
class CameraCOLMAP:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class View:
    image_id: int
    camera_id: int
    name: str
    w2c: np.ndarray
    K: np.ndarray
    width: int
    height: int
    dist: Dict[str, np.ndarray]


def _qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_colmap_txt_model(
    model_dir: Path,
) -> Tuple[Dict[int, CameraCOLMAP], List[View], np.ndarray]:
    """
    Parse cameras.txt, images.txt, points3D.txt from COLMAP text format.

    Returns:
      cameras: mapping from camera_id to CameraCOLMAP
      views: list of View objects containing w2c matrices and intrinsics
      points: Nx6 array of sparse points with XYZRGB values
    """
    cameras: Dict[int, CameraCOLMAP] = {}
    views: List[View] = []

    cam_path = model_dir / "cameras.txt"
    with cam_path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            cam_id = int(tokens[0])
            model = tokens[1]
            width, height = int(tokens[2]), int(tokens[3])
            params = list(map(float, tokens[4:]))
            cameras[cam_id] = CameraCOLMAP(cam_id, model, width, height, params)

    def intrinsics_from_model(
        camera: CameraCOLMAP,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        model = camera.model.upper()
        params = camera.params
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            fx = fy = f
        elif model.startswith("OPENCV"):
            fx, fy, cx, cy = params[:4]
        elif model.startswith("SIMPLE_RADIAL"):
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported camera model: {model}")

        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        distortion: Dict[str, np.ndarray] = {}
        if model.startswith("OPENCV"):
            coeffs = np.zeros((6,), dtype=np.float64)
            count = min(6, max(0, len(params) - 4))
            coeffs[:count] = np.array(params[4 : 4 + count], dtype=np.float64)
            distortion["radial_coeffs"] = coeffs[[0, 1, 4]]
            distortion["tangential_coeffs"] = coeffs[[2, 3]]
        return intrinsic, distortion

    img_path = model_dir / "images.txt"
    with img_path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 10:
                continue
            image_id = int(tokens[0])
            qvec = np.array(list(map(float, tokens[1:5])))
            tvec = np.array(list(map(float, tokens[5:8])))
            camera_id = int(tokens[8])
            name = tokens[9]
            rotation = _qvec2rotmat(qvec)
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = tvec

            camera = cameras[camera_id]
            intrinsic, distortion = intrinsics_from_model(camera)
            views.append(
                View(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    w2c=w2c,
                    K=intrinsic,
                    width=camera.width,
                    height=camera.height,
                    dist=distortion,
                )
            )

    points_xyzrgb: List[List[float]] = []
    p3_path = model_dir / "points3D.txt"
    if p3_path.exists():
        with p3_path.open("r", encoding="utf8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                if len(tokens) < 8:
                    continue
                x, y, z = map(float, tokens[1:4])
                r, g, b = map(float, tokens[4:7])
                points_xyzrgb.append([x, y, z, r / 255.0, g / 255.0, b / 255.0])
    points = (
        np.array(points_xyzrgb, dtype=np.float32)
        if points_xyzrgb
        else np.zeros((0, 6), dtype=np.float32)
    )

    return cameras, views, points


@dataclass
class GaussianParams:
    means: torch.nn.Parameter
    quats: torch.nn.Parameter
    scales: torch.nn.Parameter
    opacities: torch.nn.Parameter
    colors: torch.nn.Parameter


def initialize_gaussians_from_sparse_points(
    points_xyzrgb: np.ndarray, device: torch.device, min_points: int = 1000
) -> GaussianParams:
    """
    Initialize Gaussian parameters from COLMAP sparse point cloud.
    """
    if points_xyzrgb.shape[0] == 0:
        raise RuntimeError(
            "No sparse points from COLMAP; cannot initialize Gaussians. "
            "Ensure enough overlap and texture in frames."
        )

    pts = points_xyzrgb[:, :3].astype(np.float32)
    cols = points_xyzrgb[:, 3:6].astype(np.float32)

    if pts.shape[0] < min_points:
        reps = int(math.ceil(min_points / float(pts.shape[0])))
        pts = np.repeat(pts, reps, axis=0)[:min_points]
        cols = np.repeat(cols, reps, axis=0)[:min_points]
        pts += np.random.normal(scale=0.005, size=pts.shape).astype(np.float32)

    bb_min, bb_max = pts.min(0), pts.max(0)
    scene_diag = float(np.linalg.norm(bb_max - bb_min))
    base_scale = np.full(
        (pts.shape[0], 3), 0.01 * (scene_diag + 1e-6), dtype=np.float32
    )

    means = torch.nn.Parameter(torch.from_numpy(pts).to(device))
    quats = torch.nn.Parameter(
        torch.zeros((pts.shape[0], 4), dtype=torch.float32, device=device)
    )
    quats.data[:, 0] = 1.0
    scales = torch.nn.Parameter(
        torch.from_numpy(np.log(np.clip(base_scale, 1e-3, None))).to(device)
    )
    opacities = torch.nn.Parameter(
        torch.full((pts.shape[0],), -2.0, dtype=torch.float32, device=device)
    )
    colors = torch.nn.Parameter(torch.from_numpy(_logit(cols)).to(device))

    return GaussianParams(means, quats, scales, opacities, colors)

@dataclass
class Dataset:
    views: List[View]
    image_root: Path
    downsample: int = 1

    def __post_init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.views)

    def get_image(self, idx: int) -> np.ndarray:
        view = self.views[idx]
        image_path = self.image_root / view.name
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            candidates = list(self.image_root.glob(Path(view.name).name))
            if candidates:
                image_path = candidates[0]
        key = str(image_path)
        if key not in self._cache:
            img = imageio.imread(image_path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if self.downsample > 1:
                height, width = img.shape[:2]
                new_width = max(1, int(round(width / self.downsample)))
                new_height = max(1, int(round(height / self.downsample)))
                img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )
            self._cache[key] = img
        return self._cache[key]

    def camera_tensors(
        self, indices: List[int], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        view_mats: List[np.ndarray] = []
        intrinsic_mats: List[np.ndarray] = []
        height, width = None, None
        for idx in indices:
            view = self.views[idx]
            intrinsic = view.K.copy()
            if self.downsample > 1:
                width = max(1, int(round(view.width / self.downsample)))
                height = max(1, int(round(view.height / self.downsample)))
                scale_x = width / view.width
                scale_y = height / view.height
                intrinsic[0, :] *= scale_x
                intrinsic[1, :] *= scale_y
            else:
                width = view.width
                height = view.height
            view_mats.append(view.w2c.astype(np.float32))
            intrinsic_mats.append(intrinsic.astype(np.float32))
        view_tensor = torch.from_numpy(np.stack(view_mats, axis=0)).to(device)
        intrinsic_tensor = torch.from_numpy(np.stack(intrinsic_mats, axis=0)).to(device)
        return view_tensor, intrinsic_tensor, (height or 0, width or 0)


class GaussianModel(torch.nn.Module):
    """Wrap learnable Gaussian parameters and expose a forward pass that renders via gsplat."""

    def __init__(self, params: GaussianParams):
        super().__init__()
        self.params = params

    @staticmethod
    def _activate(
        means: torch.Tensor,
        quats_raw: torch.Tensor,
        scales_raw: torch.Tensor,
        opacities_raw: torch.Tensor,
        colors_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quats = F.normalize(quats_raw, dim=-1)
        scales = torch.exp(scales_raw)
        opacities = torch.sigmoid(opacities_raw)
        colors = torch.clamp(torch.sigmoid(colors_raw), 0.0, 1.0)
        return means, quats, scales, opacities, colors

    def forward(
        self,
        view_mats: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        antialias: bool = True,
        radius_clip: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        means, quats, scales, opacities, colors = self._activate(
            self.params.means,
            self.params.quats,
            self.params.scales,
            self.params.opacities,
            self.params.colors,
        )
        rasterize_mode = "antialiased" if antialias else "classic"
        rgb, alpha, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view_mats,
            Ks=intrinsics,
            width=width,
            height=height,
            render_mode="RGB",
            packed=True,
            rasterize_mode=rasterize_mode,
            radius_clip=radius_clip,
        )
        return rgb, alpha, meta


def train_gaussians(
    dataset: Dataset,
    params: GaussianParams,
    out_dir: Path,
    iters: int = 15000,
    batch: int = 1,
    lr: float = 1e-2,
    l2_reg_scales: float = 1e-5,
    l2_reg_opacity: float = 1e-6,
    device: str = "cuda",
    antialias: bool = True,
    radius_clip: float = 0.0,
) -> GaussianParams:
    """
    Photometric optimization with MSE loss and mild regularization.
    """
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GaussianModel(params).to(device_t)

    optimizers = {
        "means": torch.optim.Adam([params.means], lr=lr),
        "quats": torch.optim.Adam([params.quats], lr=lr),
        "scales": torch.optim.Adam([params.scales], lr=lr),
        "opacities": torch.optim.Adam([params.opacities], lr=lr * 0.1),
        "colors": torch.optim.Adam([params.colors], lr=lr),
    }

    strategy = DefaultStrategy()
    strategy.check_sanity(
        torch.nn.ParameterDict(
            {
                "means": params.means,
                "quats": params.quats,
                "scales": params.scales,
                "opacities": params.opacities,
            }
        ),
        {
            k: v
            for k, v in optimizers.items()
            if k in ["means", "quats", "scales", "opacities"]
        },
    )
    state = strategy.initialize_state()

    ensure_dir(out_dir / "ckpts")
    ensure_dir(out_dir / "logs")

    indices = list(range(len(dataset)))
    pbar = tqdm(range(iters), desc="Optimize Gaussians")

    for step in pbar:
        batch_ids = random.sample(indices, k=min(batch, len(indices)))
        view_mats, intrinsics, shape = dataset.camera_tensors(batch_ids, device_t)
        height, width = shape

        model.train()
        pred_rgb, _, meta = model(
            view_mats,
            intrinsics,
            width=width,
            height=height,
            antialias=antialias,
            radius_clip=radius_clip,
        )

        gts: List[np.ndarray] = []
        for idx in batch_ids:
            img = dataset.get_image(idx).astype(np.float32) / 255.0
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            gts.append(img)
        gt = torch.from_numpy(np.stack(gts, axis=0)).to(device_t)

        loss_rgb = F.mse_loss(pred_rgb, gt)
        loss_reg = l2_reg_scales * (
            torch.exp(params.scales).pow(2).mean()
        ) + l2_reg_opacity * (torch.sigmoid(params.opacities).pow(2).mean())
        loss = loss_rgb + loss_reg

        strategy.step_pre_backward(
            torch.nn.ParameterDict(
                {
                    "means": params.means,
                    "quats": params.quats,
                    "scales": params.scales,
                    "opacities": params.opacities,
                }
            ),
            {
                k: v
                for k, v in optimizers.items()
                if k in ["means", "quats", "scales", "opacities"]
            },
            state,
            step,
            meta,
        )

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()

        strategy.step_post_backward(
            torch.nn.ParameterDict(
                {
                    "means": params.means,
                    "quats": params.quats,
                    "scales": params.scales,
                    "opacities": params.opacities,
                }
            ),
            {
                k: v
                for k, v in optimizers.items()
                if k in ["means", "quats", "scales", "opacities"]
            },
            state,
            step,
            meta,
            packed=True,
        )

        pbar.set_postfix({"mse": f"{float(loss_rgb):.4f}", "N": params.means.shape[0]})

        if (step + 1) % 2000 == 0 or step == iters - 1:
            checkpoint = {
                "step": step,
                "means": params.means.detach().cpu().numpy(),
                "quats": F.normalize(params.quats, dim=-1).detach().cpu().numpy(),
                "scales": torch.exp(params.scales).detach().cpu().numpy(),
                "opacities": torch.sigmoid(params.opacities).detach().cpu().numpy(),
                "colors": torch.clamp(torch.sigmoid(params.colors), 0, 1)
                .detach()
                .cpu()
                .numpy(),
            }
            np.save(
                out_dir / "ckpts" / f"gaussians_step{step:06d}.npy",
                checkpoint,
                allow_pickle=True,
            )

    return params


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
) -> np.ndarray:
    """Build a world-to-camera view matrix looking from eye to target."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up_norm = up / (np.linalg.norm(up) + 1e-8)
    side = np.cross(forward, up_norm)
    side = side / (np.linalg.norm(side) + 1e-8)
    up_ortho = np.cross(side, forward)

    rotation = np.stack([side, up_ortho, -forward], axis=0)
    translation = -rotation @ eye
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = translation
    return view


def render_trajectory(
    params: GaussianParams,
    center: np.ndarray,
    radius: float,
    height: int,
    width: int,
    intrinsic: np.ndarray,
    n_frames: int,
    out_dir: Path,
    device: str = "cuda",
    antialias: bool = True,
    radius_clip: float = 0.0,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> List[Path]:
    """
    Render a circular trajectory around the scene center.
    """
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GaussianModel(params).to(device_t)
    ensure_dir(out_dir)
    frames: List[Path] = []

    Ks = torch.from_numpy(np.stack([intrinsic], axis=0).astype(np.float32)).to(device_t)

    for i in tqdm(range(n_frames), desc="Render novel views"):
        theta = 2 * math.pi * (i / n_frames)
        eye = center + radius * np.array(
            [math.cos(theta), 0.1, math.sin(theta)], dtype=np.float32
        )
        view = look_at(eye, center)
        view_tensor = torch.from_numpy(np.stack([view], axis=0)).to(device_t)

        rgb, alpha, _ = model(
            view_tensor,
            Ks,
            width=width,
            height=height,
            antialias=antialias,
            radius_clip=radius_clip,
        )
        rgb_np = rgb[0].detach().cpu().numpy()
        alpha_np = alpha[0].detach().cpu().numpy()
        alpha_np = np.squeeze(alpha_np)
        if alpha_np.ndim == 2:
            alpha_np = alpha_np[..., None]
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        comp = rgb_np * alpha_np + (1.0 - alpha_np) * bg
        rgb_img = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)
        out_path = out_dir / f"novel_{i:04d}.png"
        imageio.imwrite(out_path, rgb_img)
        frames.append(out_path)

    return frames


def write_mp4_from_frames(frames: List[Path], out_path: Path, fps: int = 24) -> None:
    """Encode frames as an MP4 using imageio-ffmpeg."""
    ensure_dir(out_path.parent)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    for frame_path in tqdm(frames, desc=f"Write {out_path.name}"):
        image = imageio.imread(frame_path)
        writer.append_data(image)
    writer.close()


def export_ply_and_view(
    params: GaussianParams, out_ply: Path, view_now: bool = True
) -> None:
    """
    Export Gaussian means as a colored point cloud and launch the Open3D viewer.
    """
    if o3d is None:
        print("[warn] open3d not installed; skipping viewer.")
        return

    colors = torch.clamp(torch.sigmoid(params.colors), 0, 1).detach().cpu().numpy()
    means = params.means.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    ensure_dir(out_ply.parent)
    o3d.io.write_point_cloud(str(out_ply), pcd)
    print(f"[info] wrote point cloud: {out_ply}")

    if view_now:
        o3d.visualization.draw_geometries([pcd])


def process_one_scene(
    scene_dir: Path,
    video_path: Path,
    target_fps: float = 2.0,
    max_edge: int = 1600,
    downsample_factor_for_training: int = 2,
    iters: int = 15000,
    batch: int = 1,
    device: str = "cuda",
    mask_foreground: bool = False,
) -> None:
    """Run the full pipeline for a single scene."""
    print(f"\n=== Scene: {scene_dir.name} ===")
    frames_dir = scene_dir / "frames"
    colmap_dir = scene_dir / "colmap"
    train_dir = scene_dir / "training"

    frame_paths = extract_frames_from_video(
        video_path,
        frames_dir,
        target_fps=target_fps,
        max_edge=max_edge,
        mask_foreground=mask_foreground,
    )

    images_dir = colmap_dir / "images"
    ensure_dir(colmap_dir)
    if images_dir.exists():
        shutil.rmtree(images_dir)
    shutil.copytree(frames_dir, images_dir)

    colmap_out = colmap_reconstruct_from_images(
        images_dir=images_dir,
        work_dir=colmap_dir,
        use_gpu=True,
        undistort=True,
    )
    model_txt = colmap_out["model_txt"]
    images_final = colmap_out["images_final"]

    _, views, points = read_colmap_txt_model(model_txt)
    if len(views) == 0:
        raise RuntimeError(
            "No registered views from COLMAP. Verify capture coverage/overlap."
        )

    dataset = Dataset(
        views=views,
        image_root=images_final,
        downsample=downsample_factor_for_training,
    )

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    params = initialize_gaussians_from_sparse_points(
        points, device=device_t, min_points=2000
    )

    ensure_dir(train_dir)
    trained = train_gaussians(
        dataset,
        params,
        train_dir,
        iters=iters,
        batch=batch,
        device=device,
        lr=1e-2,
    )

    means_np = trained.means.detach().cpu().numpy()
    center = means_np.mean(0).astype(np.float32)
    radius = float(np.linalg.norm(means_np.max(0) - means_np.min(0))) * 0.7 + 1e-3

    first_view = views[0]
    intrinsic = first_view.K.copy()
    height = int(round(first_view.height / dataset.downsample))
    width = int(round(first_view.width / dataset.downsample))
    if dataset.downsample > 1:
        scale_x = width / first_view.width
        scale_y = height / first_view.height
        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y
    novel_dir = train_dir / "novel_views"
    frames = render_trajectory(
        trained,
        center=center,
        radius=radius,
        height=height,
        width=width,
        intrinsic=intrinsic,
        n_frames=180,
        out_dir=novel_dir,
        device=device,
    )
    write_mp4_from_frames(frames, train_dir / "novel_views.mp4", fps=24)

    export_ply_and_view(trained, train_dir / "gaussians.ply", view_now=False)
    print(
        f"[done] Scene {scene_dir.name}: {len(frames)} novel views rendered -> {train_dir / 'novel_views.mp4'}"
    )


def main() -> None:
    set_seed(7)
    root = Path("workspace/columbia_statues").absolute()
    ensure_dir(root)

    video_a = Path("/mnt/data/1.MP4")
    video_b = Path("/mnt/data/2.MP4")

    process_one_scene(
        scene_dir=root / "statue_A",
        video_path=video_a,
        target_fps=2.0,
        max_edge=1600,
        downsample_factor_for_training=2,
        iters=15000,
        batch=1,
        device="cuda",
        mask_foreground=False,
    )

    process_one_scene(
        scene_dir=root / "statue_B",
        video_path=video_b,
        target_fps=2.0,
        max_edge=1600,
        downsample_factor_for_training=2,
        iters=15000,
        batch=1,
        device="cuda",
        mask_foreground=False,
    )


if __name__ == "__main__":
    main()
