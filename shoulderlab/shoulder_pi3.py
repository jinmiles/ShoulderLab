"""Pi3/Pi3X camera pose wrapper for synchronized shoulder trials."""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_OUTPUTS, configure_pi3_paths
from shoulderlab.shoulder_geometry import pairwise_relative_transforms, pose_stability, robust_average_transform
from shoulderlab.shoulder_json import write_json
from shoulderlab.shoulder_manifest import VIEW_LAYOUT


logger = get_logger()

PI3_OUTPUT_ROOT = DATA_OUTPUTS / "shoulder" / "pi3"


def run_pi3_camera_poses(
    trial: dict,
    output_root: Path = PI3_OUTPUT_ROOT,
    model_name: str = "pi3",
    checkpoint: Optional[Path] = None,
    device: str = "cuda",
    max_samples: int = 30,
    sample_interval: Optional[int] = None,
    pixel_limit: int = 255000,
    max_pointcloud_points: int = 200000,
    force: bool = False,
) -> dict:
    """Run Pi3/Pi3X on synchronized sampled frames and save camera geometry artifacts."""
    subject = trial["subject"]
    movement = trial["movement"]
    output_dir = Path(output_root) / subject / movement
    geometry_path = output_dir / "pi3_geometry.npz"
    camera_json_path = output_dir / "camera_poses.json"
    stability_json_path = output_dir / "pose_stability.json"
    pointcloud_path = output_dir / "pointcloud.ply"
    if geometry_path.exists() and camera_json_path.exists() and stability_json_path.exists() and not force:
        logger.info("Reusing Pi3 geometry for %s/%s: %s", subject, movement, geometry_path)
        return {
            "output_dir": output_dir,
            "geometry": geometry_path,
            "camera_poses": camera_json_path,
            "pose_stability": stability_json_path,
            "pointcloud": pointcloud_path if pointcloud_path.exists() else None,
            "reused": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    view_names = list(VIEW_LAYOUT)
    frame_indices = choose_sample_frame_indices(trial, max_samples=max_samples, sample_interval=sample_interval)
    logger.info(
        "Running %s camera pose estimation for %s/%s (%s synchronized frames)",
        model_name,
        subject,
        movement,
        len(frame_indices),
    )
    images, colors, image_shape = load_synchronized_frames(
        {view: Path(trial["views"][view]) for view in view_names},
        frame_indices=frame_indices,
        view_names=view_names,
        pixel_limit=pixel_limit,
    )
    result = _run_pi3_model(images, model_name=model_name, checkpoint=checkpoint, device=device)
    camera_poses_flat = result["camera_poses"]
    expected = len(frame_indices) * len(view_names)
    if len(camera_poses_flat) != expected:
        raise RuntimeError(f"Pi3 returned {len(camera_poses_flat)} camera poses; expected {expected}")
    camera_poses = camera_poses_flat.reshape(len(frame_indices), len(view_names), 4, 4)
    representative_poses = np.stack(
        [robust_average_transform(camera_poses[:, view_idx]) for view_idx in range(len(view_names))],
        axis=0,
    )

    confidence = result.get("confidence")
    image_confidence = None
    if confidence is not None:
        image_confidence = confidence.reshape(expected, -1).mean(axis=1).reshape(len(frame_indices), len(view_names))

    sampled_points, sampled_colors, sampled_confidence = _sample_pointcloud(
        result.get("points"),
        colors,
        confidence,
        max_points=max_pointcloud_points,
    )
    if sampled_points is not None:
        _write_pointcloud(pointcloud_path, sampled_points, sampled_colors)

    relative_poses = pairwise_relative_transforms(camera_poses, view_names)
    np.savez_compressed(
        geometry_path,
        camera_poses=camera_poses,
        representative_poses=representative_poses,
        sampled_frame_indices=np.asarray(frame_indices, dtype=np.int64),
        view_names=np.asarray(view_names),
        image_confidence=image_confidence,
        sampled_points=sampled_points if sampled_points is not None else np.zeros((0, 3)),
        sampled_colors=sampled_colors if sampled_colors is not None else np.zeros((0, 3)),
        sampled_confidence=sampled_confidence if sampled_confidence is not None else np.zeros((0,)),
        image_shape=np.asarray(image_shape, dtype=np.int64),
        **relative_poses,
    )

    camera_payload = {
        "subject": subject,
        "movement": movement,
        "model": model_name,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "coordinate_convention": "OpenCV camera-to-world as returned by Pi3/Pi3X",
        "sampled_frame_indices": frame_indices,
        "view_names": view_names,
        "image_shape": {"height": image_shape[0], "width": image_shape[1]},
        "camera_poses": camera_poses,
        "representative_poses": {
            view: representative_poses[idx]
            for idx, view in enumerate(view_names)
        },
        "image_confidence": image_confidence,
    }
    stability_payload = {
        view: pose_stability(camera_poses[:, idx], representative=representative_poses[idx])
        for idx, view in enumerate(view_names)
    }
    write_json(camera_json_path, camera_payload)
    write_json(stability_json_path, stability_payload)
    logger.info("Pi3 geometry output directory: %s", output_dir.resolve())
    return {
        "output_dir": output_dir,
        "geometry": geometry_path,
        "camera_poses": camera_json_path,
        "pose_stability": stability_json_path,
        "pointcloud": pointcloud_path if pointcloud_path.exists() else None,
        "reused": False,
    }


def choose_sample_frame_indices(
    trial: dict,
    max_samples: int = 30,
    sample_interval: Optional[int] = None,
) -> list:
    """Choose synchronized frame indices from the minimum available view length."""
    frame_counts = []
    for metadata in trial.get("video_metadata", {}).values():
        frame_count = metadata.get("frame_count")
        if frame_count:
            frame_counts.append(int(frame_count))
    if not frame_counts:
        raise ValueError("No frame counts are available in the manifest; rebuild it with OpenCV available")
    min_frames = min(frame_counts)
    if min_frames <= 0:
        raise ValueError("Trial videos have no frames")
    if sample_interval and sample_interval > 0:
        indices = list(range(0, min_frames, sample_interval))
        return indices[:max_samples]
    sample_count = min(max_samples, min_frames)
    return sorted(set(int(round(idx)) for idx in np.linspace(0, min_frames - 1, sample_count)))


def load_synchronized_frames(
    views: Dict[str, Path],
    frame_indices: Sequence[int],
    view_names: Sequence[str],
    pixel_limit: int = 255000,
) -> Tuple["torch.Tensor", np.ndarray, Tuple[int, int]]:
    """Load synchronized trial frames into a Pi3-ready tensor and RGB color array."""
    import cv2
    import torch

    frames = []
    for frame_idx in frame_indices:
        for view in view_names:
            frames.append(_read_rgb_frame(Path(views[view]), frame_idx, cv2=cv2))
    resized = _resize_rgb_frames(frames, pixel_limit=pixel_limit, cv2=cv2)
    colors = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0
    height, width = resized.shape[1], resized.shape[2]
    return tensor, colors, (height, width)


def _run_pi3_model(
    images: "torch.Tensor",
    model_name: str,
    checkpoint: Optional[Path],
    device: str,
) -> dict:
    import torch

    configure_pi3_paths()
    model_name = model_name.lower()
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for Pi3, but torch.cuda.is_available() is false")

    if model_name == "pi3":
        try:
            from pi3.models.pi3 import Pi3
        except TypeError as exc:
            _raise_pi3_import_error(exc)

        model_cls = Pi3
        pretrained_name = "yyfz233/Pi3"
    elif model_name == "pi3x":
        try:
            from pi3.models.pi3x import Pi3X
        except TypeError as exc:
            _raise_pi3_import_error(exc)

        model_cls = Pi3X
        pretrained_name = "yyfz233/Pi3X"
    else:
        raise ValueError("model_name must be 'pi3' or 'pi3x'")

    logger.info("Loading %s model", model_name)
    if checkpoint:
        if model_name == "pi3x":
            model = model_cls(use_multimodal=False).to(device_obj).eval()
        else:
            model = model_cls().to(device_obj).eval()
        if str(checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file

            weights = load_file(str(checkpoint))
        else:
            weights = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        model.load_state_dict(weights, strict=False)
    else:
        model = model_cls.from_pretrained(pretrained_name).to(device_obj).eval()
        if model_name == "pi3x" and hasattr(model, "disable_multimodal"):
            model.disable_multimodal()

    images = images.to(device_obj)
    autocast = nullcontext()
    if device_obj.type == "cuda":
        major = torch.cuda.get_device_capability(device_obj)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        autocast = torch.amp.autocast("cuda", dtype=dtype)

    logger.info("Running Pi3 inference on %s images", images.shape[0])
    with torch.no_grad():
        with autocast:
            if model_name == "pi3x":
                output = model(imgs=images[None], poses=None, depths=None, intrinsics=None)
            else:
                output = model(images[None])

    confidence = torch.sigmoid(output["conf"][0, ..., 0]).detach().float().cpu().numpy()
    return {
        "camera_poses": output["camera_poses"][0].detach().float().cpu().numpy(),
        "points": output["points"][0].detach().float().cpu().numpy() if "points" in output else None,
        "confidence": confidence,
    }


def _read_rgb_frame(path: Path, frame_idx: int, cv2) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Cannot read frame {frame_idx} from {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _resize_rgb_frames(frames: Sequence[np.ndarray], pixel_limit: int, cv2) -> np.ndarray:
    if not frames:
        raise ValueError("No frames loaded")
    height, width = frames[0].shape[:2]
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > pixel_limit else 1.0
    target_w = max(1, round(width * scale / 14)) * 14
    target_h = max(1, round(height * scale / 14)) * 14
    while target_w * target_h > pixel_limit:
        if target_w / target_h > width / height:
            target_w -= 14
        else:
            target_h -= 14
    resized = [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(resized, axis=0)


def _sample_pointcloud(
    points: Optional[np.ndarray],
    colors: np.ndarray,
    confidence: Optional[np.ndarray],
    max_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if points is None or confidence is None:
        return None, None, None
    point_flat = points.reshape(-1, 3)
    color_flat = colors.reshape(-1, 3)
    confidence_flat = confidence.reshape(-1)
    valid = np.isfinite(point_flat).all(axis=1) & np.isfinite(confidence_flat) & (confidence_flat > 0.1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return None, None, None
    if valid_indices.size > max_points:
        keep = np.linspace(0, valid_indices.size - 1, max_points).astype(np.int64)
        valid_indices = valid_indices[keep]
    return point_flat[valid_indices], color_flat[valid_indices], confidence_flat[valid_indices]


def _write_pointcloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    configure_pi3_paths()
    try:
        import torch
        from pi3.utils.basic import write_ply

        write_ply(torch.from_numpy(points).float(), torch.from_numpy(colors).float(), str(path))
    except Exception as exc:
        logger.warning("Could not write Pi3 point cloud %s: %s", path, exc)


def _raise_pi3_import_error(exc: TypeError) -> None:
    if sys.version_info < (3, 9) and "not subscriptable" in str(exc):
        raise RuntimeError(
            "Upstream Pi3/Pi3X is not importable under Python 3.8 because it "
            "uses runtime PEP 585 annotations such as tuple[...]. Run the Pi3 "
            "stage from a Python >=3.9 environment or update upstream Pi3 in "
            "third_party separately. HSMR stages can still run in the "
            "shoulderlab env."
        ) from exc
    raise exc
