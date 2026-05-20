"""Per-view HSMR execution and SKEL joint recovery for shoulder trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_OUTPUTS, DEFAULT_MODEL_ROOT, configure_hsmr_paths
from shoulderlab.shoulder_manifest import VIEW_LAYOUT


logger = get_logger()

HSMR_OUTPUT_ROOT = DATA_OUTPUTS / "shoulder" / "hsmr"
JOINT_CACHE_ROOT = DATA_OUTPUTS / "shoulder" / "joints"


def run_trial_hsmr(
    trial: dict,
    output_root: Path = HSMR_OUTPUT_ROOT,
    force: bool = False,
    **hsmr_kwargs,
) -> Dict[str, Path]:
    """Run or reuse HSMR reconstruction for each camera view in one trial."""
    from shoulderlab.hsmr import run_hsmr

    subject = trial["subject"]
    movement = trial["movement"]
    outputs: Dict[str, Path] = {}
    for view in VIEW_LAYOUT:
        video_path = Path(trial["views"][view])
        view_output_dir = Path(output_root) / subject / movement / view
        existing = find_hsmr_output(view_output_dir, video_path, required=False)
        if existing is not None and not force:
            logger.info("Reusing HSMR output for %s/%s/%s: %s", subject, movement, view, existing)
            outputs[view] = existing
            continue
        view_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running HSMR for %s/%s/%s", subject, movement, view)
        run_hsmr(input_path=video_path, output_path=view_output_dir, **hsmr_kwargs)
        outputs[view] = find_hsmr_output(view_output_dir, video_path, required=True)
    return outputs


def recover_trial_joints(
    trial: dict,
    hsmr_root: Path = HSMR_OUTPUT_ROOT,
    cache_root: Path = JOINT_CACHE_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    device: str = "cuda:0",
    skel_bs: int = 200,
    force: bool = False,
) -> Dict[str, dict]:
    """Recover and cache SKEL joints for all HSMR view outputs in one trial."""
    subject = trial["subject"]
    movement = trial["movement"]
    caches: Dict[str, dict] = {}
    missing = []
    for view in VIEW_LAYOUT:
        video_path = Path(trial["views"][view])
        hsmr_path = find_hsmr_output(Path(hsmr_root) / subject / movement / view, video_path, required=False)
        if hsmr_path is None:
            missing.append(view)
            continue
        cache_path = Path(cache_root) / subject / movement / view / "joints.npz"
        if cache_path.exists() and not force:
            caches[view] = _load_joint_cache(cache_path)
            logger.info("Reusing joint cache for %s/%s/%s: %s", subject, movement, view, cache_path)
            continue
        caches[view] = {
            "cache_path": cache_path,
            "hsmr_path": hsmr_path,
            "needs_recovery": True,
        }

    if missing:
        raise SystemExit(
            "Missing HSMR outputs for "
            + ", ".join(f"{subject}/{movement}/{view}" for view in missing)
            + ". Run hsmr-shoulder first or use shoulder-pipeline without --skip-hsmr."
        )

    pending = {view: entry for view, entry in caches.items() if entry.get("needs_recovery")}
    if not pending:
        return caches

    logger.info("Building SKEL model for joint recovery from %s", model_root)
    configure_hsmr_paths()
    from lib.modeling.pipelines.hsmr import build_inference_pipeline
    from shoulderlab.analyze import recover_joints

    pipeline = build_inference_pipeline(model_root=Path(model_root), device=device)
    for view, entry in pending.items():
        sequence = load_hsmr_primary_sequence(entry["hsmr_path"])
        joints = recover_joints(sequence["poses"], sequence["betas"], pipeline, device, skel_bs)
        cache_path = Path(entry["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            joints=joints,
            poses=sequence["poses"],
            betas=sequence["betas"],
            frame_indices=sequence["frame_indices"],
            primary_indices=sequence["primary_indices"],
            bbox_scales=sequence["bbox_scales"],
            full_cam_t=sequence["full_cam_t"],
            patch_cam_t=sequence["patch_cam_t"],
            hsmr_path=str(entry["hsmr_path"]),
            view=view,
        )
        logger.info("Wrote joint cache for %s/%s/%s: %s", subject, movement, view, cache_path)
        caches[view] = _load_joint_cache(cache_path)
    return caches


def load_hsmr_primary_sequence(path: Path) -> dict:
    """Load HSMR video/image output and preserve selected primary-subject metadata."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {
            "poses": np.asarray(data["poses"], dtype=np.float32),
            "betas": np.asarray(data["betas"], dtype=np.float32),
            "frame_indices": np.arange(len(data["poses"]), dtype=np.int64),
            "primary_indices": np.zeros(len(data["poses"]), dtype=np.int64),
            "bbox_scales": np.full(len(data["poses"]), np.nan, dtype=np.float32),
            "full_cam_t": np.full((len(data["poses"]), 3), np.nan, dtype=np.float32),
            "patch_cam_t": np.full((len(data["poses"]), 3), np.nan, dtype=np.float32),
        }
    if path.suffix != ".npy":
        raise ValueError(f"Unsupported HSMR output type: {path}")

    frames = np.load(path, allow_pickle=True)
    poses, betas = [], []
    frame_indices, primary_indices, bbox_scales = [], [], []
    full_cam_t, patch_cam_t = [], []
    for frame_idx, frame in enumerate(frames):
        frame = dict(frame)
        frame_poses = frame.get("poses")
        frame_betas = frame.get("betas")
        if frame_poses is None or len(frame_poses) == 0:
            continue
        primary = _primary_person_index(frame.get("bbx_cs"), len(frame_poses))
        poses.append(frame_poses[primary])
        betas.append(frame_betas[primary])
        frame_indices.append(frame_idx)
        primary_indices.append(primary)
        bbox_scales.append(_bbox_scale(frame.get("bbx_cs"), primary))
        full_cam_t.append(_optional_vector(frame.get("full_cam_t"), primary))
        patch_cam_t.append(_optional_vector(frame.get("patch_cam_t"), primary))

    if not poses:
        raise ValueError(f"No detected primary-person frames in {path}")
    return {
        "poses": np.asarray(poses, dtype=np.float32),
        "betas": np.asarray(betas, dtype=np.float32),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "primary_indices": np.asarray(primary_indices, dtype=np.int64),
        "bbox_scales": np.asarray(bbox_scales, dtype=np.float32),
        "full_cam_t": np.asarray(full_cam_t, dtype=np.float32),
        "patch_cam_t": np.asarray(patch_cam_t, dtype=np.float32),
    }


def find_hsmr_output(output_dir: Path, video_path: Path, required: bool = True) -> Optional[Path]:
    """Find the HSMR `.npy`/`.npz` output created for one input video."""
    output_dir = Path(output_dir)
    video_path = Path(video_path)
    candidates = [
        output_dir / f"HSMR-{video_path.stem}.npy",
        output_dir / f"HSMR-{video_path.stem}.npz",
    ]
    candidates.extend(sorted(output_dir.glob("*.npy")))
    candidates.extend(sorted(output_dir.glob("*.npz")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise SystemExit(f"No HSMR output found in {output_dir} for {video_path.name}")
    return None


def _load_joint_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "cache_path": Path(path),
        "hsmr_path": Path(str(data["hsmr_path"])) if "hsmr_path" in data else None,
        "joints": data["joints"],
        "poses": data["poses"],
        "betas": data["betas"],
        "frame_indices": data["frame_indices"],
        "primary_indices": data["primary_indices"],
        "bbox_scales": data["bbox_scales"],
        "full_cam_t": data["full_cam_t"],
        "patch_cam_t": data["patch_cam_t"],
        "view": str(data["view"]) if "view" in data else path.parent.name,
    }


def _primary_person_index(bbx_cs, instance_count: int) -> int:
    if bbx_cs is None:
        return 0
    arr = np.asarray(bbx_cs)
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
        return int(np.argmax(arr[:, 2]))
    return min(0, instance_count - 1)


def _bbox_scale(bbx_cs, primary: int) -> float:
    if bbx_cs is None:
        return float("nan")
    arr = np.asarray(bbx_cs)
    if arr.ndim == 2 and arr.shape[0] > primary and arr.shape[1] >= 3:
        return float(arr[primary, 2])
    return float("nan")


def _optional_vector(value, primary: int) -> np.ndarray:
    if value is None:
        return np.full(3, np.nan, dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[0] > primary:
        return np.asarray(arr[primary, :3], dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] >= 3:
        return np.asarray(arr[:3], dtype=np.float32)
    return np.full(3, np.nan, dtype=np.float32)
