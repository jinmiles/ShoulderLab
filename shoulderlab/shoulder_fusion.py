"""Robust multiview skeleton fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from shoulderlab.shoulder_json import write_json


def fuse_aligned_joints(
    joints_by_view: Dict[str, np.ndarray],
    confidence_by_view: Optional[Dict[str, np.ndarray]] = None,
    max_disagreement: Optional[float] = None,
    smoothing_window: int = 7,
) -> dict:
    """Fuse aligned view-specific joints into one robust joint sequence."""
    if not joints_by_view:
        raise ValueError("joints_by_view is empty")
    view_names = sorted(joints_by_view)
    frame_count = min(len(joints_by_view[view]) for view in view_names)
    stack = np.stack([np.asarray(joints_by_view[view], dtype=np.float64)[:frame_count] for view in view_names])
    median = np.nanmedian(stack, axis=0)
    distances = np.linalg.norm(stack - median[None], axis=-1)

    finite_distances = distances[np.isfinite(distances)]
    if max_disagreement is None:
        if finite_distances.size:
            robust_scale = 1.4826 * np.median(np.abs(finite_distances - np.median(finite_distances)))
            max_disagreement = max(0.10, float(np.median(finite_distances) + 3.0 * robust_scale))
        else:
            max_disagreement = 0.10

    weights = np.ones_like(distances, dtype=np.float64)
    weights[~np.isfinite(distances)] = 0.0
    weights[distances > max_disagreement] = 0.0

    if confidence_by_view:
        for view_idx, view in enumerate(view_names):
            conf = confidence_by_view.get(view)
            if conf is None:
                continue
            conf = np.asarray(conf, dtype=np.float64)[:frame_count]
            if conf.ndim == 1:
                conf = conf[:, None]
            weights[view_idx] *= np.clip(conf, 0.0, None)

    weighted = stack * weights[..., None]
    weight_sum = np.sum(weights, axis=0)
    fused = np.divide(
        np.sum(weighted, axis=0),
        weight_sum[..., None],
        out=median.copy(),
        where=weight_sum[..., None] > 1e-12,
    )
    fused_smoothed = smooth_sequence(fused, smoothing_window=smoothing_window)

    per_view_disagreement = {
        view: _summary_stats(distances[view_idx])
        for view_idx, view in enumerate(view_names)
    }
    outlier_mask = distances > max_disagreement
    quality = {
        "view_names": view_names,
        "frames": int(frame_count),
        "joints": int(fused.shape[1]),
        "max_disagreement": float(max_disagreement),
        "outlier_joint_observations": int(np.sum(outlier_mask)),
        "outlier_rate": float(np.mean(outlier_mask)),
        "per_view_disagreement": per_view_disagreement,
        "weight_sum": _summary_stats(weight_sum),
        "smoothing_window": int(smoothing_window),
    }
    return {
        "fused_joints": fused_smoothed,
        "raw_fused_joints": fused,
        "median_joints": median,
        "weights": weights,
        "outlier_mask": outlier_mask,
        "distances": distances,
        "quality": quality,
        "view_names": view_names,
    }


def smooth_sequence(joints: np.ndarray, smoothing_window: int = 7) -> np.ndarray:
    """Apply light temporal smoothing to a joint sequence."""
    joints = np.asarray(joints, dtype=np.float64)
    if smoothing_window <= 1 or len(joints) < 5:
        return joints.copy()

    window = min(int(smoothing_window), len(joints) if len(joints) % 2 == 1 else len(joints) - 1)
    if window < 5:
        return joints.copy()
    if window % 2 == 0:
        window -= 1

    try:
        from scipy.signal import savgol_filter

        return savgol_filter(joints, window_length=window, polyorder=2, axis=0, mode="interp")
    except Exception:
        return _moving_average(joints, window)


def save_fusion_outputs(fusion: dict, output_dir: Path) -> dict:
    """Write fused joints, weights, and quality metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_path = output_dir / "fused_joints.npz"
    weights_path = output_dir / "fusion_weights.npz"
    quality_path = output_dir / "fusion_quality.json"
    np.savez_compressed(
        fused_path,
        fused_joints=fusion["fused_joints"],
        raw_fused_joints=fusion["raw_fused_joints"],
        median_joints=fusion["median_joints"],
        view_names=np.array(fusion["view_names"]),
    )
    np.savez_compressed(
        weights_path,
        weights=fusion["weights"],
        outlier_mask=fusion["outlier_mask"],
        distances=fusion["distances"],
        view_names=np.array(fusion["view_names"]),
    )
    write_json(quality_path, fusion["quality"])
    return {
        "fused_joints": fused_path,
        "fusion_weights": weights_path,
        "fusion_quality": quality_path,
    }


def _moving_average(joints: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(joints, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(joints, dtype=np.float64)
    for idx in range(len(joints)):
        out[idx] = np.mean(padded[idx : idx + window], axis=0)
    return out


def _summary_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "max": None, "std": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }
