"""Geometry helpers for shoulder multiview alignment."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from shoulderlab.rom import JNT


TORSO_ANCHORS = (
    JNT["neck"],
    JNT["r_shoulder"],
    JNT["l_shoulder"],
    JNT["mid_hip"],
    JNT["r_hip"],
    JNT["l_hip"],
)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one or more homogeneous 4x4 transforms."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape == (4, 4):
        return np.linalg.inv(transform)
    return np.linalg.inv(transform)


def transform_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Apply one static transform or a transform sequence to points shaped (T, J, 3)."""
    points = np.asarray(points, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (T, J, 3), got {points.shape}")

    if transforms.shape == (4, 4):
        homog = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        return np.einsum("ij,tbj->tbi", transforms, homog)[..., :3]
    if transforms.shape[:2] == (points.shape[0], 4) and transforms.shape[2:] == (4,):
        homog = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
        return np.einsum("tij,tbj->tbi", transforms, homog)[..., :3]
    raise ValueError(f"transforms must be (4,4) or (T,4,4), got {transforms.shape}")


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Return the unsigned angle of a rotation matrix in degrees."""
    rotation = np.asarray(rotation, dtype=np.float64)
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def robust_average_transform(transforms: np.ndarray) -> np.ndarray:
    """Compute a robust trial-level transform from frame-level camera poses."""
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"transforms must have shape (N, 4, 4), got {transforms.shape}")
    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]

    mean_rotation = np.nanmean(rotations, axis=0)
    u, _, vh = np.linalg.svd(mean_rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh

    averaged = np.eye(4, dtype=np.float64)
    averaged[:3, :3] = rotation
    averaged[:3, 3] = np.nanmedian(translations, axis=0)
    return averaged


def pose_stability(transforms: np.ndarray, representative: Optional[np.ndarray] = None) -> dict:
    """Summarize frame-to-frame deviation from a representative camera pose."""
    transforms = np.asarray(transforms, dtype=np.float64)
    representative = robust_average_transform(transforms) if representative is None else representative
    rep_inv = invert_transform(representative)
    translation_errors = []
    rotation_errors = []
    for transform in transforms:
        delta = rep_inv @ transform
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors.append(rotation_angle_deg(delta[:3, :3]))
    return {
        "translation_error": _summary_stats(np.asarray(translation_errors)),
        "rotation_error_deg": _summary_stats(np.asarray(rotation_errors)),
    }


def expand_sampled_transforms(
    sampled_transforms: np.ndarray,
    sampled_frame_indices: Sequence[int],
    frame_indices: Sequence[int],
) -> np.ndarray:
    """Assign each target frame the nearest sampled camera pose."""
    sampled_transforms = np.asarray(sampled_transforms, dtype=np.float64)
    sampled_frame_indices = np.asarray(sampled_frame_indices, dtype=np.int64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if len(sampled_transforms) != len(sampled_frame_indices):
        raise ValueError("sampled_transforms and sampled_frame_indices length mismatch")
    nearest = np.abs(frame_indices[:, None] - sampled_frame_indices[None, :]).argmin(axis=1)
    return sampled_transforms[nearest]


def estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate source-to-target similarity transform with Umeyama alignment."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[valid]
    target = target[valid]
    if len(source) < 3:
        return 1.0, np.eye(3), np.zeros(3)

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    variance = np.mean(np.sum(source_centered ** 2, axis=1))
    scale = float(np.sum(singular_values * sign) / variance) if with_scale and variance > 1e-12 else 1.0
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a row-vector similarity transform."""
    return scale * (np.asarray(points) @ rotation.T) + translation


def align_sequence_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    anchor_indices: Iterable[int] = TORSO_ANCHORS,
    with_scale: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Framewise similarity-align one joint sequence to a reference sequence."""
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    frame_count = min(len(source), len(reference))
    anchor_indices = list(anchor_indices)
    aligned = np.full_like(source[:frame_count], np.nan, dtype=np.float64)
    scales = np.full(frame_count, np.nan, dtype=np.float64)
    rotation_angles = np.full(frame_count, np.nan, dtype=np.float64)
    translations = np.full((frame_count, 3), np.nan, dtype=np.float64)

    for frame_idx in range(frame_count):
        scale, rotation, translation = estimate_similarity_transform(
            source[frame_idx, anchor_indices],
            reference[frame_idx, anchor_indices],
            with_scale=with_scale,
        )
        aligned[frame_idx] = apply_similarity(source[frame_idx], scale, rotation, translation)
        scales[frame_idx] = scale
        rotation_angles[frame_idx] = rotation_angle_deg(rotation)
        translations[frame_idx] = translation

    report = {
        "scale": _summary_stats(scales),
        "rotation_deg": _summary_stats(rotation_angles),
        "translation_norm": _summary_stats(np.linalg.norm(translations, axis=1)),
    }
    return aligned, report


def root_aligned(points: np.ndarray, root_index: int = JNT["mid_hip"]) -> np.ndarray:
    """Translate a sequence so the root joint is at the origin."""
    points = np.asarray(points, dtype=np.float64)
    return points - points[:, root_index : root_index + 1, :]


def pairwise_relative_transforms(camera_poses: np.ndarray, view_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Compute relative camera transforms for each frame and view pair."""
    camera_poses = np.asarray(camera_poses, dtype=np.float64)
    relatives: Dict[str, np.ndarray] = {}
    for from_idx, from_view in enumerate(view_names):
        from_pose = camera_poses[:, from_idx]
        from_inv = invert_transform(from_pose)
        for to_idx, to_view in enumerate(view_names):
            key = f"{from_view}_to_{to_view}"
            relatives[key] = np.einsum("tij,tjk->tik", from_inv, camera_poses[:, to_idx])
    return relatives


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
