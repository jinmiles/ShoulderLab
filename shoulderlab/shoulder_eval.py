"""Consistency evaluation for shoulder multiview reconstructions."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from shoulderlab.rom import JNT, compute_angles, compute_lcs, compute_temporal_features, transform_to_lcs
from shoulderlab.shoulder_geometry import root_aligned
from shoulderlab.shoulder_json import write_json


JOINT_GROUPS = {
    "right_arm": [JNT["r_shoulder"], JNT["r_elbow"], JNT["r_wrist"]],
    "left_arm": [JNT["l_shoulder"], JNT["l_elbow"], JNT["l_wrist"]],
    "torso": [JNT["neck"], JNT["mid_hip"], JNT["r_hip"], JNT["l_hip"]],
}
LIMB_PAIRS = {
    "right_upper_arm": (JNT["r_shoulder"], JNT["r_elbow"]),
    "right_forearm": (JNT["r_elbow"], JNT["r_wrist"]),
    "left_upper_arm": (JNT["l_shoulder"], JNT["l_elbow"]),
    "left_forearm": (JNT["l_elbow"], JNT["l_wrist"]),
    "shoulder_width": (JNT["r_shoulder"], JNT["l_shoulder"]),
    "hip_width": (JNT["r_hip"], JNT["l_hip"]),
}


def evaluate_sequences(
    joints_by_label: Dict[str, np.ndarray],
    reference_label: str = "cam_c",
    fps: float = 30.0,
    side: str = "both",
) -> dict:
    """Compute view consistency, trajectory, and ROM comparison metrics."""
    labels = sorted(joints_by_label)
    frame_count = min(len(joints_by_label[label]) for label in labels)
    trimmed = {label: np.asarray(joints_by_label[label], dtype=np.float64)[:frame_count] for label in labels}

    pair_metrics = {}
    for left, right in combinations(labels, 2):
        pair_metrics[f"{left}_vs_{right}"] = _pair_metrics(trimmed[left], trimmed[right])

    reference_metrics = {}
    if reference_label in trimmed:
        for label in labels:
            if label == reference_label:
                continue
            reference_metrics[f"{label}_vs_{reference_label}"] = _pair_metrics(trimmed[label], trimmed[reference_label])

    limb_metrics = {
        label: _limb_length_metrics(joints)
        for label, joints in trimmed.items()
    }
    trajectory_metrics = {
        label: _trajectory_metrics(joints, fps=fps)
        for label, joints in trimmed.items()
    }
    rom_metrics = {
        label: _rom_metrics(joints, fps=fps, side=side)
        for label, joints in trimmed.items()
    }

    return {
        "frames": int(frame_count),
        "labels": labels,
        "reference_label": reference_label,
        "pair_metrics": pair_metrics,
        "reference_metrics": reference_metrics,
        "limb_metrics": limb_metrics,
        "trajectory_metrics": trajectory_metrics,
        "rom_metrics": rom_metrics,
    }


def save_evaluation_outputs(report: dict, output_dir: Path) -> dict:
    """Write JSON, CSV, and Markdown comparison reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "view_alignment_report.json"
    csv_path = output_dir / "single_view_vs_fused_metrics.csv"
    md_path = output_dir / "comparison_report.md"
    write_json(json_path, report)
    _write_metric_csv(report, csv_path)
    _write_markdown_report(report, md_path)
    return {
        "view_alignment_report": json_path,
        "single_view_vs_fused_metrics": csv_path,
        "comparison_report": md_path,
    }


def _pair_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    left_root = root_aligned(left)
    right_root = root_aligned(right)
    distances = np.linalg.norm(left_root - right_root, axis=-1)
    group_metrics = {
        name: _summary_stats(distances[:, indices])
        for name, indices in JOINT_GROUPS.items()
    }
    return {
        "root_aligned_mpjpe": _summary_stats(distances),
        "joint_groups": group_metrics,
        "outlier_frame_count": int(np.sum(np.nanmean(distances, axis=1) > 0.20)),
    }


def _limb_length_metrics(joints: np.ndarray) -> dict:
    metrics = {}
    for name, (start, end) in LIMB_PAIRS.items():
        lengths = np.linalg.norm(joints[:, start] - joints[:, end], axis=-1)
        stats = _summary_stats(lengths)
        if stats["mean"]:
            stats["coefficient_of_variation"] = stats["std"] / stats["mean"]
        else:
            stats["coefficient_of_variation"] = None
        metrics[name] = stats
    return metrics


def _trajectory_metrics(joints: np.ndarray, fps: float) -> dict:
    targets = {
        "right_wrist": JNT["r_wrist"],
        "left_wrist": JNT["l_wrist"],
        "right_elbow": JNT["r_elbow"],
        "left_elbow": JNT["l_elbow"],
    }
    metrics = {}
    for name, joint_idx in targets.items():
        trajectory = joints[:, joint_idx]
        velocity = np.diff(trajectory, axis=0) * fps
        acceleration = np.diff(velocity, axis=0) * fps
        speed = np.linalg.norm(velocity, axis=-1)
        accel_norm = np.linalg.norm(acceleration, axis=-1)
        metrics[name] = {
            "speed": _summary_stats(speed),
            "acceleration": _summary_stats(accel_norm),
            "temporal_discontinuity_count": int(np.sum(accel_norm > _adaptive_threshold(accel_norm))),
        }
    return metrics


def _rom_metrics(joints: np.ndarray, fps: float, side: str) -> dict:
    sides = ["right", "left"] if side == "both" else [side]
    R_list, origin_list = compute_lcs(joints)
    joints_local = transform_to_lcs(joints, R_list, origin_list)
    metrics = {}
    for current_side in sides:
        angles = compute_angles(joints_local, side=current_side)
        features = compute_temporal_features(angles, fps=fps)
        metrics[current_side] = {
            angle_name: {
                "min": float(np.nanmin(values)) if np.isfinite(values).any() else None,
                "max": float(np.nanmax(values)) if np.isfinite(values).any() else None,
                "rom": float(np.nanmax(values) - np.nanmin(values)) if np.isfinite(values).any() else None,
            }
            for angle_name, values in angles.items()
        }
        metrics[current_side]["temporal_features"] = features
    return metrics


def _write_metric_csv(report: dict, path: Path) -> None:
    rows: List[dict] = []
    for pair_name, pair in report.get("pair_metrics", {}).items():
        rows.append(
            {
                "metric_scope": "pair",
                "label": pair_name,
                "metric": "root_aligned_mpjpe",
                **_flat_stats(pair["root_aligned_mpjpe"]),
            }
        )
        for group_name, stats in pair.get("joint_groups", {}).items():
            rows.append(
                {
                    "metric_scope": "joint_group",
                    "label": f"{pair_name}:{group_name}",
                    "metric": "distance",
                    **_flat_stats(stats),
                }
            )

    for label, limb_metrics in report.get("limb_metrics", {}).items():
        for limb_name, stats in limb_metrics.items():
            rows.append(
                {
                    "metric_scope": "limb",
                    "label": f"{label}:{limb_name}",
                    "metric": "length",
                    **_flat_stats(stats),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric_scope", "label", "metric", "count", "mean", "median", "max", "std"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# Shoulder Multiview Comparison Report",
        "",
        f"- Frames compared: {report.get('frames')}",
        f"- Reference label: `{report.get('reference_label')}`",
        f"- Labels: {', '.join(f'`{label}`' for label in report.get('labels', []))}",
        "",
        "## Pairwise Root-Aligned MPJPE",
        "",
        "| Pair | Mean | Median | Max |",
        "|:--|--:|--:|--:|",
    ]
    for pair_name, pair in report.get("pair_metrics", {}).items():
        stats = pair["root_aligned_mpjpe"]
        lines.append(
            f"| `{pair_name}` | {_fmt(stats.get('mean'))} | {_fmt(stats.get('median'))} | {_fmt(stats.get('max'))} |"
        )
    lines.extend(["", "## Notes", "", "- Metrics are consistency checks, not clinical diagnostic claims."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _flat_stats(stats: dict) -> dict:
    return {key: stats.get(key) for key in ("count", "mean", "median", "max", "std")}


def _adaptive_threshold(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    return float(np.median(values) + 4.0 * np.std(values))


def _fmt(value: float) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
