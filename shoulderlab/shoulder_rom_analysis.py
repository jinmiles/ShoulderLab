"""Run ShoulderLab ROM analysis from recovered or fused joint sequences."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from shoulderlab.log import get_logger
from shoulderlab.rom import (
    compute_angles,
    compute_lcs,
    compute_reach_volume,
    compute_temporal_features,
    print_summary,
    render_video,
    transform_to_lcs,
    visualize_angles,
    visualize_reach_space,
    visualize_skeleton_with_reach,
    visualize_temporal_features,
)
from shoulderlab.shoulder_json import write_json


logger = get_logger()


def run_joints_rom_analysis(
    joints: np.ndarray,
    output_dir: Path,
    stem: str = "fused",
    side: str = "both",
    fps: float = 30.0,
    sg_window_sec: float = 0.33,
    sg_polyorder: int = 3,
    peak_prominence_deg: Optional[float] = None,
    skip_video: bool = True,
    source_label: str = "abc_fused",
) -> dict:
    """Analyze a joint sequence directly, without re-loading an HSMR file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joints must have shape (T, J, 3), got {joints.shape}")
    if side not in {"right", "left", "both"}:
        raise ValueError(f"side must be 'right', 'left', or 'both', got {side!r}")

    logger.info("Running ROM analysis for %s joints (%s frames)", source_label, len(joints))
    R_list, origin_list = compute_lcs(joints)
    joints_local = transform_to_lcs(joints, R_list, origin_list)
    sides = ["right", "left"] if side == "both" else [side]

    combined = {
        "source": source_label,
        "stem": stem,
        "frames": int(len(joints)),
        "fps": float(fps),
        "sides": {},
    }
    results_per_side = {}
    angles_per_side = {}

    for current_side in sides:
        logger.info("Computing fused ROM for %s shoulder", current_side.upper())
        angles = compute_angles(joints_local, side=current_side)
        volume, hull, wrist_pts = compute_reach_volume(joints_local, side=current_side)
        temporal_features = compute_temporal_features(
            angles,
            fps=fps,
            sg_window_sec=sg_window_sec,
            sg_polyorder=sg_polyorder,
            peak_prominence_deg=peak_prominence_deg,
        )

        angle_plot_path = output_dir / f"{stem}_{current_side}_angles.png"
        feature_plot_path = output_dir / f"{stem}_{current_side}_temporal_features.png"
        reach_plot_path = output_dir / f"{stem}_{current_side}_reach_3d.png"
        side_json_path = output_dir / f"{stem}_{current_side}_results.json"

        visualize_angles(
            angles,
            save_path=str(angle_plot_path),
            title=f"Shoulder ROM - {current_side.capitalize()} ({source_label})",
            fps=fps,
        )
        visualize_temporal_features(
            angles,
            temporal_features,
            save_path=str(feature_plot_path),
            title=f"Shoulder Temporal Features - {current_side.capitalize()} ({source_label})",
            fps=fps,
        )
        visualize_reach_space(
            wrist_pts,
            hull,
            volume=volume,
            save_path=str(reach_plot_path),
            title=f"3D Arm Reach Space - {current_side.capitalize()} ({source_label})",
        )

        stats = print_summary(angles, volume, current_side)
        for angle_name in ("flexion", "abduction", "ext_rotation"):
            angle_stats = stats.get(angle_name, {})
            if angle_stats.get("max") is not None and angle_stats.get("min") is not None:
                angle_stats["rom"] = angle_stats["max"] - angle_stats["min"]
            else:
                angle_stats["rom"] = None
        stats["source"] = source_label
        stats["frames"] = int(len(joints))
        stats["temporal_features"] = temporal_features
        write_json(side_json_path, stats)

        combined["sides"][current_side] = stats
        results_per_side[current_side] = (volume, hull, wrist_pts)
        angles_per_side[current_side] = angles

    empty = (None, None, np.zeros((0, 3)))
    combined_path = output_dir / f"{stem}_combined_skeleton_reach.png"
    visualize_skeleton_with_reach(
        joints_local,
        results_r=results_per_side.get("right", empty),
        results_l=results_per_side.get("left", empty),
        save_path=str(combined_path),
        title=f"Upper Body Skeleton + Arm Reach Space ({source_label})",
    )

    combined_json_path = output_dir / f"{stem}_results.json"
    write_json(combined_json_path, combined)
    quality_report_path = output_dir / "quality_report.md"
    _write_quality_report(combined, quality_report_path)

    if not skip_video:
        video_path = output_dir / f"{stem}_skeleton_video.mp4"
        render_video(
            joints_local,
            angles_r=angles_per_side.get("right"),
            angles_l=angles_per_side.get("left"),
            save_path=str(video_path),
            fps=fps,
            title=f"Shoulder ROM ({source_label})",
            stride=max(1, len(joints_local) // 300),
        )

    logger.info("Fused ROM analysis output directory: %s", output_dir.resolve())
    return {
        "output_dir": output_dir,
        "results_json": combined_json_path,
        "quality_report": quality_report_path,
        "frames": int(len(joints)),
        "sides": sides,
    }


def write_single_view_comparison_csv(rom_report: dict, output_path: Path) -> None:
    """Write a compact ROM summary CSV from a multi-label evaluation report."""
    rows = []
    for label, label_metrics in rom_report.get("rom_metrics", {}).items():
        for side, side_metrics in label_metrics.items():
            if not isinstance(side_metrics, dict):
                continue
            for angle_name, angle_stats in side_metrics.items():
                if angle_name == "temporal_features":
                    continue
                rows.append(
                    {
                        "label": label,
                        "side": side,
                        "angle": angle_name,
                        "min": angle_stats.get("min"),
                        "max": angle_stats.get("max"),
                        "rom": angle_stats.get("rom"),
                    }
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "side", "angle", "min", "max", "rom"])
        writer.writeheader()
        writer.writerows(rows)


def _write_quality_report(combined: dict, path: Path) -> None:
    lines = [
        "# Fused Shoulder ROM Quality Report",
        "",
        f"- Source: `{combined['source']}`",
        f"- Frames: {combined['frames']}",
        f"- FPS: {combined['fps']}",
        "",
        "## ROM Summary",
        "",
        "| Side | Angle | Min | Max | ROM |",
        "|:--|:--|--:|--:|--:|",
    ]
    for side, stats in combined["sides"].items():
        for angle in ("flexion", "abduction", "ext_rotation"):
            angle_stats = stats.get(angle, {})
            lines.append(
                "| {side} | {angle} | {min_v} | {max_v} | {rom_v} |".format(
                    side=side,
                    angle=angle,
                    min_v=_fmt(angle_stats.get("min")),
                    max_v=_fmt(angle_stats.get("max")),
                    rom_v=_fmt(angle_stats.get("rom")),
                )
            )
    lines.extend(["", "Metrics are analysis outputs and not medical diagnoses."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
