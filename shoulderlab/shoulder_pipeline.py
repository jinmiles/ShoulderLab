"""End-to-end multiview shoulder pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_INPUTS, DATA_OUTPUTS, DEFAULT_MODEL_ROOT
from shoulderlab.shoulder_eval import evaluate_sequences, save_evaluation_outputs
from shoulderlab.shoulder_fusion import fuse_aligned_joints, save_fusion_outputs
from shoulderlab.shoulder_geometry import (
    align_sequence_to_reference,
    expand_sampled_transforms,
    transform_points,
)
from shoulderlab.shoulder_hsmr_pipeline import (
    HSMR_OUTPUT_ROOT,
    JOINT_CACHE_ROOT,
    recover_trial_joints,
    run_trial_hsmr,
)
from shoulderlab.shoulder_json import read_json, write_json
from shoulderlab.shoulder_manifest import (
    DEFAULT_REFERENCE_VIEW,
    MANIFEST_OUTPUT,
    SHOULDER_INPUT_ROOT,
    VIEW_LAYOUT,
    build_manifest,
    select_trial,
)
from shoulderlab.shoulder_pi3 import PI3_OUTPUT_ROOT, run_pi3_camera_poses
from shoulderlab.shoulder_rom_analysis import run_joints_rom_analysis, write_single_view_comparison_csv


logger = get_logger()

FUSED_OUTPUT_ROOT = DATA_OUTPUTS / "shoulder" / "fused"
EVAL_OUTPUT_ROOT = DATA_OUTPUTS / "shoulder" / "eval"
ANALYSIS_OUTPUT_ROOT = DATA_OUTPUTS / "shoulder" / "analysis"


def run_shoulder_manifest(
    input_root: Path = SHOULDER_INPUT_ROOT,
    output_path: Path = MANIFEST_OUTPUT,
    subject: Optional[str] = None,
    movement: Optional[str] = None,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    strict: bool = False,
) -> dict:
    """CLI wrapper for manifest discovery."""
    return build_manifest(
        input_root=input_root,
        output_path=output_path,
        subject=subject,
        movement=movement,
        reference_view=reference_view,
        strict=strict,
    )


def run_pi3_shoulder(
    subject: str,
    movement: str,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    **pi3_kwargs,
) -> dict:
    """Run Pi3/Pi3X for one shoulder trial."""
    trial = _get_trial(subject, movement, manifest_path, input_root, reference_view)
    return run_pi3_camera_poses(trial, **pi3_kwargs)


def run_hsmr_shoulder(
    subject: str,
    movement: str,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    output_root: Path = HSMR_OUTPUT_ROOT,
    force: bool = False,
    **hsmr_kwargs,
) -> Dict[str, Path]:
    """Run HSMR for all camera views in one shoulder trial."""
    trial = _get_trial(subject, movement, manifest_path, input_root, reference_view)
    return run_trial_hsmr(trial, output_root=output_root, force=force, **hsmr_kwargs)


def run_fuse_shoulder(
    subject: str,
    movement: str,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    hsmr_root: Path = HSMR_OUTPUT_ROOT,
    cache_root: Path = JOINT_CACHE_ROOT,
    pi3_root: Path = PI3_OUTPUT_ROOT,
    fused_root: Path = FUSED_OUTPUT_ROOT,
    eval_root: Path = EVAL_OUTPUT_ROOT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    device: str = "cuda:0",
    skel_bs: int = 200,
    alignment_variant: str = "static",
    side: str = "both",
    fps: Optional[float] = None,
    require_pi3: bool = False,
    force_joints: bool = False,
    force_fusion: bool = False,
) -> dict:
    """Recover joints, align views to the reference, evaluate, and fuse them."""
    trial = _get_trial(subject, movement, manifest_path, input_root, reference_view)
    if alignment_variant not in {"static", "dynamic"}:
        raise ValueError("alignment_variant must be 'static' or 'dynamic'")

    joint_entries = recover_trial_joints(
        trial,
        hsmr_root=hsmr_root,
        cache_root=cache_root,
        model_root=model_root,
        device=device,
        skel_bs=skel_bs,
        force=force_joints,
    )
    joints_by_view = {view: entry["joints"] for view, entry in joint_entries.items()}
    frame_indices_by_view = {view: entry["frame_indices"] for view, entry in joint_entries.items()}
    bbox_confidence = {
        view: _normalize_confidence(entry["bbox_scales"])
        for view, entry in joint_entries.items()
    }

    pi3_geometry = Path(pi3_root) / subject / movement / "pi3_geometry.npz"
    if pi3_geometry.exists():
        pi3_data = np.load(pi3_geometry, allow_pickle=True)
        camera_pose_available = True
    elif require_pi3:
        raise SystemExit(f"Pi3 geometry is required but missing: {pi3_geometry}")
    else:
        logger.warning("Pi3 geometry is missing; falling back to skeleton-only similarity alignment")
        pi3_data = None
        camera_pose_available = False

    fps = float(fps or _trial_fps(trial, reference_view))
    variant_outputs = {}
    variant_reports = {}
    for variant in ("static", "dynamic"):
        aligned, alignment_report = align_trial_joints(
            joints_by_view=joints_by_view,
            frame_indices_by_view=frame_indices_by_view,
            reference_view=reference_view,
            pi3_data=pi3_data,
            variant=variant,
        )
        variant_outputs[variant] = aligned
        variant_reports[variant] = alignment_report

    aligned_for_fusion = variant_outputs[alignment_variant]
    fusion = fuse_aligned_joints(aligned_for_fusion, confidence_by_view=bbox_confidence)
    fused_dir = Path(fused_root) / subject / movement
    eval_dir = Path(eval_root) / subject / movement
    if force_fusion or not (fused_dir / "fused_joints.npz").exists():
        fusion_paths = save_fusion_outputs(fusion, fused_dir)
    else:
        fusion_paths = {
            "fused_joints": fused_dir / "fused_joints.npz",
            "fusion_weights": fused_dir / "fusion_weights.npz",
            "fusion_quality": fused_dir / "fusion_quality.json",
        }

    _save_alignment_npz(eval_dir, variant_outputs, reference_view)
    evaluation_inputs = dict(aligned_for_fusion)
    evaluation_inputs["abc_fused"] = fusion["fused_joints"]
    evaluation_report = evaluate_sequences(
        evaluation_inputs,
        reference_label=reference_view,
        fps=fps,
        side=side,
    )
    evaluation_report["alignment_variant"] = alignment_variant
    evaluation_report["camera_pose_available"] = camera_pose_available
    evaluation_report["alignment_reports"] = variant_reports
    evaluation_paths = save_evaluation_outputs(evaluation_report, eval_dir)
    logger.info("Fused shoulder output directory: %s", fused_dir.resolve())
    logger.info("Evaluation output directory: %s", eval_dir.resolve())
    return {
        "trial": trial,
        "fused_dir": fused_dir,
        "eval_dir": eval_dir,
        "fusion_paths": fusion_paths,
        "evaluation_paths": evaluation_paths,
        "evaluation_report": evaluation_report,
    }


def run_analyze_shoulder(
    subject: str,
    movement: str,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    fused_root: Path = FUSED_OUTPUT_ROOT,
    analysis_root: Path = ANALYSIS_OUTPUT_ROOT,
    eval_root: Path = EVAL_OUTPUT_ROOT,
    side: str = "both",
    fps: Optional[float] = None,
    sg_window_sec: float = 0.33,
    sg_polyorder: int = 3,
    peak_prominence_deg: Optional[float] = None,
    skip_video: bool = True,
) -> dict:
    """Run ROM analysis on fused joints for one shoulder trial."""
    trial = _get_trial(subject, movement, manifest_path, input_root, reference_view)
    fused_path = Path(fused_root) / subject / movement / "fused_joints.npz"
    if not fused_path.exists():
        raise SystemExit(f"Missing fused joints: {fused_path}. Run fuse-shoulder first.")
    fused = np.load(fused_path, allow_pickle=True)["fused_joints"]
    fps = float(fps or _trial_fps(trial, reference_view))
    analysis_dir = Path(analysis_root) / subject / movement
    result = run_joints_rom_analysis(
        fused,
        output_dir=analysis_dir,
        stem="fused",
        side=side,
        fps=fps,
        sg_window_sec=sg_window_sec,
        sg_polyorder=sg_polyorder,
        peak_prominence_deg=peak_prominence_deg,
        skip_video=skip_video,
        source_label="abc_fused",
    )
    eval_report_path = Path(eval_root) / subject / movement / "view_alignment_report.json"
    if eval_report_path.exists():
        eval_report = read_json(eval_report_path)
        write_single_view_comparison_csv(eval_report, analysis_dir / "single_view_comparison.csv")
    return result


def run_shoulder_pipeline(
    subject: str,
    movement: str,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    skip_pi3: bool = False,
    skip_hsmr: bool = False,
    force: bool = False,
    pi3_kwargs: Optional[dict] = None,
    hsmr_kwargs: Optional[dict] = None,
    fuse_kwargs: Optional[dict] = None,
    analysis_kwargs: Optional[dict] = None,
) -> dict:
    """Run the planned manifest -> Pi3 -> HSMR -> fusion -> ROM pipeline."""
    manifest = run_shoulder_manifest(
        input_root=input_root,
        output_path=manifest_path,
        subject=subject,
        movement=movement,
        reference_view=reference_view,
    )
    trial = select_trial(manifest, subject, movement)
    return _run_trial_pipeline(
        trial=trial,
        manifest_path=manifest_path,
        input_root=input_root,
        reference_view=reference_view,
        skip_pi3=skip_pi3,
        skip_hsmr=skip_hsmr,
        force=force,
        pi3_kwargs=pi3_kwargs,
        hsmr_kwargs=hsmr_kwargs,
        fuse_kwargs=fuse_kwargs,
        analysis_kwargs=analysis_kwargs,
    )


def run_shoulder_dataset(
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    manifest_path: Path = MANIFEST_OUTPUT,
    input_root: Path = SHOULDER_INPUT_ROOT,
    subject: Optional[str] = None,
    movement: Optional[str] = None,
    skip_pi3: bool = False,
    skip_hsmr: bool = False,
    force: bool = False,
    continue_on_error: bool = False,
    pi3_kwargs: Optional[dict] = None,
    hsmr_kwargs: Optional[dict] = None,
    fuse_kwargs: Optional[dict] = None,
    analysis_kwargs: Optional[dict] = None,
) -> dict:
    """Run the end-to-end shoulder pipeline for every valid manifest trial."""
    manifest = run_shoulder_manifest(
        input_root=input_root,
        output_path=manifest_path,
        subject=subject,
        movement=movement,
        reference_view=reference_view,
    )
    valid_trials = [trial for trial in manifest.get("trials", []) if trial.get("valid")]
    invalid_trials = [trial for trial in manifest.get("trials", []) if not trial.get("valid")]
    if invalid_trials:
        logger.warning(
            "Skipping %s invalid shoulder trial(s): %s",
            len(invalid_trials),
            ", ".join(f"{trial['subject']}/{trial['movement']}" for trial in invalid_trials[:10]),
        )
    if not valid_trials:
        raise SystemExit("No valid shoulder trials found")

    outputs = {}
    failures = []
    logger.info("Running shoulder dataset pipeline for %s valid trial(s)", len(valid_trials))
    for index, trial in enumerate(valid_trials, start=1):
        subject_name = trial["subject"]
        movement_name = trial["movement"]
        label = f"{subject_name}/{movement_name}"
        logger.info("Running shoulder trial %s/%s: %s", index, len(valid_trials), label)
        try:
            outputs[label] = _run_trial_pipeline(
                trial=trial,
                manifest_path=manifest_path,
                input_root=input_root,
                reference_view=reference_view,
                skip_pi3=skip_pi3,
                skip_hsmr=skip_hsmr,
                force=force,
                pi3_kwargs=pi3_kwargs,
                hsmr_kwargs=hsmr_kwargs,
                fuse_kwargs=fuse_kwargs,
                analysis_kwargs=analysis_kwargs,
            )
        except SystemExit as exc:
            if exc.code in (0, None):
                raise
            failures.append({"trial": label, "error": _system_exit_message(exc)})
            logger.error("Shoulder trial failed: %s: %s", label, failures[-1]["error"])
            if not continue_on_error:
                raise
        except Exception as exc:
            failures.append({"trial": label, "error": str(exc)})
            logger.exception("Shoulder trial failed: %s", label)
            if not continue_on_error:
                raise

    if failures:
        labels = ", ".join(failure["trial"] for failure in failures[:10])
        raise SystemExit(f"Shoulder dataset finished with {len(failures)} failed trial(s): {labels}")
    logger.info("Finished shoulder dataset pipeline for %s valid trial(s)", len(valid_trials))
    return {"manifest": manifest_path, "outputs": outputs, "failures": failures}


def _run_trial_pipeline(
    trial: dict,
    manifest_path: Path,
    input_root: Path,
    reference_view: str,
    skip_pi3: bool,
    skip_hsmr: bool,
    force: bool,
    pi3_kwargs: Optional[dict],
    hsmr_kwargs: Optional[dict],
    fuse_kwargs: Optional[dict],
    analysis_kwargs: Optional[dict],
) -> dict:
    subject = trial["subject"]
    movement = trial["movement"]
    outputs = {"manifest": manifest_path}

    pi3_kwargs = dict(pi3_kwargs or {})
    hsmr_kwargs = dict(hsmr_kwargs or {})
    fuse_kwargs = dict(fuse_kwargs or {})
    analysis_kwargs = dict(analysis_kwargs or {})

    if not skip_pi3:
        outputs["pi3"] = run_pi3_camera_poses(trial, force=force, **pi3_kwargs)
    else:
        logger.info("Skipping Pi3 stage by request")

    if not skip_hsmr:
        outputs["hsmr"] = run_trial_hsmr(trial, force=force, **hsmr_kwargs)
    else:
        logger.info("Skipping HSMR stage by request")

    outputs["fusion"] = run_fuse_shoulder(
        subject=subject,
        movement=movement,
        manifest_path=manifest_path,
        input_root=input_root,
        reference_view=reference_view,
        force_joints=force,
        force_fusion=force,
        **fuse_kwargs,
    )
    outputs["analysis"] = run_analyze_shoulder(
        subject=subject,
        movement=movement,
        manifest_path=manifest_path,
        input_root=input_root,
        reference_view=reference_view,
        **analysis_kwargs,
    )
    return outputs


def _system_exit_message(exc: SystemExit) -> str:
    if exc.code in (0, None):
        return "0"
    return str(exc.code)


def align_trial_joints(
    joints_by_view: Dict[str, np.ndarray],
    frame_indices_by_view: Dict[str, np.ndarray],
    reference_view: str,
    pi3_data,
    variant: str,
) -> tuple:
    """Apply camera-pose transforms then framewise torso similarity alignment."""
    if reference_view not in joints_by_view:
        raise SystemExit(f"Reference view {reference_view!r} is missing from recovered joints")
    frame_count = min(len(joints) for joints in joints_by_view.values())
    trimmed = {view: np.asarray(joints[:frame_count], dtype=np.float64) for view, joints in joints_by_view.items()}

    transformed = {}
    pose_report = {"variant": variant, "camera_pose_used": pi3_data is not None}
    if pi3_data is not None:
        view_names = [str(view) for view in pi3_data["view_names"]]
        sampled_indices = pi3_data["sampled_frame_indices"]
        camera_poses = pi3_data["camera_poses"]
        representative_poses = pi3_data["representative_poses"]
        for view, joints in trimmed.items():
            view_idx = view_names.index(view)
            if variant == "static":
                transforms = representative_poses[view_idx]
            else:
                target_frames = np.asarray(frame_indices_by_view[view][:frame_count], dtype=np.int64)
                transforms = expand_sampled_transforms(camera_poses[:, view_idx], sampled_indices, target_frames)
            transformed[view] = transform_points(joints, transforms)
    else:
        transformed = trimmed

    reference = transformed[reference_view]
    aligned = {reference_view: reference}
    per_view_reports = {
        reference_view: {
            "scale": {"count": int(frame_count), "mean": 1.0, "median": 1.0, "max": 1.0, "std": 0.0},
            "rotation_deg": {"count": int(frame_count), "mean": 0.0, "median": 0.0, "max": 0.0, "std": 0.0},
            "translation_norm": {"count": int(frame_count), "mean": 0.0, "median": 0.0, "max": 0.0, "std": 0.0},
        }
    }
    for view, joints in transformed.items():
        if view == reference_view:
            continue
        aligned[view], per_view_reports[view] = align_sequence_to_reference(joints, reference)
    pose_report["per_view_similarity_alignment"] = per_view_reports
    return aligned, pose_report


def _get_trial(
    subject: str,
    movement: str,
    manifest_path: Path,
    input_root: Path,
    reference_view: str,
) -> dict:
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        try:
            return select_trial(manifest, subject, movement)
        except SystemExit:
            pass
    manifest = build_manifest(
        input_root=input_root,
        output_path=manifest_path,
        subject=subject,
        movement=movement,
        reference_view=reference_view,
    )
    return select_trial(manifest, subject, movement)


def _trial_fps(trial: dict, reference_view: str) -> float:
    metadata = trial.get("video_metadata", {}).get(reference_view, {})
    fps = metadata.get("fps")
    return float(fps) if fps else 30.0


def _normalize_confidence(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return np.ones(len(values), dtype=np.float64)
    finite = values[np.isfinite(values)]
    low = np.percentile(finite, 5)
    high = np.percentile(finite, 95)
    if high <= low:
        return np.ones(len(values), dtype=np.float64)
    normalized = (values - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 1.0
    return np.clip(normalized, 0.25, 1.0)


def _save_alignment_npz(eval_dir: Path, variants: Dict[str, Dict[str, np.ndarray]], reference_view: str) -> None:
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {"reference_view": reference_view, "variants": {}}
    for variant, joints_by_view in variants.items():
        path = eval_dir / f"aligned_{variant}_joints.npz"
        np.savez_compressed(
            path,
            **{view: joints for view, joints in joints_by_view.items()},
            view_names=np.asarray(sorted(joints_by_view)),
        )
        summary["variants"][variant] = str(path)
    write_json(eval_dir / "alignment_artifacts.json", summary)
