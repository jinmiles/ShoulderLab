"""Discover and validate synchronized shoulder multiview trials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_INPUTS, DATA_OUTPUTS
from shoulderlab.shoulder_json import write_json


logger = get_logger()

VIEW_LAYOUT = {
    "cam_a": "left",
    "cam_b": "right",
    "cam_c": "center",
}
DEFAULT_REFERENCE_VIEW = "cam_c"
SHOULDER_INPUT_ROOT = DATA_INPUTS / "shoulder"
MANIFEST_OUTPUT = DATA_OUTPUTS / "shoulder" / "manifests" / "shoulder_manifest.json"


@dataclass
class VideoProbe:
    """Lightweight video metadata used for sync validation."""

    path: str
    exists: bool
    readable: bool
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


def probe_video(path: Path) -> VideoProbe:
    """Probe one video with OpenCV, falling back to existence checks on failure."""
    path = Path(path)
    if not path.exists():
        return VideoProbe(path=str(path), exists=False, readable=False, error="missing")

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on local environment
        return VideoProbe(
            path=str(path),
            exists=True,
            readable=False,
            error=f"opencv_unavailable: {exc}",
        )

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return VideoProbe(path=str(path), exists=True, readable=False, error="cannot_open")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else None
        readable = frame_count > 0 and width > 0 and height > 0
        return VideoProbe(
            path=str(path),
            exists=True,
            readable=readable,
            fps=fps if fps > 0 else None,
            frame_count=frame_count if frame_count > 0 else None,
            duration_sec=duration,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
            error=None if readable else "empty_or_invalid_metadata",
        )
    finally:
        capture.release()


def _iter_trial_dirs(input_root: Path, subject: Optional[str], movement: Optional[str]) -> Iterable[Path]:
    if subject:
        subject_dirs = [Path(input_root) / subject]
    else:
        subject_dirs = sorted(path for path in Path(input_root).glob("subject*") if path.is_dir())

    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        if movement:
            trial_dirs = [subject_dir / movement]
        else:
            trial_dirs = sorted(path for path in subject_dir.iterdir() if path.is_dir())
        for trial_dir in trial_dirs:
            if trial_dir.is_dir():
                yield trial_dir


def _sync_warnings(probes: Dict[str, VideoProbe]) -> List[str]:
    warnings: List[str] = []
    missing = [view for view, probe in probes.items() if not probe.exists]
    unreadable = [view for view, probe in probes.items() if probe.exists and not probe.readable]
    if missing:
        warnings.append(f"missing_views={','.join(missing)}")
    if unreadable:
        warnings.append(f"unreadable_views={','.join(unreadable)}")

    readable = [probe for probe in probes.values() if probe.readable]
    if len(readable) < 2:
        return warnings

    fps_values = [probe.fps for probe in readable if probe.fps is not None]
    frame_counts = [probe.frame_count for probe in readable if probe.frame_count is not None]
    durations = [probe.duration_sec for probe in readable if probe.duration_sec is not None]
    sizes = {(probe.width, probe.height) for probe in readable}

    if fps_values and max(fps_values) - min(fps_values) > 0.05:
        warnings.append(f"fps_mismatch={min(fps_values):.3f}-{max(fps_values):.3f}")
    if frame_counts and max(frame_counts) - min(frame_counts) > 1:
        warnings.append(f"frame_count_mismatch={min(frame_counts)}-{max(frame_counts)}")
    if durations and max(durations) - min(durations) > 0.05:
        warnings.append(f"duration_mismatch={min(durations):.3f}-{max(durations):.3f}")
    if len(sizes) > 1:
        warnings.append("resolution_mismatch")
    return warnings


def build_trial_entry(trial_dir: Path, reference_view: str = DEFAULT_REFERENCE_VIEW) -> dict:
    """Build one manifest entry from a trial directory."""
    subject = trial_dir.parent.name
    movement = trial_dir.name
    views = {view: trial_dir / f"{view}.mp4" for view in VIEW_LAYOUT}
    probes = {view: probe_video(path) for view, path in views.items()}
    warnings = _sync_warnings(probes)
    valid = all(probe.exists and probe.readable for probe in probes.values()) and not warnings

    return {
        "subject": subject,
        "movement": movement,
        "trial_dir": str(trial_dir),
        "views": {view: str(path) for view, path in views.items()},
        "camera_layout": dict(VIEW_LAYOUT),
        "reference_view": reference_view,
        "sync": "frame_aligned",
        "video_metadata": {view: asdict(probe) for view, probe in probes.items()},
        "valid": valid,
        "warnings": warnings,
    }


def build_manifest(
    input_root: Path = SHOULDER_INPUT_ROOT,
    output_path: Path = MANIFEST_OUTPUT,
    subject: Optional[str] = None,
    movement: Optional[str] = None,
    reference_view: str = DEFAULT_REFERENCE_VIEW,
    strict: bool = False,
) -> dict:
    """Discover shoulder trials, validate synchronized views, and write a manifest."""
    input_root = Path(input_root)
    output_path = Path(output_path)
    if reference_view not in VIEW_LAYOUT:
        raise ValueError(f"reference_view must be one of {sorted(VIEW_LAYOUT)}, got {reference_view!r}")

    trials = [
        build_trial_entry(trial_dir, reference_view=reference_view)
        for trial_dir in _iter_trial_dirs(input_root, subject=subject, movement=movement)
    ]
    if not trials:
        raise SystemExit(f"No shoulder trials found under {input_root}")

    invalid = [trial for trial in trials if not trial["valid"]]
    if strict and invalid:
        labels = ", ".join(f"{trial['subject']}/{trial['movement']}" for trial in invalid[:10])
        raise SystemExit(f"Found invalid shoulder trials: {labels}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_root),
        "camera_layout": dict(VIEW_LAYOUT),
        "reference_view": reference_view,
        "trials": trials,
        "summary": {
            "trial_count": len(trials),
            "valid_trial_count": len(trials) - len(invalid),
            "invalid_trial_count": len(invalid),
        },
    }
    write_json(output_path, manifest)
    logger.info(
        "Wrote shoulder manifest: %s (%s trials, %s valid)",
        output_path,
        len(trials),
        len(trials) - len(invalid),
    )
    return manifest


def select_trial(manifest: dict, subject: str, movement: str) -> dict:
    """Return one trial entry from a manifest."""
    matches = [
        trial
        for trial in manifest.get("trials", [])
        if trial.get("subject") == subject and trial.get("movement") == movement
    ]
    if not matches:
        raise SystemExit(f"No manifest trial found for {subject}/{movement}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple manifest trials found for {subject}/{movement}")
    return matches[0]
