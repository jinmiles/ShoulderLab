"""Load HSMR outputs and run ShoulderLab ROM analysis."""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_OUTPUTS, DEFAULT_MODEL_ROOT, configure_hsmr_paths

configure_hsmr_paths()

from lib.platform.sliding_batches import asb
from lib.modeling.pipelines.hsmr import build_inference_pipeline
from shoulderlab.rom import (
    compute_lcs,
    transform_to_lcs,
    compute_angles,
    compute_temporal_features,
    compute_reach_volume,
    visualize_angles,
    visualize_temporal_features,
    visualize_reach_space,
    visualize_skeleton_with_reach,
    render_video,
    print_summary,
)


logger = get_logger()


def load_video_npy(path: Path):
    """
    Load video mode output (.npy) → poses (T,46), betas (T,10) for primary person.
    Primary person = largest bounding box scale per frame.
    Frames with no detection are skipped.
    """
    data = np.load(path, allow_pickle=True)  # list of T dicts
    logger.info("Loaded %s frames from %s", len(data), path.name)

    poses_list, betas_list = [], []
    for frame in data:
        bbx_cs = frame.get('bbx_cs', None)
        poses  = frame.get('poses',  None)
        betas  = frame.get('betas',  None)

        if poses is None or len(poses) == 0:
            continue

        # Pick the person with the largest bounding box
        # bbx_cs is saved as a list-of-lists or list-of-tuples, convert to ndarray first
        if bbx_cs is not None and len(bbx_cs) > 0:
            bbx_cs_arr = np.array(bbx_cs)  # (N, 3): [cx, cy, scale]
            primary = int(np.argmax(bbx_cs_arr[:, 2]))
        else:
            primary = 0

        poses_list.append(poses[primary])
        betas_list.append(betas[primary])

    poses = np.stack(poses_list)  # (T, 46)
    betas = np.stack(betas_list)  # (T, 10)
    logger.info("Extracted %s frames for the primary person", len(poses))
    return poses, betas


def load_image_npz(path: Path):
    """
    Load image mode output (.npz) → poses (N,46), betas (N,10) for all instances,
    but filter to only retain the main person (largest bounding box scale).
    """
    data = np.load(path, allow_pickle=True)
    poses = data['poses']  # (N, 46)
    betas = data['betas']  # (N, 10)

    num_instances = len(poses)
    if num_instances > 1 and 'bbx_cs' in data:
        bbx_cs = data['bbx_cs']
        if bbx_cs is not None and len(bbx_cs.shape) == 2 and bbx_cs.shape[1] >= 3:
            main_idx = np.argmax(bbx_cs[:, 2])  # pick the one with max bounding box scale
            poses = poses[main_idx:main_idx+1]
            betas = betas[main_idx:main_idx+1]
            logger.info("Loaded %s instances from %s; selected the main person", num_instances, path.name)
        else:
            poses = poses[0:1]
            betas = betas[0:1]
            logger.info("Loaded %s instances from %s; selected the first person", num_instances, path.name)
    else:
        logger.info("Loaded %s instances from %s", num_instances, path.name)

    return poses, betas


# ─────────────────────────────────────────────
#  Joint Recovery via SKEL Model
# ─────────────────────────────────────────────

def recover_joints(poses, betas, pipeline, device, batch_size=200):
    """
    Run SKELWrapper forward to recover 3D joint positions.

    Args:
        poses     : (T, 46) numpy array
        betas     : (T, 10) numpy array
        pipeline  : HSMRPipeline (contains skel_model)
        device    : torch device string
        batch_size: SKEL forward batch size (reduce if OOM)

    Returns:
        joints: (T, 44, 3) numpy array in camera/world space
    """
    logger.info("Recovering joints for %s frames (batch_size=%s)", len(poses), batch_size)
    skel_model = pipeline.skel_model
    joints_all = []

    for bw in tqdm(
        asb(total=len(poses), bs_scope=batch_size, enable_tqdm=False),
        total=int(np.ceil(len(poses) / batch_size)),
        desc='SKEL forward',
    ):
        poses_t = torch.tensor(poses[bw.sid:bw.eid], dtype=torch.float32).to(device)
        betas_t = torch.tensor(betas[bw.sid:bw.eid], dtype=torch.float32).to(device)
        with torch.no_grad():
            skel_out = skel_model(poses=poses_t, betas=betas_t)
        joints_all.append(skel_out.joints.detach().cpu().numpy())  # (B, 44, 3)

    joints = np.concatenate(joints_all, axis=0)  # (T, 44, 3)
    logger.info("Recovered joints with shape %s", joints.shape)
    return joints


def run_analysis(
    input_path: Path,
    output_path: Path = DATA_OUTPUTS / "shoulder_analysis",
    model_root: Path = DEFAULT_MODEL_ROOT,
    side: str = "both",
    device: str = "cuda:0",
    fps: float = 30.0,
    sg_window_sec: float = 0.33,
    sg_polyorder: int = 3,
    peak_prominence_deg: Optional[float] = None,
    skip_video: bool = False,
    skel_bs: int = 200,
) -> dict:
    """Run ROM analysis for one HSMR `.npy` or `.npz` output."""
    input_path = Path(input_path)
    output_root = Path(output_path)
    model_root = Path(model_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem  # used as filename prefix

    # ── 1. Load HSMR output ──────────────────────────────────────────
    if input_path.suffix == '.npy':
        poses, betas = load_video_npy(input_path)
        mode = 'video'
    elif input_path.suffix == '.npz':
        poses, betas = load_image_npz(input_path)
        mode = 'image'
    else:
        raise ValueError(f'Unsupported file type: {input_path.suffix}  (expected .npy or .npz)')

    # ── 2. Build pipeline & recover joints ──────────────────────────
    logger.info("Building SKEL model from %s", model_root)
    pipeline = build_inference_pipeline(
        model_root=model_root,
        device=device,
    )

    joints = recover_joints(poses, betas, pipeline, device, skel_bs)

    # ── 3. LCS Transformation (torso compensation filtering) ─────────
    logger.info("Computing Local Coordinate System (LCS)")
    R_list, origin_list = compute_lcs(joints)
    joints_local = transform_to_lcs(joints, R_list, origin_list)
    logger.info("LCS transformation complete; joints_local shape %s", joints_local.shape)

    # ── 4. Angle computation & Convex Hull ───────────────────────────
    if side not in {"right", "left", "both"}:
        raise ValueError(f"side must be 'right', 'left', or 'both', got {side!r}")
    sides = ['right', 'left'] if side == 'both' else [side]
    results_per_side = {}   # (volume, hull, wrist_pts) keyed by side
    angles_per_side  = {}   # angles dict keyed by side (for video renderer)

    for side in sides:
        logger.info("Computing angles for %s shoulder", side.upper())
        angles = compute_angles(joints_local, side=side)

        if mode == 'video':
            logger.info("Computing 3D arm reach space for %s shoulder", side.upper())
            volume, hull, wrist_pts = compute_reach_volume(joints_local, side=side)
        else:
            from shoulderlab.rom import SIDE_JOINTS
            wrist_idx = SIDE_JOINTS[side][2]
            wrist_pts = joints_local[:, wrist_idx, :]
            volume, hull = None, None

        # ── 5. Visualize ─────────────────────────────────────────────
        angle_plot_path  = output_root / f'{stem}_{side}_angles.png'
        feature_plot_path = output_root / f'{stem}_{side}_temporal_features.png'
        reach_plot_path  = output_root / f'{stem}_{side}_reach_3d.png'
        results_json_path = output_root / f'{stem}_{side}_results.json'

        temporal_features = compute_temporal_features(
            angles,
            fps=fps,
            sg_window_sec=sg_window_sec,
            sg_polyorder=sg_polyorder,
            peak_prominence_deg=peak_prominence_deg,
        )

        visualize_angles(
            angles,
            save_path=str(angle_plot_path),
            title=f'Shoulder ROM — {side.capitalize()} ({stem})',
            fps=fps,
        )

        visualize_temporal_features(
            angles,
            temporal_features,
            save_path=str(feature_plot_path),
            title=f'Shoulder Temporal Features — {side.capitalize()} ({stem})',
            fps=fps,
        )

        visualize_reach_space(
            wrist_pts,
            hull,
            volume=volume,
            save_path=str(reach_plot_path),
            title=f'3D Arm Reach Space — {side.capitalize()} ({stem})',
        )

        # ── 6. Save JSON summary ─────────────────────────────────────
        stats = print_summary(angles, volume, side)
        stats['input'] = str(input_path)
        stats['mode']  = mode
        stats['frames'] = len(joints)
        stats['temporal_features'] = temporal_features

        with open(results_json_path, 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: None if x is None else float(x))
        logger.info("Results saved to %s", results_json_path)

        results_per_side[side] = (volume, hull, wrist_pts)
        angles_per_side[side]  = angles

    # ── 7. Combined skeleton + reach space visualization ─────────────
    # Run for any combination of sides; fill missing side with empty results.
    _EMPTY = (None, None, np.zeros((0, 3)))
    combined_path = output_root / f'{stem}_combined_skeleton_reach.png'
    visualize_skeleton_with_reach(
        joints_local,
        results_r = results_per_side.get('right', _EMPTY),
        results_l = results_per_side.get('left',  _EMPTY),
        save_path = str(combined_path),
        title     = f'Upper Body Skeleton + Arm Reach Space  ({stem})  [LCS]',
    )

    logger.info("Shoulder ROM analysis complete")
    logger.info("Output directory: %s", output_root.resolve())

    # ── 8. Video rendering (only for .npy video input) ────────────
    if mode == 'video' and not skip_video:
        video_path = output_root / f'{stem}_skeleton_video.mp4'
        logger.info("Rendering skeleton video to %s", video_path)
        render_video(
            joints_local,
            angles_r   = angles_per_side.get('right', None),
            angles_l   = angles_per_side.get('left',  None),
            save_path  = str(video_path),
            fps        = fps,
            title      = f'Shoulder ROM  ({stem})  [LCS]',
            stride     = max(1, len(joints_local) // 300),  # cap ~300 rendered frames
        )

    return {
        "input": input_path,
        "output_dir": output_root,
        "mode": mode,
        "frames": len(joints),
        "sides": sides,
    }


def run_batch_analysis(
    input_dir: Path = DATA_OUTPUTS / "UUCM",
    output_dir: Path = DATA_OUTPUTS / "UUCM" / "analysis",
    write_summary: bool = True,
    **analysis_kwargs,
) -> List[Dict]:
    """Run ROM analysis for all reconstructed `.npy` files and write summaries."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"No .npy files found in {input_dir}")

    results = []
    for npy_file in npy_files:
        logger.info("Analyzing %s", npy_file.name)
        results.append(
            run_analysis(
                input_path=npy_file,
                output_path=output_dir,
                **analysis_kwargs,
            )
        )
    if write_summary:
        from shoulderlab.summary import summarize_analysis

        summarize_analysis(input_dir=output_dir)
    return results
