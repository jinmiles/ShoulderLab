"""
Shoulder ROM Analysis Kit
=========================
Implements:
  1. Local Coordinate System (LCS) transformation to filter out torso compensation
  2. Pure shoulder joint angle extraction (Flexion, Abduction, External Rotation)
  3. 3D arm reach space quantification via Convex Hull

Joint Index Reference (SKELWrapper output, total 44 joints):
  First 25 joints = OpenPose25 format:
    0:Nose  1:Neck  2:RShoulder  3:RElbow  4:RWrist
    5:LShoulder  6:LElbow  7:LWrist  8:MidHip
    9:RHip  10:RKnee  11:RAnkle  12:LHip  13:LKnee  14:LAnkle
    15:REye  16:LEye  17:REar  18:LEar  19-24: feet
  Last 19 joints = extra joints from J_regressor_extra
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shoulderlab.log import get_logger


logger = get_logger()


# ─────────────────────────────────────────────
#  Joint Index Definitions
# ─────────────────────────────────────────────
JNT = {
    'neck'        : 1,
    'r_shoulder'  : 2,
    'r_elbow'     : 3,
    'r_wrist'     : 4,
    'l_shoulder'  : 5,
    'l_elbow'     : 6,
    'l_wrist'     : 7,
    'mid_hip'     : 8,
    'r_hip'       : 9,
    'l_hip'       : 12,
}

SIDE_JOINTS = {
    'right': (JNT['r_shoulder'], JNT['r_elbow'], JNT['r_wrist']),
    'left' : (JNT['l_shoulder'], JNT['l_elbow'], JNT['l_wrist']),
}


# ─────────────────────────────────────────────
#  Step 2: Local Coordinate System (LCS)
# ─────────────────────────────────────────────

def compute_lcs(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame spine-pelvis based Local Coordinate System (LCS).

    The LCS is defined as:
      - Origin  : Neck joint (becomes (0,0,0) in LCS)
      - Y-axis  : MidHip → Neck (superior / spine-up)
      - X-axis  : L.Hip → R.Hip, orthogonalised wrt Y (right-lateral)
      - Z-axis  : X × Y  (anterior / forward)

    Args:
        joints: (T, 44, 3) joint positions in camera/global space.

    Returns:
        R_list     : (T, 3, 3) rotation matrices   [col-vectors = X,Y,Z in world]
        origin_list: (T, 3)    LCS origins (pelvis center)
    """
    T = len(joints)
    R_list      = np.zeros((T, 3, 3), dtype=np.float64)
    origin_list = np.zeros((T, 3),    dtype=np.float64)

    for t in range(T):
        mid_hip    = joints[t, JNT['mid_hip']]    # idx 8
        r_hip      = joints[t, JNT['r_hip']]      # idx 9
        l_hip      = joints[t, JNT['l_hip']]      # idx 12
        neck       = joints[t, JNT['neck']]        # idx 1
        r_shoulder = joints[t, JNT['r_shoulder']]  # idx 2
        l_shoulder = joints[t, JNT['l_shoulder']]  # idx 5

        # Origin: Neck → becomes (0, 0, 0) in LCS
        origin = neck

        # Y: spine up direction = MidHip(8) → Neck(1)
        y_raw  = neck - mid_hip
        y_norm = np.linalg.norm(y_raw)
        y_axis = y_raw / y_norm if y_norm > 1e-8 else np.array([0.0, 1.0, 0.0])

        # X: right-lateral defined by hip line (L.Hip → R.Hip)
        # This guarantees R.Hip and L.Hip share the same Y & Z in LCS,
        # i.e. the pelvis line is exactly parallel to the X-axis.
        x_raw = r_hip - l_hip
        x_raw = x_raw - np.dot(x_raw, y_axis) * y_axis   # Gram-Schmidt wrt Y
        x_norm = np.linalg.norm(x_raw)
        x_axis = x_raw / x_norm if x_norm > 1e-8 else np.array([1.0, 0.0, 0.0])

        # Z: anterior / facing direction
        # Y × X to cancel out the Left-Handedness of camera Image Space, maintaining physical Right-Handedness
        z_axis = np.cross(y_axis, x_axis)
        z_norm = np.linalg.norm(z_axis)
        z_axis = z_axis / z_norm if z_norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        # Column-stack: each column is a basis vector
        R_list[t]      = np.column_stack([x_axis, y_axis, z_axis])  # (3, 3)
        origin_list[t] = origin   # = Neck position

    return R_list, origin_list


def transform_to_lcs(
    joints      : np.ndarray,
    R_list      : np.ndarray,
    origin_list : np.ndarray,
) -> np.ndarray:
    """
    Transform joints from global/camera space to the spine-pelvis LCS.

    joints_local = R[t].T @ (joints[t] - origin[t])

    Args:
        joints     : (T, 44, 3)
        R_list     : (T, 3, 3)
        origin_list: (T, 3)

    Returns:
        joints_local: (T, 44, 3)  in LCS
          X = right-lateral, Y = superior, Z = anterior
    """
    T, J, _ = joints.shape
    joints_local = np.zeros_like(joints)
    for t in range(T):
        centered = joints[t] - origin_list[t]          # (44, 3)
        joints_local[t] = (R_list[t].T @ centered.T).T  # (44, 3)
    return joints_local


# ─────────────────────────────────────────────
#  Step 3a: Joint Angle Computation
# ─────────────────────────────────────────────

def compute_angles(
    joints_local : np.ndarray,
    side         : str = 'right',
) -> Dict[str, np.ndarray]:
    """
    Compute pure shoulder joint angles from LCS-transformed joints.

    Local Coordinate System axes:
      X = right-lateral,  Y = superior (up),  Z = anterior (forward)

    Angle Definitions (axis-projected humerus direction):
      Flexion    — signed angle of the humerus projected onto the sagittal
                   YZ plane, from resting down (-Y) toward anterior (+Z).
                   0° = arm at side, positive = forward, negative = extension.
      Abduction  — signed angle of the humerus projected onto the coronal
                   XY plane, from resting down (-Y) toward lateral (+X for
                   right arm, -X for left arm).
                   0° = arm at side, positive = lateral, negative = adduction.
      Ext.Rot    — forearm twist around humerus axis (Swing-Twist decomposition)
                   0° = neutral, positive = external rotation

    Flexion and abduction are independent plane-projected direction angles, not
    a conserved decomposition of one total elevation angle.

    Args:
        joints_local: (T, 44, 3)
        side        : 'right' or 'left'

    Returns:
        dict with keys 'flexion', 'abduction', 'ext_rotation'  (T,) arrays, degrees
    """
    assert side in ('right', 'left'), f"side must be 'right' or 'left', got {side}"
    shoulder_idx, elbow_idx, wrist_idx = SIDE_JOINTS[side]
    # For left side, abduction is toward -X in LCS
    lat_sign = 1.0 if side == 'right' else -1.0

    T = len(joints_local)
    flexion   = np.full(T, np.nan)
    abduction = np.full(T, np.nan)
    ext_rot   = np.full(T, np.nan)

    # Reference resting vectors in LCS
    Y_rest = np.array([0.0, -1.0, 0.0])  # Humerus points down
    F_rest = np.array([0.0, 0.0, 1.0])   # Forearm points forward (anterior)

    for t in range(T):
        shoulder = joints_local[t, shoulder_idx]  # (3,)
        elbow    = joints_local[t, elbow_idx]
        wrist    = joints_local[t, wrist_idx]

        humerus_raw = elbow - shoulder
        h_norm = np.linalg.norm(humerus_raw)
        if h_norm < 1e-6:
            continue
        H = humerus_raw / h_norm

        # ── 1. Flexion & Abduction (axis-projected humerus direction) ──
        # Cosine of the angle from resting posture (-Y); reused below for
        # external-rotation swing/twist.
        cos_el = np.clip(np.dot(H, Y_rest), -1.0, 1.0)

        # Flexion: signed angle in sagittal YZ plane, from -Y toward +Z.
        # If there is no forward/back component, report 0 for this component.
        if abs(H[2]) > 1e-6:
            flexion[t] = np.degrees(np.arctan2(H[2], -H[1]))
        else:
            flexion[t] = 0.0

        # Abduction: signed angle in coronal XY plane, from -Y toward lateral.
        # Right lateral is +X; left lateral is -X, so lat_sign maps both sides
        # to positive abduction.
        lateral = lat_sign * H[0]
        if abs(lateral) > 1e-6:
            abduction[t] = np.degrees(np.arctan2(lateral, -H[1]))
        else:
            abduction[t] = 0.0

        # ── 2. External Rotation (Swing-Twist Decomposition) ──
        forearm_raw = wrist - elbow
        f_norm = np.linalg.norm(forearm_raw)
        if f_norm < 1e-6:
            continue
        F = forearm_raw / f_norm

        # Swing: The minimal rotation axis/angle that swings Y_rest to H
        axis = np.cross(Y_rest, H)
        sin_alpha = np.linalg.norm(axis)
        cos_alpha = cos_el

        if sin_alpha > 1e-6:
            k = axis / sin_alpha
            # Apply Rodrigues' rotation formula to F_rest
            F_ref = (F_rest * cos_alpha +
                     np.cross(k, F_rest) * sin_alpha +
                     k * np.dot(k, F_rest) * (1.0 - cos_alpha))
        else:
            # If no swing, F_ref remains resting. (Covers H == -Y_rest)
            F_ref = F_rest if cos_alpha > 0 else -F_rest

        # Project actual forearm and predicted forearm onto the plane normal to H
        F_proj = F - np.dot(F, H) * H
        F_ref_proj = F_ref - np.dot(F_ref, H) * H

        norm_p = np.linalg.norm(F_proj)
        norm_r = np.linalg.norm(F_ref_proj)

        # Twist happens only if the elbow is meaningfully flexed (F_proj isn't zero)
        if norm_p > 1e-6 and norm_r > 1e-6:
            F_proj = F_proj / norm_p
            F_ref_proj = F_ref_proj / norm_r

            cos_twist = np.clip(np.dot(F_ref_proj, F_proj), -1.0, 1.0)
            cross_twist = np.cross(F_ref_proj, F_proj)
            sin_twist = np.dot(cross_twist, H)

            twist_angle = np.degrees(np.arctan2(sin_twist, cos_twist))

            # Sign correction: Map outward swing to positive External Rotation
            ext_rot[t] = -lat_sign * twist_angle

    return {
        'flexion'     : flexion,
        'abduction'   : abduction,
        'ext_rotation': ext_rot,
    }


# ─────────────────────────────────────────────
#  Step 3b: 3D Arm Reach Space (Convex Hull)
# ─────────────────────────────────────────────

def compute_reach_volume(
    joints_local : np.ndarray,
    side         : str = 'right',
    min_points   : int = 10,
) -> Tuple[Optional[float], Optional[ConvexHull], np.ndarray]:
    """
    Compute 3D arm-reach space volume via Convex Hull on wrist point cloud.

    Args:
        joints_local: (T, 44, 3)  in LCS
        side        : 'right' or 'left'
        min_points  : minimum number of valid frames required

    Returns:
        volume (m³) or None if insufficient points,
        ConvexHull object or None,
        wrist_pts (T', 3) — filtered wrist positions
    """
    _, _, wrist_idx = SIDE_JOINTS[side]
    wrist_pts = joints_local[:, wrist_idx, :]  # (T, 3)
    # Remove frames with NaN
    valid = ~np.any(np.isnan(wrist_pts), axis=1)
    wrist_pts = wrist_pts[valid]

    if len(wrist_pts) < min_points:
        logger.warning(
            "Only %s valid wrist points, need >= %s. Skipping Convex Hull.",
            len(wrist_pts),
            min_points,
        )
        return None, None, wrist_pts

    try:
        hull = ConvexHull(wrist_pts)
        volume = hull.volume
        return volume, hull, wrist_pts
    except QhullError as e:
        logger.warning("ConvexHull failed: %s", e)
        return None, None, wrist_pts


# ─────────────────────────────────────────────
#  Step 3c: Temporal Features and Noise Estimate
# ─────────────────────────────────────────────

def _as_python_number(value):
    """Convert numpy scalars to plain JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs so filtering and derivatives stay defined."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    valid = np.isfinite(x)
    if valid.all():
        return x.copy()
    if not valid.any():
        return np.zeros_like(x)

    idx = np.arange(len(x))
    filled = x.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return filled


def _savgol_window_length(n: int, fps: float, window_sec: float, polyorder: int) -> int:
    """Return a valid odd Savitzky-Golay window length for this sequence."""
    if n <= polyorder + 1:
        return 0

    requested = max(polyorder + 2, int(round(window_sec * fps)))
    if requested % 2 == 0:
        requested += 1

    max_odd = n if n % 2 == 1 else n - 1
    window = min(requested, max_odd)
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window > max_odd:
        return 0
    return window


def _summarize_series(values: np.ndarray) -> Dict:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {'max': None, 'min': None, 'mean': None, 'std': None, 'rms': None, 'max_abs': None}
    return {
        'max': float(np.max(valid)),
        'min': float(np.min(valid)),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'rms': float(np.sqrt(np.mean(valid ** 2))),
        'max_abs': float(np.max(np.abs(valid))),
    }


def _movement_intervals(
    signal: np.ndarray,
    peaks: np.ndarray,
    fps: float,
    baseline: float,
    threshold_fraction: float = 0.20,
) -> Tuple[List[Tuple[int, int]], float]:
    """Find threshold-bounded movement intervals around each detected peak."""
    intervals = []
    n = len(signal)
    for peak_idx in peaks:
        amp = signal[peak_idx] - baseline
        if not np.isfinite(amp) or amp <= 0:
            continue

        threshold = baseline + threshold_fraction * amp
        left = int(peak_idx)
        right = int(peak_idx)
        while left > 0 and signal[left - 1] >= threshold:
            left -= 1
        while right < n - 1 and signal[right + 1] >= threshold:
            right += 1
        intervals.append((left, right))

    if not intervals:
        return [], 0.0

    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    movement_time = sum((end - start + 1) / fps for start, end in merged)
    return merged, float(movement_time)


def compute_temporal_features(
    angles: Dict[str, np.ndarray],
    fps: float = 30.0,
    sg_window_sec: float = 0.33,
    sg_polyorder: int = 3,
    peak_prominence_deg: Optional[float] = None,
) -> Dict:
    """
    Compute smoothed angle, angular velocity/acceleration, peak count,
    movement time, and derivative-noise amplification estimates.

    Noise is estimated from the high-frequency residual:
      residual = raw_angle - Savitzky-Golay-smoothed_angle

    If residual standard deviation is sigma_theta, finite differences amplify it
    approximately by sqrt(2) / dt for velocity and sqrt(6) / dt^2 for acceleration.
    Empirical derivative noise is also reported as raw derivative minus smoothed
    derivative on the same sequence.
    """
    if fps <= 0:
        raise ValueError(f'fps must be positive, got {fps}')

    dt = 1.0 / fps
    result = {
        'fps': float(fps),
        'dt_sec': float(dt),
        'savgol': {
            'window_sec_requested': float(sg_window_sec),
            'polyorder_requested': int(sg_polyorder),
        },
        'angles': {},
    }

    for key in ['flexion', 'abduction', 'ext_rotation']:
        raw = np.asarray(angles[key], dtype=np.float64)
        filled = _fill_nan_linear(raw)
        n = len(filled)
        polyorder = min(int(sg_polyorder), max(1, n - 2))
        window = _savgol_window_length(n, fps, sg_window_sec, polyorder)

        if window > 0:
            smooth = savgol_filter(filled, window_length=window, polyorder=polyorder, mode='interp')
        else:
            smooth = filled.copy()
            window = None

        vel_raw = np.gradient(filled, dt) if n > 1 else np.zeros_like(filled)
        vel_smooth = np.gradient(smooth, dt) if n > 1 else np.zeros_like(smooth)
        acc_raw = np.gradient(vel_raw, dt) if n > 1 else np.zeros_like(vel_raw)
        acc_smooth = np.gradient(vel_smooth, dt) if n > 1 else np.zeros_like(vel_smooth)

        residual = filled - smooth
        sigma_theta = float(np.std(residual[np.isfinite(residual)])) if n > 0 else 0.0
        rms_theta = float(np.sqrt(np.mean(residual ** 2))) if n > 0 else 0.0
        empirical_vel_noise = vel_raw - vel_smooth
        empirical_acc_noise = acc_raw - acc_smooth

        dynamic_range = float(np.nanmax(smooth) - np.nanmin(smooth)) if n > 0 else 0.0
        prominence = peak_prominence_deg
        if prominence is None:
            prominence = max(3.0, 0.10 * dynamic_range)

        baseline = float(np.nanpercentile(smooth, 10)) if n > 0 else 0.0
        peaks, properties = find_peaks(smooth, prominence=prominence) if n > 2 else (np.array([], dtype=int), {})
        intervals, movement_time = _movement_intervals(smooth, peaks, fps, baseline)

        result['angles'][key] = {
            'smoothing': {
                'method': 'Savitzky-Golay',
                'window_length_frames': int(window) if window is not None else None,
                'window_sec_actual': float(window / fps) if window is not None else None,
                'polyorder': int(polyorder),
            },
            'raw_angle_deg': _summarize_series(filled),
            'smoothed_angle_deg': _summarize_series(smooth),
            'angular_velocity_deg_s': _summarize_series(vel_smooth),
            'angular_acceleration_deg_s2': _summarize_series(acc_smooth),
            'peaks': {
                'count': int(len(peaks)),
                'prominence_deg': float(prominence),
                'frames': [int(p) for p in peaks],
                'times_sec': [float(p / fps) for p in peaks],
                'values_deg': [float(smooth[p]) for p in peaks],
                'prominences_deg': [float(x) for x in properties.get('prominences', [])],
            },
            'movement': {
                'threshold_baseline_deg': float(baseline),
                'threshold_fraction_of_peak': 0.20,
                'intervals_frame': [[int(s), int(e)] for s, e in intervals],
                'intervals_sec': [[float(s / fps), float(e / fps)] for s, e in intervals],
                'movement_time_sec': movement_time,
            },
            'noise': {
                'jitter_std_deg': sigma_theta,
                'jitter_rms_deg': rms_theta,
                'theoretical_velocity_noise_std_deg_s': float(np.sqrt(2.0) * sigma_theta / dt),
                'theoretical_acceleration_noise_std_deg_s2': float(np.sqrt(6.0) * sigma_theta / (dt ** 2)),
                'empirical_velocity_noise_std_deg_s': float(np.std(empirical_vel_noise)),
                'empirical_velocity_noise_rms_deg_s': float(np.sqrt(np.mean(empirical_vel_noise ** 2))),
                'empirical_acceleration_noise_std_deg_s2': float(np.std(empirical_acc_noise)),
                'empirical_acceleration_noise_rms_deg_s2': float(np.sqrt(np.mean(empirical_acc_noise ** 2))),
            },
        }

    return result


def visualize_temporal_features(
    angles      : Dict[str, np.ndarray],
    features    : Dict,
    save_path   : Optional[str] = None,
    title       : str = 'Shoulder Temporal Features',
    fps         : float = 30.0,
) -> None:
    """Plot raw/smoothed angles plus smoothed angular velocity and acceleration."""
    T = len(angles['flexion'])
    t_axis = np.arange(T) / fps

    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    configs = [
        ('flexion', 'Flexion', '#E94F37'),
        ('abduction', 'Abduction', '#393E41'),
        ('ext_rotation', 'Ext. Rotation', '#44BBA4'),
    ]

    for row, (key, label, color) in enumerate(configs):
        raw = _fill_nan_linear(np.asarray(angles[key], dtype=np.float64))
        f = features['angles'][key]
        window = f['smoothing']['window_length_frames']
        polyorder = f['smoothing']['polyorder']
        if window is not None:
            smooth = savgol_filter(raw, window_length=window, polyorder=polyorder, mode='interp')
        else:
            smooth = raw
        dt = 1.0 / fps
        vel = np.gradient(smooth, dt) if len(smooth) > 1 else np.zeros_like(smooth)
        acc = np.gradient(vel, dt) if len(vel) > 1 else np.zeros_like(vel)

        ax = axes[row, 0]
        ax.plot(t_axis, raw, color='#999999', lw=0.9, alpha=0.55, label='raw')
        ax.plot(t_axis, smooth, color=color, lw=1.7, label='SG smooth')
        for p in f['peaks']['frames']:
            ax.scatter(t_axis[p], smooth[p], color=color, s=28, zorder=4)
        ax.set_ylabel(f'{label}\n(deg)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        ax = axes[row, 1]
        ax.plot(t_axis, vel, color=color, lw=1.4)
        ax.axhline(0, color='#BBBBBB', lw=0.8)
        ax.set_ylabel('deg/s', fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        ax.plot(t_axis, acc, color=color, lw=1.4)
        ax.axhline(0, color='#BBBBBB', lw=0.8)
        ax.set_ylabel('deg/s2', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title('Angle: raw vs smoothed', fontsize=10)
    axes[0, 1].set_title('Angular velocity', fontsize=10)
    axes[0, 2].set_title('Angular acceleration', fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        logger.info("Temporal feature plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def visualize_angles(
    angles      : Dict[str, np.ndarray],
    save_path   : Optional[str] = None,
    title       : str = 'Shoulder ROM – Joint Angles',
    fps         : float = 30.0,
) -> None:
    """
    Plot Flexion / Abduction / External Rotation over time.

    Args:
        angles   : dict with 'flexion', 'abduction', 'ext_rotation' (T,)
        save_path: output image path; if None, calls plt.show()
        title    : figure title
        fps      : frames per second (for x-axis in seconds)
    """
    T = len(angles['flexion'])
    t_axis = np.arange(T) / fps  # convert to seconds

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    configs = [
        ('flexion',      'Flexion',            '#E94F37', ( -90, 180)),
        ('abduction',    'Abduction',           '#393E41', ( -90, 180)),
        ('ext_rotation', 'External Rotation',   '#44BBA4', (-120, 120)),
    ]

    for ax, (key, label, color, ylim) in zip(axes, configs):
        vals = angles[key]
        ax.plot(t_axis, vals, color=color, linewidth=1.5, label=label)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.fill_between(t_axis, vals, 0, color=color, alpha=0.15)
        ax.set_ylabel(f'{label}\n(degrees)', fontsize=10)
        ax.set_ylim(*ylim)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate max
        valid = ~np.isnan(vals)
        if valid.any():
            max_t = t_axis[np.nanargmax(np.abs(vals[valid]))]
            max_v = np.nanmax(np.abs(vals[valid])) * np.sign(vals[np.nanargmax(np.abs(vals[valid]))])
            ax.annotate(f'max {max_v:.1f}°', xy=(max_t, max_v),
                        fontsize=8, color=color,
                        xytext=(max_t + 0.2, max_v),
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        logger.info("Angle plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


def visualize_reach_space(
    wrist_pts  : np.ndarray,
    hull       : Optional[ConvexHull],
    volume     : Optional[float] = None,
    save_path  : Optional[str] = None,
    title      : str = '3D Arm Reach Space',
) -> None:
    """
    Plot 3D wrist point cloud and Convex Hull mesh.

    Args:
        wrist_pts : (T, 3)  wrist positions in LCS
        hull      : scipy ConvexHull object (or None)
        volume    : hull volume in m³ for annotation
        save_path : output image path; if None, calls plt.show()
        title     : figure title
    """
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Scatter wrist points
    ax.scatter(wrist_pts[:, 0], wrist_pts[:, 2], wrist_pts[:, 1],
               c=np.arange(len(wrist_pts)), cmap='plasma',
               s=8, alpha=0.6, label='Wrist trajectory')

    # Draw Convex Hull facets
    if hull is not None:
        hull_pts = wrist_pts[hull.vertices]
        verts = [wrist_pts[simplex][:, [0, 2, 1]] for simplex in hull.simplices]
        poly  = Poly3DCollection(verts, alpha=0.15,
                                  facecolor='#44BBA4', edgecolor='#2d8272', linewidth=0.3)
        ax.add_collection3d(poly)

        if volume is not None:
            vol_cm3 = volume * 1e6  # m³ → cm³
            ax.text2D(0.02, 0.95,
                      f'Volume: {vol_cm3:.1f} cm³\n({volume*1e3:.4f} dm³)',
                      transform=ax.transAxes, fontsize=11,
                      color='#44BBA4', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Enforce equal aspect ratio
    if len(wrist_pts) > 0:
        valid_pts = wrist_pts[~np.any(np.isnan(wrist_pts), axis=1)]
        if len(valid_pts) > 0:
            hm = 1.0
            ax.set_xlim(-hm, hm)
            ax.set_ylim(-hm, hm)
            ax.set_zlim(-hm, hm)

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    # Adhere to same perspective as combined view
    ax.view_init(elev=20, azim=110)

    # Axis labels (LCS convention)
    ax.set_xlabel('X  (right-lateral, m)', fontsize=9)
    ax.set_ylabel('Z  (anterior, m)',       fontsize=9)
    ax.set_zlabel('Y  (superior, m)',       fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        logger.info("Reach space plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()


def print_summary(
    angles : Dict[str, np.ndarray],
    volume : Optional[float],
    side   : str,
) -> Dict:
    """Print and return a summary dict of ROM statistics."""
    stats = {'side': side}
    logger.info("%s", "-" * 50)
    logger.info("Shoulder ROM Analysis Summary (%s side)", side.upper())
    logger.info("%s", "-" * 50)
    for key, label in [('flexion','Flexion'), ('abduction','Abduction'), ('ext_rotation','Ext.Rotation')]:
        vals = angles[key]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            logger.info("%-20s: no data", label)
            stats[key] = {'max': None, 'mean': None}
        else:
            mx   = float(np.max(valid))
            mn   = float(np.min(valid))
            mean = float(np.mean(valid))
            logger.info("%-20s: max=%+6.1f deg  min=%+6.1f deg  mean=%+6.1f deg", label, mx, mn, mean)
            stats[key] = {'max': mx, 'min': mn, 'mean': mean}

    if volume is not None:
        vol_cm3 = volume * 1e6
        logger.info("%-20s: %.1f cm^3  (%.4f dm^3)", "3D Reach Volume", vol_cm3, volume * 1e3)
        stats['reach_volume_cm3'] = vol_cm3
    else:
        stats['reach_volume_cm3'] = None
    logger.info("%s", "-" * 50)
    return stats


# ─────────────────────────────────────────────
#  Skeleton + Reach Space Combined Visualization
# ─────────────────────────────────────────────

# Skeleton bone connections: (joint_a, joint_b, hex_color)
# Blue family = right arm,  Red family = left arm,  Gray = torso/pelvis/legs
SKELETON_BONES = [
    (9,  8,  '#888888'),  # R.Hip   → MidHip
    (12, 8,  '#888888'),  # L.Hip   → MidHip
    (8,  1,  '#888888'),  # MidHip  → Neck
    (1,  2,  '#2E86AB'),  # Neck       → R.Shoulder
    (2,  3,  '#2E86AB'),  # R.Shoulder → R.Elbow
    (3,  4,  '#2E86AB'),  # R.Elbow    → R.Wrist
    (9,  10, '#888888'),  # R.Hip      → R.Knee
    (10, 11, '#888888'),  # R.Knee     → R.Ankle
    (1,  5,  '#E84855'),  # Neck       → L.Shoulder
    (5,  6,  '#E84855'),  # L.Shoulder → L.Elbow
    (6,  7,  '#E84855'),  # L.Elbow    → L.Wrist
    (12, 13, '#888888'),  # L.Hip      → L.Knee
    (13, 14, '#888888'),  # L.Knee     → L.Ankle
]

SIDE_BONE_COLOR = {'right': '#2E86AB', 'left': '#E84855'}


def _draw_bones_3d(ax, joints, bones, alpha=0.8, lw=2.0, jnt_size=25):
    """
    Draw upper-body skeleton on 3D axes.
    Matplotlib 3D uses (X, Y_screen, Z_screen); we plot as (X, Z_lcs, Y_lcs)
    so that the superior (Y_lcs) axis points upward on screen.
    """
    drawn = set()
    for j_a, j_b, color in bones:
        p = joints[[j_a, j_b]]                          # (2, 3): [X, Y_lcs, Z_lcs]
        ax.plot(p[:, 0], p[:, 2], p[:, 1],              # X, Z→screen-Y, Y→screen-Z
                color=color, alpha=alpha, linewidth=lw,
                solid_capstyle='round', zorder=3)
        if jnt_size > 0:
            for ji in [j_a, j_b]:
                if ji not in drawn:
                    ax.scatter(joints[ji, 0], joints[ji, 2], joints[ji, 1],
                               color=color, s=jnt_size, alpha=alpha,
                               zorder=5, depthshade=False)
                    drawn.add(ji)


def _draw_bones_2d(ax, joints, bones, dims=(0, 1), alpha=0.8, lw=2.0, jnt_size=20):
    """
    Draw upper-body skeleton on a 2D axes.
    dims = (d0, d1): column indices into the (X, Y, Z) LCS array.
    """
    d0, d1 = dims
    drawn = set()
    for j_a, j_b, color in bones:
        p = joints[[j_a, j_b]]
        ax.plot(p[:, d0], p[:, d1], color=color, alpha=alpha,
                linewidth=lw, solid_capstyle='round', zorder=3)
        if jnt_size > 0:
            for ji in [j_a, j_b]:
                if ji not in drawn:
                    ax.scatter(joints[ji, d0], joints[ji, d1],
                               color=color, s=jnt_size, alpha=alpha, zorder=5)
                    drawn.add(ji)


def _hull_2d_overlay(ax, pts: np.ndarray, color: str, d0: int, d1: int):
    """Project wrist point cloud onto a 2D plane and overlay its convex hull."""
    pts2d = pts[:, [d0, d1]]
    try:
        h = ConvexHull(pts2d)
        verts = np.vstack([pts2d[h.vertices], pts2d[h.vertices[0]]])
        ax.fill(verts[:, 0], verts[:, 1], alpha=0.13, color=color, zorder=2)
        ax.plot(verts[:, 0], verts[:, 1], color=color, linewidth=1.8,
                alpha=0.75, zorder=4)
    except (QhullError, Exception):
        pass


def visualize_skeleton_with_reach(
    joints_local : np.ndarray,          # (T, 44, 3) in LCS
    results_r    : Tuple,               # (volume_r, hull_r, wrist_pts_r)
    results_l    : Tuple,               # (volume_l, hull_l, wrist_pts_l)
    save_path    : Optional[str] = None,
    title        : str = 'Upper Body Skeleton + Arm Reach Space (LCS)',
    stride       : Optional[int] = None,
) -> None:
    """
    Four-panel combined visualization in LCS (upper-body corrected space):

      ┌──────────────────┬──────────┬──────────┐
      │                  │  Front   │  Side    │
      │   3D View        │  (X–Y)   │  (Z–Y)   │
      │  skeleton +      ├──────────┴──────────┤
      │  reach hulls     │    Top  (X–Z)       │
      └──────────────────┴─────────────────────┘

    LCS axes:  X = right-lateral,  Y = superior (up),  Z = anterior

    Ghost skeletons (low alpha) show all sampled frames; the mean skeleton
    is drawn at full opacity so the resting posture is clearly visible.
    """
    T = len(joints_local)
    if stride is None:
        stride = max(1, T // 40)   # show at most ~40 ghost skeleton frames

    volume_r, hull_r, wrist_pts_r = results_r
    volume_l, hull_l, wrist_pts_l = results_l
    mean_joints = np.nanmean(joints_local, axis=0)   # (44, 3) mean posture

    # ── Figure & layout ───────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.40)
    ax_3d    = fig.add_subplot(gs[:, :2], projection='3d')
    ax_front = fig.add_subplot(gs[0, 2])
    ax_side  = fig.add_subplot(gs[0, 3])
    ax_top   = fig.add_subplot(gs[1, 2:])

    # ── 3D VIEW ───────────────────────────────────────────────────────
    ax_3d.set_facecolor('#FAFAFA')
    ax_3d.set_title('3D View  (LCS — upper-body corrected)', fontsize=10)

    # Ghost skeleton traces (every `stride` frames)
    for t in range(0, T, stride):
        _draw_bones_3d(ax_3d, joints_local[t], SKELETON_BONES,
                       alpha=0.05, lw=0.9, jnt_size=0)

    # Mean skeleton (reference posture, full opacity)
    _draw_bones_3d(ax_3d, mean_joints, SKELETON_BONES,
                   alpha=0.92, lw=2.8, jnt_size=40)

    # Wrist point clouds + convex hulls
    for wrist_pts, hull, color, cmap_name, label in [
        (wrist_pts_r, hull_r, '#2E86AB', 'Blues',   'Right Arm'),
        (wrist_pts_l, hull_l, '#E84855', 'Oranges', 'Left Arm'),
    ]:
        if len(wrist_pts) == 0:
            continue

        # Add proxy artist for clear legend (circular marker)
        ax_3d.plot([], [], [], marker='o', color=color, linestyle='None',
                   markersize=8, label=label)

        ax_3d.scatter(wrist_pts[:, 0], wrist_pts[:, 2], wrist_pts[:, 1],
                      c=np.arange(len(wrist_pts)), cmap=cmap_name,
                      s=9, alpha=0.55, zorder=2)
        if hull is not None:
            verts3d = [wrist_pts[s][:, [0, 2, 1]] for s in hull.simplices]
            ax_3d.add_collection3d(Poly3DCollection(
                verts3d, alpha=0.10, facecolor=color,
                edgecolor=color, linewidth=0.15))

    # Volume annotation box
    ann = []
    if volume_r is not None:
        ann.append(f'Right: {volume_r*1e6:.1f} cm³')
    if volume_l is not None:
        ann.append(f'Left : {volume_l*1e6:.1f} cm³')
    if ann:
        ax_3d.text2D(0.02, 0.97, '\n'.join(ann),
                     transform=ax_3d.transAxes, fontsize=10,
                     fontweight='bold', va='top',
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='white', alpha=0.85, edgecolor='#CCCCCC'))

    # Symmetric axis limits: each axis is centered at 0
    # ("+0.5 ⇒ -0.5" box so left/right appear equal)
    # Note: we plot as ax.plot(LCS_X, LCS_Z, LCS_Y), so:
    #   matplotlib x-axis ≈ LCS X (lateral)
    #   matplotlib y-axis ≈ LCS Z (anterior)
    #   matplotlib z-axis ≈ LCS Y (superior)
    _pts_all = [joints_local.reshape(-1, 3)]
    if len(wrist_pts_r) > 0: _pts_all.append(wrist_pts_r)
    if len(wrist_pts_l) > 0: _pts_all.append(wrist_pts_l)
    _clean = np.concatenate(_pts_all, axis=0)
    _clean = _clean[~np.any(np.isnan(_clean), axis=1)]
    # Fixed box limits for consistent scale
    hm = 1.0
    ax_3d.set_xlim(-hm, hm)
    ax_3d.set_ylim(-hm, hm)
    ax_3d.set_zlim(-hm, hm)

    try:
        ax_3d.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    # Adjust 3D camera so person faces the viewer
    ax_3d.view_init(elev=20, azim=110)

    ax_3d.set_xlabel('X  (right-lateral, m)', fontsize=8)
    ax_3d.set_ylabel('Z  (anterior, m)',       fontsize=8)
    ax_3d.set_zlabel('Y  (superior, m)',       fontsize=8)
    ax_3d.legend(fontsize=8, loc='upper right', framealpha=0.8)

    # ── 2D PROJECTION VIEWS ───────────────────────────────────────────
    view_cfgs = [
        # (axes,    dims,   title,                              xlabel,                  ylabel,    invert_x)
        (ax_front, (0, 1), 'Front View\n(X: lateral ↔  Y: superior ↕)',
         'X  right-lateral (m)', 'Y  superior (m)', True),
        (ax_side,  (2, 1), 'Side View\n(Z: anterior ↔  Y: superior ↕)',
         'Z  anterior (m)',      'Y  superior (m)', True),
        (ax_top,   (2, 0), 'Top View\n(Z: anterior ↔  X: lateral ↕)',
         'Z  anterior (m)',     'X  right-lateral (m)', True),
    ]

    for ax2d, dims, t2d, xl, yl, inv_x in view_cfgs:
        ax2d.set_facecolor('#FAFAFA')
        ax2d.set_title(t2d, fontsize=9)
        ax2d.set_xlabel(xl, fontsize=8)
        ax2d.set_ylabel(yl, fontsize=8)
        ax2d.grid(True, alpha=0.30, linewidth=0.6)

        ax2d.set_aspect('equal', adjustable='box')
        ax2d.set_xlim(-hm, hm)
        ax2d.set_ylim(-hm, hm)

        if inv_x:
            ax2d.invert_xaxis()

        d0, d1 = dims

        # Ghost skeleton traces
        for t in range(0, T, stride):
            _draw_bones_2d(ax2d, joints_local[t], SKELETON_BONES,
                           dims=dims, alpha=0.04, lw=0.7, jnt_size=0)

        # Mean skeleton
        _draw_bones_2d(ax2d, mean_joints, SKELETON_BONES,
                       dims=dims, alpha=0.90, lw=2.0, jnt_size=22)

        # Wrist clouds + 2D hull projections
        for wrist_pts, hull, color, cmap_name, label in [
            (wrist_pts_r, hull_r, '#2E86AB', 'Blues',   'Right Arm'),
            (wrist_pts_l, hull_l, '#E84855', 'Oranges', 'Left Arm'),
        ]:
            if len(wrist_pts) == 0:
                continue

            # Add proxy artist for clear legend (circular marker)
            ax2d.plot([], [], marker='o', color=color, linestyle='None',
                      markersize=8, label=label)

            ax2d.scatter(wrist_pts[:, d0], wrist_pts[:, d1],
                         c=np.arange(len(wrist_pts)), cmap=cmap_name,
                         s=5, alpha=0.50, zorder=3)

            # Draw 2D convex hull ONLY if we computed a 3D hull (i.e. video mode)
            if hull is not None:
                _hull_2d_overlay(ax2d, wrist_pts, color, d0, d1)

        ax2d.legend(fontsize=7, loc='best', framealpha=0.8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        logger.info("Combined skeleton and reach plot saved to %s", save_path)
    else:
        plt.show()
    plt.close()



# ─────────────────────────────────────────────
#  Video Renderer  (for .npy temporal data)
# ─────────────────────────────────────────────

def render_video(
    joints_local : np.ndarray,
    angles_r     : Optional[Dict],
    angles_l     : Optional[Dict],
    save_path    : str,
    fps          : float = 30.0,
    title        : str  = 'Shoulder ROM Analysis',
    stride       : int  = 1,
) -> None:
    """
    Render a per-frame animation of LCS skeleton + angle time-series
    and save as MP4 (falls back to GIF if FFMpeg unavailable).

    Layout:
      Left  — 3D skeleton (current frame) + accumulated wrist trail
      Right — Flexion / Abduction / Ext.Rotation time-series with cursor
    """
    import matplotlib.animation as mpl_anim

    T      = len(joints_local)
    frames = list(range(0, T, stride))
    t_axis = np.arange(T) / fps

    wr_r = joints_local[:, SIDE_JOINTS['right'][2], :]
    wr_l = joints_local[:, SIDE_JOINTS['left'][2],  :]

    _clean = joints_local.reshape(-1, 3)
    _clean = _clean[~np.any(np.isnan(_clean), axis=1)]
    hx = float(np.nanmax(np.abs(_clean[:, 0]))) * 1.15
    hz = float(np.nanmax(np.abs(_clean[:, 2]))) * 1.15
    hy = float(np.nanmax(np.abs(_clean[:, 1]))) * 1.15
    hm = max(hx, hz, hy)

    BG, FG = '#F8F9FA', '#111111'
    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    fig.suptitle(title, color=FG, fontsize=13, fontweight='bold', y=0.97)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.50, wspace=0.30,
                           left=0.05, right=0.97, top=0.92, bottom=0.07)
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_fl = fig.add_subplot(gs[0, 1])
    ax_ab = fig.add_subplot(gs[1, 1])
    ax_er = fig.add_subplot(gs[2, 1])

    _ANGLE_CFG = [
        (ax_fl, 'flexion',      'Flexion (deg)',        '#E94F37', ( -90, 180)),
        (ax_ab, 'abduction',    'Abduction (deg)',      '#44BBA4', ( -90, 180)),
        (ax_er, 'ext_rotation', 'Ext. Rotation (deg)', '#F7C59F', (-120, 120)),
    ]
    t_max = t_axis[-1] if len(t_axis) > 1 else 1.0
    v_lines = []
    for ax2, key, ylabel, col, ylim in _ANGLE_CFG:
        ax2.set_facecolor('#FAFAFA')
        ax2.set_xlim(0, t_max)
        ax2.set_ylim(*ylim)
        ax2.set_ylabel(ylabel, color=FG, fontsize=8)
        ax2.axhline(0, color='#BBBBBB', lw=0.8)
        ax2.grid(True, alpha=0.50, color='#DDDDDD')
        for sp in ax2.spines.values():
            sp.set_color('#CCCCCC')
        ax2.tick_params(colors='#444444', labelsize=7)
        if angles_r and key in angles_r:
            ax2.plot(t_axis, angles_r[key], color=col, lw=1.5, alpha=0.95, label='R', zorder=3)
        if angles_l and key in angles_l:
            ax2.plot(t_axis, angles_l[key], color=col, lw=1.5, alpha=0.50, linestyle='--', label='L', zorder=3)
        ax2.legend(fontsize=7, loc='upper right', facecolor=BG, edgecolor='#CCCCCC', labelcolor=FG)
        vl = ax2.axvline(0, color='#333333', lw=1.4, alpha=0.9, zorder=5)
        v_lines.append(vl)
    ax_er.set_xlabel('Time (s)', color=FG, fontsize=8)

    def _update(fi):
        t = frames[fi]
        ax_3d.cla()
        ax_3d.set_facecolor('#FAFAFA')
        ax_3d.view_init(elev=20, azim=110)
        # Fixed box limits for consistent scale
        hm = 1.0
        ax_3d.set_xlim(-hm, hm);  ax_3d.set_ylim(-hm, hm);  ax_3d.set_zlim(-hm, hm)
        try:
            ax_3d.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
        ax_3d.set_xlabel('X (lateral)',  color='#444', fontsize=7)
        ax_3d.set_ylabel('Z (anterior)', color='#444', fontsize=7)
        ax_3d.set_zlabel('Y (superior)', color='#444', fontsize=7)
        ax_3d.tick_params(colors='#444', labelsize=6)
        _draw_bones_3d(ax_3d, joints_local[t], SKELETON_BONES, alpha=0.95, lw=2.5, jnt_size=38)
        for trail, cmap_name in [(wr_r, 'Blues'), (wr_l, 'Oranges')]:
            seg = trail[:t+1]
            valid = ~np.any(np.isnan(seg), axis=1)
            seg = seg[valid]
            if len(seg) > 0:
                ax_3d.scatter(seg[:, 0], seg[:, 2], seg[:, 1],
                              c=np.arange(len(seg)), cmap=cmap_name,
                              s=9, alpha=0.60, zorder=2, depthshade=False)
        t_s = t / fps
        lines = [f'Frame {t+1:3d}/{T}  |  {t_s:.2f}s']
        for key, lbl in [('flexion','Flex'), ('abduction','Abd'), ('ext_rotation','ER')]:
            rv = angles_r[key][t] if (angles_r and key in angles_r) else float('nan')
            lv = angles_l[key][t] if (angles_l and key in angles_l) else float('nan')
            lines.append(f'R.{lbl}: {rv:+.1f} deg   L.{lbl}: {lv:+.1f} deg')
        ax_3d.text2D(0.02, 0.97, '\n'.join(lines),
                     transform=ax_3d.transAxes, color='black', fontsize=8,
                     va='top', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, alpha=0.75, edgecolor='#CCCCCC'))
        for vl in v_lines:
            vl.set_xdata([t_s, t_s])
        return []

    logger.info("Rendering %s frames (fps=%.0f, stride=%s)", len(frames), fps, stride)
    ani = mpl_anim.FuncAnimation(fig, _update, frames=len(frames), interval=int(1000/fps), blit=False)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Save MP4 ──
    try:
        writer = mpl_anim.FFMpegWriter(fps=fps/stride, bitrate=15000,
                                        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=1920:-2'])
        ani.save(save_path, writer=writer, dpi=600, savefig_kwargs={'facecolor': BG})
        logger.info("MP4 video saved to %s", save_path)
    except Exception as e:
        logger.warning("FFMpeg failed (%s), skipping MP4 generation.", e)



    plt.close()
