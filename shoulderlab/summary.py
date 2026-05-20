"""Summarize shoulder temporal feature outputs."""

import csv
import json
from pathlib import Path

from shoulderlab.log import get_logger
from shoulderlab.paths import DATA_OUTPUTS, SHOULDERLAB_ROOT


logger = get_logger()


ANGLE_LABELS = {
    'flexion': 'Flexion',
    'abduction': 'Abduction',
    'ext_rotation': 'Ext. Rotation',
}


def _num(value, ndigits=2):
    if value is None:
        return ''
    return round(float(value), ndigits)


def _fmt(value, ndigits=2, suffix=''):
    if value is None:
        return '-'
    return f'{float(value):.{ndigits}f}{suffix}'


def _movement_name(stem):
    if stem.startswith('HSMR-'):
        stem = stem[5:]
    return stem.replace('_', ' ')


def collect_rows(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob('*_results.json')):
        with path.open() as f:
            data = json.load(f)

        stem = path.name.removesuffix('_results.json')
        if stem.endswith('_right'):
            movement = stem[:-6]
            side = 'right'
        elif stem.endswith('_left'):
            movement = stem[:-5]
            side = 'left'
        else:
            movement = stem
            side = data.get('side', '')

        temporal = data.get('temporal_features', {})
        for angle_key, angle_data in temporal.get('angles', {}).items():
            rows.append({
                'movement_id': movement,
                'movement': _movement_name(movement),
                'side': side,
                'angle': angle_key,
                'angle_label': ANGLE_LABELS.get(angle_key, angle_key),
                'frames': data.get('frames'),
                'fps': temporal.get('fps'),
                'sg_window_frames': angle_data['smoothing'].get('window_length_frames'),
                'sg_window_sec': angle_data['smoothing'].get('window_sec_actual'),
                'angle_min_deg': angle_data['smoothed_angle_deg'].get('min'),
                'angle_max_deg': angle_data['smoothed_angle_deg'].get('max'),
                'angle_rom_deg': (
                    angle_data['smoothed_angle_deg'].get('max') -
                    angle_data['smoothed_angle_deg'].get('min')
                ),
                'angular_velocity_max_abs_deg_s': angle_data['angular_velocity_deg_s'].get('max_abs'),
                'angular_acceleration_max_abs_deg_s2': angle_data['angular_acceleration_deg_s2'].get('max_abs'),
                'peak_count': angle_data['peaks'].get('count'),
                'movement_time_sec': angle_data['movement'].get('movement_time_sec'),
                'jitter_std_deg': angle_data['noise'].get('jitter_std_deg'),
                'jitter_rms_deg': angle_data['noise'].get('jitter_rms_deg'),
                'theoretical_velocity_noise_std_deg_s': angle_data['noise'].get('theoretical_velocity_noise_std_deg_s'),
                'theoretical_acceleration_noise_std_deg_s2': angle_data['noise'].get('theoretical_acceleration_noise_std_deg_s2'),
                'empirical_velocity_noise_std_deg_s': angle_data['noise'].get('empirical_velocity_noise_std_deg_s'),
                'empirical_acceleration_noise_std_deg_s2': angle_data['noise'].get('empirical_acceleration_noise_std_deg_s2'),
                'source_json': str(path),
            })
    return rows


def write_csv(rows, path: Path):
    fieldnames = [
        'movement_id', 'movement', 'side', 'angle', 'angle_label', 'frames', 'fps',
        'sg_window_frames', 'sg_window_sec',
        'angle_min_deg', 'angle_max_deg', 'angle_rom_deg',
        'angular_velocity_max_abs_deg_s', 'angular_acceleration_max_abs_deg_s2',
        'peak_count', 'movement_time_sec',
        'jitter_std_deg', 'jitter_rms_deg',
        'theoretical_velocity_noise_std_deg_s',
        'theoretical_acceleration_noise_std_deg_s2',
        'empirical_velocity_noise_std_deg_s',
        'empirical_acceleration_noise_std_deg_s2',
        'source_json',
    ]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _num(row.get(k), 6) if isinstance(row.get(k), float) else row.get(k) for k in fieldnames})


def _dominant_rows(rows):
    by_key = {}
    for row in rows:
        key = (row['movement_id'], row['side'])
        current = by_key.get(key)
        if current is None or abs(row['angle_rom_deg']) > abs(current['angle_rom_deg']):
            by_key[key] = row
    return [by_key[k] for k in sorted(by_key)]


def write_markdown(rows, path: Path, output_dir: Path):
    dominant = _dominant_rows(rows)
    right_dominant = [r for r in dominant if r['side'] == 'right']

    all_jitter = [r['jitter_std_deg'] for r in rows if r['jitter_std_deg'] is not None]
    all_vel_noise = [r['empirical_velocity_noise_std_deg_s'] for r in rows if r['empirical_velocity_noise_std_deg_s'] is not None]
    all_acc_noise = [r['empirical_acceleration_noise_std_deg_s2'] for r in rows if r['empirical_acceleration_noise_std_deg_s2'] is not None]

    lines = []
    lines.append('# Temporal Feature and Noise Amplification Report')
    lines.append('')
    lines.append('## Data and Method')
    lines.append('')
    lines.append('- Base data: per-frame 3D joint coordinate time series recovered from HSMR output `.npy` files.')
    lines.append('- Shoulder angles: flexion, abduction, and external rotation computed in the torso-compensated local coordinate system.')
    lines.append('- Smoothing: Savitzky-Golay filter before temporal derivatives.')
    lines.append('- Feature extraction: angular velocity, angular acceleration, peak count, peak-bounded movement time.')
    lines.append('- Noise estimate: angle jitter is `raw angle - smoothed angle`; finite-difference amplification is reported both theoretically and empirically.')
    lines.append('')
    lines.append('## Smoothing Parameters')
    lines.append('')
    first = rows[0] if rows else {}
    lines.append(f"- FPS: {_fmt(first.get('fps'), 1)}")
    lines.append(f"- Savitzky-Golay window: {_fmt(first.get('sg_window_sec'), 2, ' s')} ({first.get('sg_window_frames', '-')} frames)")
    lines.append('- Savitzky-Golay polynomial order: 3')
    lines.append('')
    lines.append('## Overall Noise Scale')
    lines.append('')
    lines.append(f"- Angle jitter std range: {_fmt(min(all_jitter), 3, ' deg')} to {_fmt(max(all_jitter), 3, ' deg')}")
    lines.append(f"- Empirical angular-velocity noise std range: {_fmt(min(all_vel_noise), 2, ' deg/s')} to {_fmt(max(all_vel_noise), 2, ' deg/s')}")
    lines.append(f"- Empirical angular-acceleration noise std range: {_fmt(min(all_acc_noise), 2, ' deg/s^2')} to {_fmt(max(all_acc_noise), 2, ' deg/s^2')}")
    lines.append('')
    lines.append('The acceleration noise scale is much larger than angle noise because the second derivative divides by `dt^2`. This supports applying Savitzky-Golay smoothing before derivative features.')
    lines.append('')
    lines.append('## Dominant-Axis Summary: Right Shoulder')
    lines.append('')
    lines.append('| Movement | Dominant angle | ROM (deg) | Max | Min | Max abs acc (deg/s^2) | Peaks | Movement time (s) | Jitter std (deg) | Empirical vel noise std (deg/s) | Empirical acc noise std (deg/s^2) |')
    lines.append('|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|')
    for row in right_dominant:
        lines.append(
            f"| {row['movement']} | {row['angle_label']} | "
            f"{_fmt(row['angle_rom_deg'], 1)} | {_fmt(row['angle_max_deg'], 1)} | {_fmt(row['angle_min_deg'], 1)} | "
            f"{_fmt(row['angular_acceleration_max_abs_deg_s2'], 1)} | {row['peak_count']} | "
            f"{_fmt(row['movement_time_sec'], 2)} | {_fmt(row['jitter_std_deg'], 3)} | "
            f"{_fmt(row['empirical_velocity_noise_std_deg_s'], 2)} | {_fmt(row['empirical_acceleration_noise_std_deg_s2'], 2)} |"
        )
    lines.append('')
    lines.append('## Full Outputs')
    lines.append('')
    lines.append(f"- CSV summary: `{output_dir / 'temporal_feature_summary.csv'}`")
    lines.append(f"- Per-sample JSON and plots: `{output_dir}`")
    lines.append('- Each `*_results.json` contains `temporal_features` with smoothed angles, angular velocity, angular acceleration, peak data, movement intervals, and noise estimates.')
    lines.append('')
    lines.append('## Suggested Answer')
    lines.append('')
    lines.append('실제 HSMR 3D joint coordinate 시계열에서 어깨각도 3종을 계산한 뒤, Savitzky-Golay smoothing을 적용해 각속도와 각가속도를 산출했습니다. 미분 전 노이즈는 raw angle과 smoothed angle의 차이로 추정했으며, 30 fps 기준 1차 미분은 대략 `sqrt(2)/dt`, 2차 미분은 `sqrt(6)/dt^2`만큼 노이즈가 커질 수 있습니다. 실제 데이터에서도 각도 jitter는 deg 단위로 작아도 각가속도 노이즈는 deg/s^2 단위로 크게 증가하므로, derivative feature를 쓰기 전 smoothing 전처리가 필요하다는 결론입니다.')
    lines.append('')
    path.write_text('\n'.join(lines) + '\n')


def summarize_analysis(
    input_dir: Path = DATA_OUTPUTS / 'UUCM' / 'analysis',
    docs_path: Path = SHOULDERLAB_ROOT / 'docs' / 'Temporal_Feature_Noise_Report.md',
) -> dict:
    """Write CSV and Markdown summaries for `*_results.json` files."""
    input_dir = Path(input_dir)
    docs_path = Path(docs_path)

    rows = collect_rows(input_dir)
    if not rows:
        raise SystemExit(f'No *_results.json files found in {input_dir}')

    csv_path = input_dir / 'temporal_feature_summary.csv'
    md_path = input_dir / 'Temporal_Feature_Noise_Report.md'
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, input_dir)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, docs_path, input_dir)
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", docs_path)
    return {
        'rows': len(rows),
        'csv_path': csv_path,
        'markdown_path': md_path,
        'docs_path': docs_path,
    }
