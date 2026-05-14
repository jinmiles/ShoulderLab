# Q2 Temporal Feature and Noise Amplification Report

## Data and Method

- Base data: per-frame 3D joint coordinate time series recovered from HSMR output `.npy` files.
- Shoulder angles: flexion, abduction, and external rotation computed in the torso-compensated local coordinate system.
- Smoothing: Savitzky-Golay filter before temporal derivatives.
- Feature extraction: angular velocity, angular acceleration, peak count, peak-bounded movement time.
- Noise estimate: angle jitter is `raw angle - smoothed angle`; finite-difference amplification is reported both theoretically and empirically.

## Smoothing Parameters

- FPS: 30.0
- Savitzky-Golay window: 0.37 s (11 frames)
- Savitzky-Golay polynomial order: 3

## Overall Noise Scale

- Angle jitter std range: 0.051 deg to 3.708 deg
- Empirical angular-velocity noise std range: 1.28 deg/s to 92.50 deg/s
- Empirical angular-acceleration noise std range: 33.71 deg/s^2 to 2537.35 deg/s^2

The acceleration noise scale is much larger than angle noise because the second derivative divides by `dt^2`. This supports applying Savitzky-Golay smoothing before derivative features.

## Dominant-Axis Summary: Right Shoulder

| Movement | Dominant angle | ROM (deg) | Max | Min | Max abs acc (deg/s^2) | Peaks | Movement time (s) | Jitter std (deg) | Empirical vel noise std (deg/s) | Empirical acc noise std (deg/s^2) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 001 Flexion | Flexion | 143.4 | 127.0 | -16.4 | 2093.8 | 1 | 1.60 | 1.165 | 27.27 | 696.12 |
| 002 Abduction | Abduction | 123.9 | 130.1 | 6.3 | 1952.5 | 2 | 1.90 | 1.338 | 30.71 | 781.40 |
| 003 Internal Rotation | Ext. Rotation | 51.7 | 10.8 | -40.9 | 504.5 | 1 | 1.90 | 0.337 | 7.93 | 205.83 |
| 004 External Rotation | Ext. Rotation | 46.5 | 41.1 | -5.5 | 479.6 | 2 | 3.30 | 0.282 | 6.92 | 184.06 |
| 005 Internal Rotation Abduction | Flexion | 31.2 | 22.2 | -9.0 | 1359.2 | 2 | 2.17 | 1.326 | 39.10 | 1066.35 |
| 006 External Rotation Abduction | Ext. Rotation | 48.5 | 41.3 | -7.2 | 783.5 | 2 | 4.17 | 0.683 | 16.86 | 444.93 |
| 007 Circumduction | Flexion | 169.3 | 122.4 | -46.9 | 4063.4 | 2 | 1.97 | 3.220 | 82.30 | 2282.34 |
| 008 Guide | Abduction | 124.3 | 103.7 | -20.7 | 6056.4 | 17 | 23.03 | 2.882 | 72.55 | 2010.76 |

## Full Outputs

- CSV summary: `/home/user/extra_workdir/ShoulderLab/data_outputs/UUCM/q2_analysis/q2_temporal_feature_summary.csv`
- Per-sample JSON and plots: `/home/user/extra_workdir/ShoulderLab/data_outputs/UUCM/q2_analysis`
- Each `*_results.json` contains `temporal_features` with smoothed angles, angular velocity, angular acceleration, peak data, movement intervals, and noise estimates.

## Suggested Q2 Answer

실제 HSMR 3D joint coordinate 시계열에서 어깨각도 3종을 계산한 뒤, Savitzky-Golay smoothing을 적용해 각속도와 각가속도를 산출했습니다. 미분 전 노이즈는 raw angle과 smoothed angle의 차이로 추정했으며, 30 fps 기준 1차 미분은 대략 `sqrt(2)/dt`, 2차 미분은 `sqrt(6)/dt^2`만큼 노이즈가 커질 수 있습니다. 실제 데이터에서도 각도 jitter는 deg 단위로 작아도 각가속도 노이즈는 deg/s^2 단위로 크게 증가하므로, derivative feature를 쓰기 전 smoothing 전처리가 필요하다는 결론입니다.
