# Shoulder Multiview Pi3-HSMR Pipeline Plan

이 문서는 `data_inputs/shoulder`의 synchronized multiview shoulder 영상에
대해 Pi3/Pi3X와 HSMR을 함께 사용해 3D human pose 및 shoulder ROM 정확도를
높이는 파이프라인 계획을 정리한다.

## 핵심 아이디어

입력 trial 하나는 다음 세 개의 동기화된 카메라 영상으로 구성된다.

```text
data_inputs/shoulder/subjectXX/NNN_movement_name/
|-- cam_a.mp4  # left view
|-- cam_b.mp4  # right view
`-- cam_c.mp4  # center view
```

전체 방향은 다음과 같다.

```text
cam_a/b/c synchronized videos
-> Pi3/Pi3X camera pose estimation
-> HSMR per camera
-> recover SKEL joints per camera
-> transform per-camera human pose into a shared coordinate system
-> compare cam_a, cam_b, cam_c, and multiview-fused pose
-> run ShoulderLab ROM analysis
```

Pi3/Pi3X는 카메라 기하 추정기 역할을 하고, HSMR은 각 카메라 영상에서
human pose sequence를 추정하는 역할을 한다. ShoulderLab은 두 결과를 공통
좌표계로 정렬하고, 단일뷰와 멀티뷰 결과를 비교 및 fusion하는 분석 레이어를
담당한다.

## 중요한 전제와 주의점

- `cam_a`, `cam_b`, `cam_c`는 같은 trial 안에서 frame-level sync가 맞아 있다.
- 카메라 포즈는 대부분 거의 일정할 것으로 기대하지만, 완전히 고정이라고
  가정하면 안 된다.
- Pi3/Pi3X에서 얻은 카메라 포즈는 frame별로 흔들릴 수 있으므로, trial 단위
  대표 포즈와 frame별 포즈를 모두 보존한다.
- HSMR 출력은 monocular reconstruction이다. HSMR의 3D body/joint 결과가
  곧바로 metric world 좌표라고 가정하면 안 된다.
- 따라서 카메라 포즈 변환만으로 끝내지 않고, scale, translation, axis
  convention, handedness, root alignment를 반드시 검증한다.
- `cam_c`를 기준 view로 삼아 `cam_a`, `cam_b`, `cam_c`, 그리고 `abc` fusion
  결과를 비교한다.

## Stage 1. Shoulder Dataset Manifest

`data_inputs/shoulder`를 탐색해 trial 단위 manifest를 만든다.

출력 예:

```text
data_outputs/shoulder/manifests/shoulder_manifest.json
```

manifest 항목:

```json
{
  "subject": "subject01",
  "movement": "001_flexion",
  "views": {
    "cam_a": "data_inputs/shoulder/subject01/001_flexion/cam_a.mp4",
    "cam_b": "data_inputs/shoulder/subject01/001_flexion/cam_b.mp4",
    "cam_c": "data_inputs/shoulder/subject01/001_flexion/cam_c.mp4"
  },
  "camera_layout": {
    "cam_a": "left",
    "cam_b": "right",
    "cam_c": "center"
  },
  "reference_view": "cam_c",
  "sync": "frame_aligned"
}
```

검증 항목:

- 세 view 파일 존재 여부
- fps 일치 여부
- frame count 및 duration 차이
- 누락된 subject/movement/view 기록

## Stage 2. Pi3/Pi3X Camera Pose Estimation

각 trial에서 synchronized frame set을 샘플링해 Pi3/Pi3X에 입력한다.

예:

```text
frame 0:   cam_a, cam_b, cam_c
frame 30:  cam_a, cam_b, cam_c
frame 60:  cam_a, cam_b, cam_c
...
```

저장할 결과:

```text
data_outputs/shoulder/pi3/subject01/001_flexion/
|-- pi3_geometry.npz
|-- camera_poses.json
|-- pointcloud.ply
`-- pose_stability.json
```

저장 내용:

- `camera_poses[frame, view]`
- `relative_poses[frame, view_from, view_to]`
- `points`
- `confidence`
- sampled frame indices
- Pi3/Pi3X model and checkpoint metadata

카메라 포즈 처리:

- frame별 카메라 포즈를 그대로 저장한다.
- 카메라가 거의 고정되어 있다는 가정하에 robust average pose를 계산한다.
- frame별 pose deviation을 측정한다.
- deviation이 큰 구간은 fusion에서 confidence를 낮추거나 별도 경고로 기록한다.

## Stage 3. HSMR Per Camera

각 카메라 영상에 대해 HSMR을 독립적으로 실행한다.

```text
cam_a.mp4 -> HSMR -> cam_a pose sequence
cam_b.mp4 -> HSMR -> cam_b pose sequence
cam_c.mp4 -> HSMR -> cam_c pose sequence
```

출력 구조:

```text
data_outputs/shoulder/hsmr/subject01/001_flexion/
|-- cam_a/
|   `-- HSMR-cam_a.npy
|-- cam_b/
|   `-- HSMR-cam_b.npy
`-- cam_c/
    `-- HSMR-cam_c.npy
```

각 view에서 추출할 정보:

- HSMR poses
- HSMR betas
- selected primary subject index
- bbox scale
- detection confidence if available
- recovered SKEL joints

## Stage 4. Recover Joints and Align to Camera Poses

각 HSMR 결과에서 SKEL joints를 복원한다.

```text
joints_cam_a[t, joint, xyz]
joints_cam_b[t, joint, xyz]
joints_cam_c[t, joint, xyz]
```

Pi3/Pi3X의 카메라 포즈를 이용해 각 view의 joints를 공통 좌표계로 변환한다.
개념적으로는 다음 변환이다.

```text
joints_world_view[t] = T_world_cam_view[t] @ joints_cam_view[t]
```

실제 구현에서는 아래 보정이 필요하다.

- HSMR camera coordinate convention 확인
- Pi3/Pi3X camera-to-world 또는 world-to-camera 방향 확인
- 축 방향 및 handedness 확인
- HSMR/Pi3 scale 차이 보정
- root, pelvis, torso center, shoulder center 기준 translation 보정
- frame별 camera pose를 쓸지, trial 대표 pose를 쓸지 비교

초기 구현에서는 두 좌표 변환을 모두 비교한다.

```text
variant_static_pose:
  trial-level robust camera pose 사용

variant_dynamic_pose:
  frame-level camera pose 사용
```

## Stage 5. Center View 기준 비교

`cam_c`를 기준 view로 둔다. 모든 결과는 우선 `cam_c` 기준 좌표계 또는
`cam_c`와 정렬된 world 좌표계에서 비교한다.

비교 대상:

- `cam_c` 단일뷰 HSMR
- `cam_a`를 `cam_c` 기준으로 변환한 HSMR
- `cam_b`를 `cam_c` 기준으로 변환한 HSMR
- `cam_a + cam_b + cam_c` multiview fusion HSMR

비교 결과 저장:

```text
data_outputs/shoulder/eval/subject01/001_flexion/
|-- view_alignment_report.json
|-- single_view_vs_fused_metrics.csv
|-- cam_c_reference_overlay.mp4
`-- comparison_report.md
```

## Stage 6. Multiview Human Pose Fusion

같은 frame index에서 세 view의 world joints를 합친다.

초기 fusion 전략:

```text
1. cam_a, cam_b, cam_c joints를 공통 좌표계로 변환
2. joint별 median skeleton 계산
3. median에서 너무 멀리 떨어진 view를 outlier로 표시
4. 남은 view를 confidence-weighted mean으로 fusion
5. temporal smoothing 적용
```

가중치 후보:

- HSMR detection confidence
- HSMR bbox scale 안정성
- Pi3/Pi3X camera pose confidence
- Pi3/Pi3X point confidence
- 해당 limb의 view visibility
- 다른 view와의 joint disagreement
- temporal jitter

출력:

```text
data_outputs/shoulder/fused/subject01/001_flexion/
|-- fused_joints.npz
|-- fusion_quality.json
|-- fusion_weights.npz
`-- fused_preview.mp4
```

## Stage 7. Accuracy and Consistency Evaluation

ground truth marker data가 없는 경우, 정확도 평가는 view consistency와
biomechanical plausibility 중심으로 수행한다.

필수 비교:

- `cam_a` vs `cam_c`
- `cam_b` vs `cam_c`
- `cam_c` vs `abc_fused`
- `cam_a` vs `cam_b` vs `cam_c`
- single best view vs `abc_fused`
- static camera pose transform vs dynamic camera pose transform

joint-level metric:

- root-aligned MPJPE-like distance between views
- shoulder, elbow, wrist disagreement
- limb length variance
- shoulder width variance
- left/right shoulder consistency where relevant
- outlier frame count

trajectory metric:

- wrist trajectory smoothness
- elbow trajectory smoothness
- velocity jitter
- acceleration noise
- temporal discontinuity count

ROM metric:

- flexion ROM difference
- abduction ROM difference
- external rotation ROM difference
- angle jitter
- angular velocity noise
- angular acceleration noise
- peak count consistency
- movement time consistency

reprojection/geometry metric:

- fused joints projected back to each camera view
- reprojection error against 2D detections if available
- Pi3 point cloud confidence near body region if usable
- camera pose deviation over time

## Stage 8. Shoulder ROM Analysis

최종적으로 `abc_fused` joints를 기존 ShoulderLab ROM 분석에 넣는다.

```text
fused_joints
-> torso-compensated local coordinate system
-> shoulder angles
-> temporal features
-> summary report
```

출력:

```text
data_outputs/shoulder/analysis/subject01/001_flexion/
|-- fused_results.json
|-- fused_angles.png
|-- fused_temporal_features.png
|-- single_view_comparison.csv
`-- quality_report.md
```

## Proposed CLI

초기 CLI는 단계별 실행과 end-to-end 실행을 모두 지원한다.

```bash
python scripts/shoulderlab.py shoulder-manifest

python scripts/shoulderlab.py pi3-shoulder \
  --subject subject01 \
  --movement 001_flexion

python scripts/shoulderlab.py hsmr-shoulder \
  --subject subject01 \
  --movement 001_flexion

python scripts/shoulderlab.py fuse-shoulder \
  --subject subject01 \
  --movement 001_flexion \
  --reference-view cam_c

python scripts/shoulderlab.py analyze-shoulder \
  --subject subject01 \
  --movement 001_flexion \
  --reference-view cam_c
```

end-to-end 실행:

```bash
python scripts/shoulderlab.py shoulder-pipeline \
  --subject subject01 \
  --movement 001_flexion \
  --reference-view cam_c \
  --skip-video
```

## First Experiment

첫 실험은 `subject01/001_flexion` 하나로 제한한다.

실험 순서:

1. `cam_a`, `cam_b`, `cam_c`에서 같은 frame index를 10-30개 샘플링한다.
2. Pi3/Pi3X로 frame별 camera pose를 추정한다.
3. frame별 pose deviation과 trial-level representative pose를 저장한다.
4. 각 camera video에 HSMR을 실행한다.
5. 각 view의 SKEL joints를 복원한다.
6. `cam_c` 기준으로 `cam_a`, `cam_b`, `cam_c` skeleton을 정렬한다.
7. static camera pose와 dynamic camera pose 변환 결과를 비교한다.
8. `cam_a`, `cam_b`, `cam_c`, `abc_fused`의 joint trajectory와 shoulder ROM을 비교한다.

이 실험에서 확인할 성공 조건:

- 세 view의 skeleton이 같은 사람 위치에 안정적으로 겹친다.
- limb length와 shoulder width가 view 간 크게 흔들리지 않는다.
- `abc_fused`의 angle jitter와 acceleration noise가 단일뷰보다 감소한다.
- `cam_c` 기준에서 `cam_a`, `cam_b`가 일관된 shoulder trajectory를 보인다.

## Implementation Milestones

1. Dataset manifest and validation
2. Pi3/Pi3X camera pose wrapper
3. Per-view HSMR batch runner
4. SKEL joint recovery cache
5. Camera pose alignment prototype
6. `cam_c` reference comparison report
7. Robust multiview fusion
8. Fused ROM analysis and quality report

