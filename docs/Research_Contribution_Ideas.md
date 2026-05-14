# 연구 기여점 정리

현재 작업은 기존 HSMR/SKEL 모델을 이용해 입력 영상에서 3D skeleton과 어깨 ROM 지표를 산출하는 순차 파이프라인에 가깝다. 이 상태만으로는 논문 기여가 "기존 모델을 적용했다"는 수준에 머물 수 있다.

따라서 연구 기여는 단순 inference pipeline이나 multi-view 입력 지원이 아니라, HSMR/SKEL 출력이 임상적 어깨 분석에 바로 쓰이기 어려운 지점을 정의하고, 이를 multi-view, 생체역학적 지표, 신뢰도 평가로 보완하는 방향으로 잡는 것이 적절하다.

## 1. 권장 논문 주장

권장하는 중심 주장은 다음과 같다.

```text
Sparse multi-view RGB 입력으로부터 HSMR/SKEL 기반 markerless shoulder kinematics를 추정하고,
단순 3D pose 정확도가 아니라 임상적으로 해석 가능한 shoulder ROM, temporal feature,
confidence를 산출하는 프레임워크를 제안한다.
```

여기서 핵심은 multi-view 자체가 아니라 다음 세 가지의 결합이다.

- sparse multi-view RGB 입력
- SKEL 기반 생체역학적 skeleton reconstruction
- shoulder-specific ROM, temporal feature, confidence 분석

즉 "multi-view HMR을 했다"가 아니라 "어깨 기능 평가를 위한 markerless biomechanical analysis layer를 제안했다"는 방향으로 논문을 구성해야 한다.

## 2. Contribution 1: Confidence-aware shoulder ROM

현재 `flexion`, `abduction`, `external rotation` 계산은 HSMR/SKEL에서 얻은 관절점을 이용한 어깨 ROM proxy다. 이 값을 단순히 출력하는 것만으로는 기여가 약하다. 대신 각 프레임의 ROM 값과 함께 신뢰도를 추정하면 연구성이 생긴다.

가능한 confidence 항목은 다음과 같다.

- multi-view reprojection consistency
- view 간 SKEL pose parameter agreement
- humerus vector와 forearm vector의 안정성
- external rotation 계산 시 projection norm
- elbow, wrist, shoulder keypoint visibility 또는 occlusion score
- temporal jitter score
- LCS 축 직교성 및 determinant 안정성

이렇게 하면 결과는 단순한 각도 시계열이 아니라, 어떤 프레임의 ROM이 믿을 만하고 어떤 프레임은 해석에서 제외해야 하는지를 함께 제공하는 `confidence-aware shoulder assessment`가 된다.

논문에서 사용할 수 있는 주장:

```text
We propose a confidence-aware shoulder ROM estimation framework that reports not only
ROM angles but also frame-level reliability scores derived from multi-view consistency,
limb geometry stability, and temporal noise.
```

## 3. Contribution 2: Shoulder-specific biomechanical descriptor

HSMR 자체는 전신의 biomechanically accurate skeleton reconstruction을 목표로 한다. 따라서 후속 연구에서는 전신 mesh reconstruction 정확도보다 shoulder-specific kinematic representation을 명확히 정의하는 것이 좋다.

현재 구현은 다음과 같이 해석하는 것이 안전하다.

```text
torso-compensated thoracohumeral arm-angle proxy
```

즉, 현재 각도는 엄밀한 glenohumeral joint angle이나 ISB 표준 shoulder kinematics가 아니라, 몸통 좌표계로 보정한 상완 움직임 지표다. 이 한계를 숨기기보다 논문에서 명확히 정의하고, shoulder function analysis에 필요한 descriptor set으로 발전시키는 것이 좋다.

제안 가능한 descriptor는 다음과 같다.

- thorax 기준 humerus elevation
- plane of elevation
- flexion component
- abduction component
- axial rotation proxy
- external rotation confidence
- ROM range
- peak angle
- movement time
- angular velocity
- angular acceleration
- smoothness
- peak count
- temporal jitter

논문에서 사용할 수 있는 주장:

```text
We define a shoulder-specific kinematic descriptor set from SKEL outputs, including
torso-compensated elevation, plane of elevation, axial rotation proxy, ROM, velocity,
acceleration, smoothness, and noise-sensitive temporal features.
```

이 방향은 "HSMR를 사용했다"가 아니라 "HSMR/SKEL 출력을 임상적 어깨 기능 분석에 맞는 표현으로 변환했다"는 기여가 된다.

## 4. Contribution 3: Biomechanically constrained multi-view refinement

단순한 multi-view 처리는 각 카메라에서 HSMR를 독립적으로 실행한 뒤 결과를 평균내거나 voting하는 방식이 될 수 있다. 이는 논문 기여로 약하다.

더 강한 방법은 하나의 공통 SKEL pose를 두고 여러 카메라 관측을 동시에 만족하도록 test-time optimization 또는 refinement를 수행하는 것이다.

예시 목적함수는 다음과 같다.

```text
L = L_reprojection
  + L_cross_view_pose_consistency
  + L_temporal_smoothness
  + L_anatomical_prior
  + L_shoulder_motion_prior
```

각 항의 의미는 다음과 같다.

| Loss | 목적 |
|---|---|
| `L_reprojection` | 각 view의 2D keypoint 또는 silhouette와 3D skeleton 투영 일치 |
| `L_cross_view_pose_consistency` | view별 HSMR 결과가 같은 SKEL pose를 가리키도록 제한 |
| `L_temporal_smoothness` | 프레임 간 pose, angle, velocity의 불필요한 jitter 감소 |
| `L_anatomical_prior` | SKEL joint limit과 비현실적 관절 자세 억제 |
| `L_shoulder_motion_prior` | 어깨 동작에서 과도한 twist, 비정상적 ROM, 불안정한 external rotation 억제 |

논문에서 사용할 수 있는 주장:

```text
We introduce a biomechanically constrained multi-view refinement step that optimizes
a shared SKEL pose sequence using multi-view reprojection, anatomical priors, and
shoulder-specific temporal regularization.
```

이 경우 기여점은 "multi-view 입력을 받는다"가 아니라 "multi-view 관측을 이용해 shoulder kinematics의 안정성과 해석 가능성을 높인다"가 된다.

## 5. Contribution 4: Shoulder movement evaluation protocol

일반적인 HMR 논문처럼 MPJPE, PA-MPJPE, PVE만 평가하면 HSMR 원 논문이나 기존 HMR 연구와 정면으로 비교해야 한다. 이 경우 새 연구의 독립성이 약해질 수 있다.

따라서 평가는 전신 pose accuracy보다 shoulder movement assessment 중심으로 설계하는 것이 좋다.

권장 동작 세트:

- flexion
- abduction
- internal rotation
- external rotation
- internal rotation with abduction
- external rotation with abduction
- circumduction
- guide 또는 functional reach motion

권장 평가 지표:

- ROM error
- peak angle error
- movement time error
- angular velocity correlation
- angular acceleration stability
- temporal smoothness
- intra-session repeatability
- single-view 대비 multi-view 개선폭
- confidence filtering 전후의 error 변화
- occlusion 구간에서의 robustness

가능하다면 기준값은 다음 중 하나를 사용한다.

- goniometer 측정값
- IMU 기반 관절각
- marker-based motion capture
- expert annotation
- calibrated multi-view triangulation 기반 pseudo-ground truth

논문에서 사용할 수 있는 주장:

```text
We propose a shoulder-specific evaluation protocol that measures clinically relevant
movement outcomes such as ROM, peak angle, movement time, angular velocity, repeatability,
and confidence-filtered reliability, rather than relying only on generic 3D pose metrics.
```

## 6. 논문 기여점 후보 정리

가장 설득력 있는 최종 contribution 구성은 다음과 같다.

1. Sparse multi-view RGB 기반 HSMR/SKEL shoulder reconstruction framework
2. Confidence-aware shoulder ROM and temporal feature extraction
3. Biomechanically constrained multi-view SKEL refinement
4. Shoulder-specific movement evaluation protocol

이를 논문 introduction에는 다음처럼 쓸 수 있다.

```text
Our contributions are threefold:

1. We present a sparse multi-view markerless framework for shoulder motion analysis
   based on biomechanically constrained SKEL reconstruction.

2. We introduce a shoulder-specific kinematic descriptor set with frame-level confidence,
   including torso-compensated ROM, plane of elevation, axial rotation proxy, temporal
   features, and noise-sensitive reliability measures.

3. We design a shoulder movement evaluation protocol that assesses ROM accuracy,
   temporal stability, repeatability, and robustness under view occlusion, demonstrating
   the benefit of multi-view biomechanical refinement over single-view baselines.
```

## 7. 피해야 할 주장

현재 구현만으로는 다음 표현은 위험하다.

- 정확한 glenohumeral ROM을 추정한다.
- ISB 표준 shoulder kinematics를 구현했다.
- 임상 goniometer와 동일한 flexion/abduction/external rotation을 계산한다.
- multi-view HMR 자체를 새롭게 제안한다.
- HSMR 모델 구조 자체를 개선했다.

대신 다음 표현이 더 안전하다.

- HSMR/SKEL 기반 shoulder ROM proxy
- torso-compensated thoracohumeral angle
- shoulder-specific functional kinematic descriptor
- confidence-aware markerless shoulder assessment
- sparse multi-view biomechanical refinement

## 8. 구현 우선순위

논문 기여로 연결하려면 다음 순서가 현실적이다.

### Priority 1: 현재 ROM 정의 정리

- `flexion`, `abduction`, `external rotation`이 정확히 무엇을 의미하는지 문서화한다.
- ISB 표준 각도와의 차이를 명확히 쓴다.
- proxy angle임을 숨기지 않는다.

### Priority 2: confidence score 추가

- 각 프레임별 geometry confidence를 저장한다.
- external rotation은 projection norm 기반 confidence를 반드시 둔다.
- low-confidence frame을 downstream temporal feature에서 제외할 수 있게 한다.

### Priority 3: multi-view consistency metric 구현

- view별 HSMR 결과의 3D joint 또는 SKEL parameter 차이를 측정한다.
- 3D skeleton을 각 카메라에 투영해 reprojection error를 계산한다.
- view occlusion과 angle error의 관계를 분석한다.

### Priority 4: temporal feature 안정화

- Savitzky-Golay smoothing 전후 angular velocity, acceleration noise를 비교한다.
- temporal jitter와 confidence의 상관관계를 분석한다.
- movement별 ROM, peak, duration, velocity feature를 표준화한다.

### Priority 5: multi-view refinement 추가

- 단순 averaging baseline을 먼저 만든다.
- 이후 shared SKEL pose optimization을 추가한다.
- single-view, naive multi-view, proposed refinement를 비교한다.

## 9. 최종 방향

가장 안전하고 설득력 있는 연구 방향은 다음과 같다.

```text
기존 HSMR/SKEL 모델을 shoulder assessment에 그대로 적용하는 것이 아니라,
어깨 분석에서 필요한 좌표계, ROM proxy, temporal feature, confidence, multi-view consistency를
정의하고 검증하는 markerless shoulder kinematics framework를 제안한다.
```

이렇게 구성하면 기존 모델을 사용하는 한계는 남지만, 논문 기여는 모델 구조가 아니라 임상적 shoulder motion analysis를 위한 방법론과 평가 프로토콜에 놓이게 된다.
