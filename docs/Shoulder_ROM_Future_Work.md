# Shoulder ROM Future Work

이 문서는 현재 `shoulderlab/rom.py`의 어깨 ROM 계산 방식에 대한 검증 결과와 향후 개선 과제를 정리한다.

현재 구현은 HSMR/SKEL에서 복원한 3D 관절 좌표를 몸통 기준 Local Coordinate System(LCS)으로 변환한 뒤, 상완과 전완 벡터에서 `flexion`, `abduction`, `ext_rotation`을 추정한다. 이 방식은 단일축 synthetic motion에서는 의도한 결과를 내지만, 수학적으로나 생체역학적으로 완전한 어깨 관절 운동학 모델은 아니다.

## 1. 현재 방식의 위치

현재 결과는 다음과 같이 해석하는 것이 가장 안전하다.

```text
torso-compensated thoracohumeral arm-angle proxy
```

즉 몸통 기울기를 보정한 뒤, 상완 방향과 전완 방향을 이용해 어깨 움직임을 추정한 기능적 지표다. 이는 임상적 glenohumeral joint angle 또는 ISB 표준 shoulder kinematics와 동일한 정의가 아니다.

## 2. 확인된 강점

### 2.1 단일축 synthetic case에서의 일관성

현재 `compute_angles()`는 다음 테스트에서 의도한 값을 낸다.

| 자세 | 기대 결과 | 현재 결과 |
|---|---|---|
| 오른팔 abduction 90도 | `abduction=90`, `flexion=0` | 일치 |
| 왼팔 abduction 90도 | `abduction=90`, `flexion=0` | 일치 |
| flexion 90도 | `flexion=90`, `abduction=0` | 일치 |
| extension 90도 | `flexion=-90`, `abduction=0` | 일치 |
| flexion 150도 | `flexion=150`, `abduction=0` | 일치 |

따라서 순수 flexion/abduction 방향의 상완 벡터를 입력했을 때, `E * sin(phi)`와 `E * cos(phi)` 분해는 cross-talk를 만들지 않는다.

### 2.2 고각도 투영 스파이크 회피

단순히 2D 평면 투영 후 `atan2()`로 각도를 구하는 방식은 고각도에서 부호 반전이나 180도 스파이크를 만들 수 있다. 현재 방식은 총 거상각 `E`와 거상 평면 방위각 `phi`를 분리하므로 이 문제를 줄인다.

### 2.3 구현이 단순하고 입력 요구사항이 낮음

현재 방식은 `neck`, `mid_hip`, 양쪽 `hip`, `shoulder`, `elbow`, `wrist`만 필요하다. 별도의 marker set, GH center, scapula landmark, elbow epicondyle이 없어도 계산 가능하다는 점은 HSMR 출력 기반 분석에서 실용적이다.

## 3. 확인된 문제

### 3.1 LCS가 proper rotation이 아님

현재 LCS는 다음처럼 구성된다.

```text
Y = normalize(neck - mid_hip)
X = normalize((r_hip - l_hip) orthogonalized to Y)
Z = normalize(cross(Y, X))
R = [X Y Z]
```

이 구성은 이상적인 기본 자세에서도 `det(R) = -1`이 된다. 즉 `R`은 수학적으로 proper rotation matrix가 아니라 reflection을 포함한 left-handed basis다.

현재 코드 주석에는 “physical Right-Handedness 유지”라는 취지의 설명이 있으나, 이는 `det(R) = +1`인 오른손 좌표계와 맞지 않는다.

개선 필요사항:

- `Z = cross(Y, X)`를 유지할지, `Z = cross(X, Y)`로 바꿀지 명확히 결정한다.
- 현재 HSMR/SKEL 좌표 convention에서 실제 전방 방향이 어느 부호인지 시각화로 검증한다.
- `R`을 좌표 변환 행렬로 쓸 때 left-handed basis를 허용할지, proper rotation으로 강제할지 결정한다.
- 코드 docstring과 주석을 실제 구현과 일치시킨다.

### 3.2 코드 docstring과 구현 불일치

`compute_lcs()`의 docstring에는 다음 불일치가 있다.

| 항목 | docstring | 실제 코드 |
|---|---|---|
| Z축 | `X x Y` | `Y x X` |
| 원점 설명 | pelvis center | `neck` |

이 불일치는 계산 자체를 바꾸지는 않지만, 향후 좌표계 검증과 디버깅을 어렵게 만든다.

우선순위 높은 정리 작업으로 처리해야 한다.

### 3.3 Flexion/abduction은 회전의 엄밀한 분해가 아님

현재 방식은 총 거상각 `E`를 다음처럼 나눈다.

```text
flexion   = E * sin(phi)
abduction = E * cos(phi)
```

이는 상완 방향을 임상적으로 이해하기 쉽게 표현하는 휴리스틱 성분 분해다. 하지만 3D 회전은 일반 벡터처럼 큰 각도에서 선형 성분 분해할 수 없다. 따라서 이 값들을 “정확한 해부학적 flexion/abduction 회전각”으로 해석하면 안 된다.

특히 다음 상황에서 해석이 애매해진다.

- flexion과 abduction이 동시에 큰 scaption 또는 diagonal reach
- 상완이 120도 이상 거상된 고각도 구간
- 몸통 회전, 견갑 움직임, 팔의 axial rotation이 함께 발생하는 동작

개선 방향:

- 현재 지표명을 `flexion_component`, `abduction_component`처럼 proxy임이 드러나게 바꿀지 검토한다.
- ISB식 `plane of elevation`, `elevation angle`, `axial rotation`을 별도 출력으로 추가한다.
- 기존 `flexion`/`abduction`은 downstream 호환을 위해 유지하되 문서에서 의미를 제한한다.

### 3.4 External rotation은 전완 기반 proxy임

현재 external rotation은 전완 벡터 `F = wrist - elbow`를 이용해 상완 장축 `H` 주변 twist를 추정한다.

문제는 손목 1점과 팔꿈치 1점만으로는 humeral axial rotation을 완전히 결정할 수 없다는 점이다. 현재 값에는 다음 성분이 섞일 수 있다.

- elbow flexion angle 변화
- forearm pronation/supination
- wrist 위치 추정 오차
- 팔꿈치가 거의 펴진 상태에서의 projection 불안정성
- 실제 humerus local frame 부재

따라서 현재 `ext_rotation`은 정확한 glenohumeral external rotation이 아니라 전완 방향으로 추정한 twist proxy다.

개선 방향:

- 팔꿈치가 충분히 굽혀진 프레임에서만 external rotation을 보고한다.
- `F_proj`와 `F_ref_proj` norm을 confidence score로 저장한다.
- 가능하면 elbow medial/lateral epicondyle 또는 humerus local frame을 구성할 수 있는 추가 landmark를 사용한다.
- forearm pronation/supination이 큰 동작에서는 external rotation 해석을 제한한다.

### 3.5 Scapula와 glenohumeral joint가 모델링되지 않음

현재 좌표계는 pelvis와 trunk 중심축을 기준으로 하며, scapula 좌표계나 GH center를 사용하지 않는다.

생체역학적으로 어깨는 단일 관절이 아니라 다음 구조가 결합된 shoulder complex다.

- sternoclavicular joint
- acromioclavicular joint
- scapulothoracic motion
- glenohumeral joint

현재 방식은 이 중 humerus가 thorax 기준으로 어떻게 움직이는지를 근사한다. 견갑골 회전과 glenohumeral 회전을 분리하지 못하므로, 진짜 GH ROM이나 scapulohumeral rhythm을 계산할 수 없다.

개선 방향:

- scapula landmark 또는 예측 가능한 scapula proxy를 도입한다.
- GH center를 추정한다.
- humerus orientation을 shoulder-to-elbow 벡터 하나가 아니라 local humerus frame으로 표현한다.
- thoracohumeral, scapulothoracic, glenohumeral motion을 분리해 출력한다.

## 4. ISB 표준 대비 차이

ISB 권장 방식은 thorax, scapula, humerus의 segment coordinate system과 shoulder joint coordinate system을 정의한 뒤, distal segment의 orientation을 proximal segment 기준으로 보고한다.

현재 구현과의 주요 차이는 다음과 같다.

| 항목 | 현재 구현 | ISB 수준 접근 |
|---|---|---|
| 기준 좌표계 | neck-midhip-hip 기반 trunk LCS | thorax, scapula, humerus anatomical SCS |
| 어깨 중심 | shoulder joint point | GH center 추정 필요 |
| 상완 자세 | shoulder-to-elbow 단일 벡터 | humerus local frame |
| 견갑 움직임 | 미분리 | scapula coordinate system 필요 |
| 회전 표현 | `E`, `phi` 기반 성분 proxy | Euler/Cardan 또는 JCS sequence |
| 외회전 | forearm vector 기반 twist proxy | humerus axial rotation |

따라서 현재 방식은 ISB 표준의 일부 개념, 특히 `plane of elevation`과 `elevation angle`의 형태를 참고하지만, ISB 표준 구현이라고 부르면 안 된다.

## 5. 개선 우선순위

### Priority 0: 문서와 코드 주석 정합성 수정

- `compute_lcs()` docstring의 `Z-axis: X x Y`를 실제 코드와 맞춘다.
- `origin_list` 설명을 pelvis center가 아니라 neck으로 수정한다.
- `Right-Handedness` 관련 주석을 실제 `det(R)` 결과와 맞게 수정한다.

이 작업은 동작을 바꾸지 않고 혼란을 줄이는 정리 작업이다.

### Priority 1: 좌표계 handedness 결정

다음 두 선택지 중 하나를 명확히 선택해야 한다.

```text
Option A: 현재 Z = cross(Y, X)를 유지
```

- 장점: 기존 결과와 호환된다.
- 단점: left-handed basis이며 proper rotation이 아니다.
- 필요 조치: 문서와 변수명을 “basis transform”으로 명확히 쓰고, 오른손 좌표계라고 설명하지 않는다.

```text
Option B: Z = cross(X, Y)로 변경
```

- 장점: `R = [X Y Z]`가 proper rotation이 된다.
- 단점: 현재 flexion/extension 부호와 anterior 방향이 뒤집힐 수 있다.
- 필요 조치: 기존 결과 JSON/plot과 호환성 검토, 부호 convention 재검증.

### Priority 2: synthetic regression test 추가

최소한 다음 케이스를 자동 테스트로 고정한다.

- right/left abduction 90도
- flexion 90도
- extension 90도
- flexion 150도
- scaption 90도
- external rotation synthetic twist
- degenerate humerus/forearm length
- LCS determinant와 축 직교성

테스트 목적은 현재 proxy의 한계를 없애는 것이 아니라, convention이 의도치 않게 바뀌는 일을 막는 것이다.

### Priority 3: 결과 schema에 confidence 추가

각 프레임별 각도와 함께 다음 품질 지표를 저장한다.

- humerus length
- forearm length
- `norm(F_proj)`
- `norm(F_ref_proj)`
- LCS axis determinant
- LCS fallback 사용 여부

특히 external rotation은 projection norm이 작은 프레임에서 신뢰도가 낮으므로, downstream 분석에서 필터링할 수 있어야 한다.

### Priority 4: ISB식 출력 추가

현재 `flexion`, `abduction`, `ext_rotation`을 바로 대체하기보다, 별도 필드를 추가하는 방식이 안전하다.

예시:

```json
{
  "angles": {
    "flexion": [],
    "abduction": [],
    "ext_rotation": [],
    "plane_of_elevation": [],
    "elevation": [],
    "axial_rotation_proxy": []
  }
}
```

이렇게 하면 기존 plot과 temporal feature pipeline을 깨지 않으면서 더 엄밀한 지표를 병행 검증할 수 있다.

### Priority 5: anatomical shoulder model 검토

장기적으로는 다음 중 하나가 필요하다.

- SKEL에서 scapula/clavicle/humerus orientation을 직접 사용할 수 있는지 확인
- GH center와 humerus local frame을 추정하는 후처리 추가
- markerless pose에서 scapula를 추정하는 별도 model 또는 regression 도입
- ISB JCS 또는 대체 Cardan sequence를 지원하는 옵션 추가

이 단계부터는 단순 문서/수식 수정이 아니라 분석 정의 자체를 바꾸는 작업이다.

## 6. 권장 결론

현재 구현은 연구/프로토타입 단계의 ROM proxy로는 사용할 수 있다. 특히 같은 pipeline, 같은 convention, 같은 운동군 안에서 상대 비교를 하는 목적에는 실용적이다.

하지만 다음 표현은 피해야 한다.

- “완전한 어깨 관절 각도”
- “정확한 glenohumeral ROM”
- “ISB 표준 shoulder kinematics 구현”
- “임상 goniometer와 동일한 flexion/abduction/external rotation”

권장 표현은 다음과 같다.

- “HSMR 관절점 기반 어깨 ROM proxy”
- “몸통 보정 thoracohumeral angle”
- “상완 방향 기반 flexion/abduction component”
- “전완 방향 기반 axial rotation proxy”

향후 완전한 생체역학 해석이 필요하다면, 좌표계 handedness 정리와 synthetic test를 먼저 고정한 뒤, scapula/GH/humerus local frame을 포함한 ISB식 모델로 확장해야 한다.
