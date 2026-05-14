# 어깨 ROM 좌표계 및 각도 계산 방식

이 문서는 `shoulderlab/rom.py`의 현재 구현을 기준으로, HSMR/SKEL에서 복원한 3D 관절 좌표를 어떤 좌표계로 변환하고, 그 좌표계에서 어깨 각도 3종을 어떻게 계산하는지 설명한다.

분석 대상 각도는 다음 3개다.

- `flexion`: 상완의 전방 거상 성분
- `abduction`: 상완의 측방 거상 성분
- `ext_rotation`: 상완 장축을 기준으로 한 전완의 내/외회전 성분

모든 최종 각도는 degree 단위로 저장된다.

## 1. 입력 관절

입력은 SKEL forward 결과에서 얻은 `joints` 배열이다.

```text
joints: (T, 44, 3)
```

현재 어깨 분석에서 사용하는 관절 인덱스는 OpenPose25 기준의 앞쪽 관절이다.

| 이름 | 인덱스 | 용도 |
|---|---:|---|
| `neck` | 1 | 국소 좌표계 원점, 몸통 상단 기준점 |
| `r_shoulder` | 2 | 오른쪽 어깨 |
| `r_elbow` | 3 | 오른쪽 팔꿈치 |
| `r_wrist` | 4 | 오른쪽 손목 |
| `l_shoulder` | 5 | 왼쪽 어깨 |
| `l_elbow` | 6 | 왼쪽 팔꿈치 |
| `l_wrist` | 7 | 왼쪽 손목 |
| `mid_hip` | 8 | 몸통 하단 기준점 |
| `r_hip` | 9 | 오른쪽 골반 |
| `l_hip` | 12 | 왼쪽 골반 |

## 2. 프레임별 국소 좌표계

각 프레임마다 몸통 기준의 Local Coordinate System(LCS)을 만든다. 목적은 카메라/월드 좌표에서의 전역 위치와 몸통 기울기를 제거하고, 팔 움직임을 몸통 기준으로 해석하는 것이다.

### 2.1 원점

원점은 `neck`이다.

```text
origin = neck
```

따라서 LCS로 변환된 좌표에서 목 관절은 `(0, 0, 0)`에 놓인다.

### 2.2 Y축: 몸통 상방

Y축은 `mid_hip`에서 `neck`으로 향하는 단위 벡터다.

```text
y_raw  = neck - mid_hip
Y      = normalize(y_raw)
```

의미는 `superior`, 즉 몸통 기준 위쪽 방향이다. 벡터 길이가 너무 작으면 fallback으로 `(0, 1, 0)`을 사용한다.

### 2.3 X축: 오른쪽 측방

X축은 왼쪽 골반에서 오른쪽 골반으로 향하는 벡터를 기반으로 한다.

```text
x_raw = r_hip - l_hip
```

이 벡터에서 Y축 성분을 제거해 Y축과 직교하도록 만든다.

```text
x_orth = x_raw - dot(x_raw, Y) * Y
X      = normalize(x_orth)
```

의미는 `right-lateral`, 즉 피험자 기준 오른쪽 방향이다. 골반선을 사용하므로 어깨 자체의 들림이나 팔 움직임이 좌표계 X축에 직접 섞이는 것을 줄인다. 벡터 길이가 너무 작으면 fallback으로 `(1, 0, 0)`을 사용한다.

### 2.4 Z축: 전방

Z축은 현재 구현에서 다음과 같이 만든다.

```text
Z = normalize(cross(Y, X))
```

코드 주석상 의미는 `anterior / facing direction`이다. 즉 LCS에서 Z축 양의 방향은 전방으로 해석한다.

주의할 점은 일반적인 오른손 좌표계라면 `cross(X, Y)`가 Z가 되지만, 현재 구현은 `cross(Y, X)`를 사용한다. 이는 HSMR/SKEL 출력과 카메라 이미지 공간의 축 방향을 보정하기 위해 현재 파이프라인에서 선택한 방향 정의다. 문서의 수식과 각도 부호는 이 구현을 기준으로 한다.

### 2.5 회전 행렬과 좌표 변환

각 축을 열벡터로 쌓아 회전 행렬 `R`을 만든다.

```text
R = [X Y Z]
```

월드/카메라 좌표의 관절 `p`는 다음 식으로 LCS 좌표 `p_local`이 된다.

```text
p_local = R.T @ (p - origin)
```

즉 LCS 좌표의 각 성분은 관절 위치 벡터를 `X`, `Y`, `Z`축에 각각 투영한 값이다.

```text
p_local.x = dot(p - origin, X)
p_local.y = dot(p - origin, Y)
p_local.z = dot(p - origin, Z)
```

정리하면 현재 LCS의 의미는 다음과 같다.

| 축 | 양의 방향 | 설명 |
|---|---|---|
| X | 피험자 오른쪽 | 오른팔 외측 방향은 `+X`, 왼팔 외측 방향은 `-X` |
| Y | 위쪽 | `mid_hip -> neck` 방향 |
| Z | 전방 | 현재 구현에서 `cross(Y, X)`로 정의한 전방 방향 |

## 3. 팔 벡터 정의

각도 계산은 LCS로 변환된 관절에서 시작한다.

상완 벡터 `H`는 어깨에서 팔꿈치로 향하는 단위 벡터다.

```text
H = normalize(elbow - shoulder)
```

전완 벡터 `F`는 팔꿈치에서 손목으로 향하는 단위 벡터다.

```text
F = normalize(wrist - elbow)
```

상완 또는 전완 벡터 길이가 너무 작으면 해당 프레임의 각도는 계산하지 않고 `NaN`으로 남긴다.

휴식 기준 상완 방향과 전완 방향은 다음과 같이 둔다.

```text
Y_rest = (0, -1, 0)  # 팔이 몸 옆으로 아래를 향함
F_rest = (0,  0, 1)  # 팔꿈치를 90도 굽혔을 때 전완이 전방을 향함
```

## 4. Flexion과 Abduction

Flexion과 abduction은 상완 벡터 `H`를 총 거상각과 거상 평면 방위각으로 나눈 뒤, 그 거상각을 시상면 성분과 관상면 성분으로 분해해 계산한다.

### 4.1 총 거상각

총 거상각 `E`는 휴식 상완 방향 `Y_rest`와 현재 상완 방향 `H`의 사이각이다.

```text
cos_el = clip(dot(H, Y_rest), -1, 1)
E      = arccos(cos_el)
```

`Y_rest = (0, -1, 0)`이므로 다음과 같다.

```text
cos_el = -H_y
E      = arccos(-H_y)
```

`E`의 범위는 `[0, pi]` radian이다.

### 4.2 좌우 방향 보정

LCS에서 오른팔의 외측 방향은 `+X`, 왼팔의 외측 방향은 `-X`다. 양쪽 팔의 abduction이 모두 양수로 나오도록 `lat_sign`을 둔다.

```text
lat_sign = +1  # right
lat_sign = -1  # left
```

### 4.3 거상 평면 방위각

거상 평면 방위각 `phi`는 상완 벡터가 수평면 XZ에서 어느 방향으로 올라갔는지를 나타낸다.

```text
phi = atan2(H_z, lat_sign * H_x)
```

해석은 다음과 같다.

| `phi` | 의미 |
|---:|---|
| `0 deg` | 순수 abduction 방향 |
| `90 deg` | 순수 flexion 방향 |
| `-90 deg` | extension 방향 |
| `45 deg` | flexion과 abduction이 섞인 scaption 방향 |

### 4.4 임상 각도 성분

총 거상각 `E`를 `phi` 방향에 따라 분해한다.

```text
flexion   = degrees(E * sin(phi))
abduction = degrees(E * cos(phi))
```

이 방식의 중요한 성질은 다음과 같다.

- 순수 flexion에서는 `cos(phi) = 0`이므로 abduction이 0이 된다.
- 순수 abduction에서는 `sin(phi) = 0`이므로 flexion이 0이 된다.
- `|flexion| <= E`, `|abduction| <= E`다.
- `flexion + abduction = E`를 강제하지 않는다. 두 값은 총 거상각의 독립 성분이지 합으로 보존되는 분할량이 아니다.

예시는 다음과 같다.

| 자세 | 오른팔 기준 `H` | `E` | `phi` | flexion | abduction |
|---|---|---:|---:|---:|---:|
| 휴식 | `(0, -1, 0)` | `0 deg` | 불안정 | `0 deg` | `0 deg` |
| 순수 오른팔 abduction 90도 | `(1, 0, 0)` | `90 deg` | `0 deg` | `0 deg` | `90 deg` |
| 순수 flexion 90도 | `(0, 0, 1)` | `90 deg` | `90 deg` | `90 deg` | `0 deg` |
| 순수 flexion 150도 | `(0, 0.866, 0.5)` | `150 deg` | `90 deg` | `150 deg` | `0 deg` |
| scaption 90도 | `(0.707, 0, 0.707)` | `90 deg` | `45 deg` | `63.6 deg` | `63.6 deg` |

## 5. External Rotation

External rotation은 전완 벡터 `F`가 상완 장축 `H` 주위로 얼마나 비틀렸는지를 계산한다. 팔 전체가 공간에서 들어 올려진 swing 성분과, 상완 장축 주변의 twist 성분을 분리하는 방식이다.

계산 순서는 다음과 같다.

1. 휴식 상완 방향 `Y_rest`가 현재 상완 방향 `H`로 이동하는 최소 회전, 즉 swing을 구한다.
2. 같은 swing을 휴식 전완 방향 `F_rest`에 적용해 기준 전완 방향 `F_ref`를 만든다.
3. 실제 전완 `F`와 기준 전완 `F_ref`를 상완 장축에 수직인 평면으로 투영한다.
4. 투영된 두 벡터의 부호 있는 사이각을 twist로 계산한다.
5. 좌우 팔 부호를 보정해 external rotation의 양수 방향을 맞춘다.

### 5.1 Swing 기준 회전

상완의 swing 회전축은 다음과 같다.

```text
axis      = cross(Y_rest, H)
sin_alpha = norm(axis)
cos_alpha = dot(H, Y_rest)
```

`sin_alpha`가 충분히 크면 회전축 단위 벡터 `k`를 만든다.

```text
k = axis / sin_alpha
```

그리고 Rodrigues 회전 공식으로 `F_rest`를 회전시켜 기준 전완 방향 `F_ref`를 만든다.

```text
F_ref =
    F_rest * cos_alpha
  + cross(k, F_rest) * sin_alpha
  + k * dot(k, F_rest) * (1 - cos_alpha)
```

`sin_alpha`가 너무 작으면 swing 회전축이 정의되지 않는다. 현재 구현은 다음 fallback을 사용한다.

```text
if cos_alpha > 0:
    F_ref = F_rest
else:
    F_ref = -F_rest
```

즉 상완이 휴식 방향과 거의 같으면 전완 기준도 그대로 두고, 상완이 휴식 방향의 반대쪽에 가까우면 전완 기준을 반대로 둔다.

### 5.2 상완 수직 평면으로 투영

전완의 순수 twist만 보려면 상완 방향 `H`와 평행한 성분을 제거해야 한다. 따라서 실제 전완과 기준 전완을 모두 `H`에 수직인 평면으로 투영한다.

```text
F_proj     = F     - dot(F, H)     * H
F_ref_proj = F_ref - dot(F_ref, H) * H
```

둘 중 하나라도 투영 벡터 길이가 너무 작으면 twist가 안정적으로 정의되지 않으므로 해당 프레임의 external rotation은 `NaN`으로 남긴다. 이는 팔꿈치가 충분히 굽혀져 있지 않아 전완 방향이 상완 장축 주변 회전을 설명하기 어려운 경우를 포함한다.

### 5.3 부호 있는 twist 각

투영 벡터를 정규화한 뒤, `atan2`로 부호 있는 회전각을 계산한다.

```text
cos_twist = clip(dot(F_ref_proj, F_proj), -1, 1)
sin_twist = dot(cross(F_ref_proj, F_proj), H)
twist     = degrees(atan2(sin_twist, cos_twist))
```

마지막으로 좌우 팔의 외회전 양수 방향을 맞추기 위해 다음 보정을 적용한다.

```text
ext_rotation = -lat_sign * twist
```

따라서 오른팔은 `-twist`, 왼팔은 `+twist`가 최종 external rotation 값이 된다.

## 6. 계산 결과의 의미와 한계

현재 방식은 몸통 기준 LCS에서 팔 벡터를 해석하므로, 카메라 좌표의 전역 회전이나 피험자의 몸통 기울기 영향을 줄인다. 또한 flexion과 abduction은 상완 벡터의 총 거상각을 기준으로 계산되므로, 단순 2D 투영 각도에서 생기는 고각도 구간의 부호 반전이나 180도 스파이크를 피한다.

다만 다음 제한은 남아 있다.

- `Z = cross(Y, X)`의 방향은 현재 파이프라인의 좌표 convention에 묶여 있다. 다른 3D 복원기나 다른 카메라 좌표계를 쓰면 전방/후방 부호를 재검증해야 한다.
- Flexion과 abduction은 상완 방향만 사용한다. 견갑골 움직임을 별도로 모델링하지는 않는다.
- External rotation은 손목/전완 방향에 의존한다. 팔꿈치가 거의 펴져 있거나 손목 위치 추정이 흔들리면 `F_proj`가 작아져 불안정해질 수 있다.
- 각도는 관절 좌표 기반의 기능적 추정값이다. 임상용 goniometer 또는 marker-based motion capture와 완전히 같은 정의라고 가정하면 안 된다.

## 7. 구현 위치

핵심 구현은 다음 함수에 있다.

- `compute_lcs(joints)`: 프레임별 LCS 구성
- `transform_to_lcs(joints, R_list, origin_list)`: 월드/카메라 좌표를 LCS로 변환
- `compute_angles(joints_local, side)`: flexion, abduction, external rotation 계산

실행 파이프라인에서는 `shoulderlab/analyze.py`가 HSMR 결과를 로드하고 SKEL forward로 3D 관절을 복원한 뒤, 위 함수들을 순서대로 호출한다.
