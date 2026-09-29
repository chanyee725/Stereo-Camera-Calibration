# Stereo Camera Calibration

ChArUco 보드로 스테레오 카메라를 캘리브레이션하는 도구입니다.
각 카메라의 내부 파라미터를 먼저 구하고, 이를 고정한 채 두 카메라 사이의 자세와 rectification을 계산합니다.

## 설치

```bash
uv sync
```

## 설정

`configs/config.yaml`에서 카메라와 보드를 설정합니다.

- `camera`: 입력 방식(`dual` / `side_by_side`), 장치 경로, 해상도, FPS
- `calibration`: ChArUco 보드 규격, 최소 코너 수, 예상 baseline

## 사용법

### 1. 이미지 촬영

```bash
uv run src/acquire_image.py            # 새 세션
uv run src/acquire_image.py --resume   # 최근 세션에 이어서 촬영
```

| 키 | 동작 |
|---|---|
| `s` | 좌/우 이미지를 한 쌍으로 저장 (`pairs.csv`에 기록) |
| `l` / `r` | 왼쪽 / 오른쪽 이미지만 저장 |
| `q` | 종료 |

코너가 `min_corners`보다 적게 검출된 이미지는 저장하지 않습니다.
결과는 `outputs/<날짜:시간>/` 아래에 저장됩니다.

### 2. 카메라별 캘리브레이션

```bash
uv run src/calibrate.py mono --side left
uv run src/calibrate.py mono --side right
```

### 3. 스테레오 캘리브레이션

```bash
uv run src/calibrate.py stereo
```

모든 명령은 기본으로 가장 최근 세션을 사용합니다. 다른 세션은 `--session <경로>`로 지정합니다.

## 결과

`<session>/calib/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `left.yaml`, `right.yaml` | 내부 파라미터(K, D), 뷰별 재투영 오차 |
| `*_coverage.png` | 검출된 코너의 화면 분포 |
| `*_undistort.png` | 왜곡 보정 전/후 비교 |
| `stereo.yaml` | R, T, E, F, rectification(R1, R2, P1, P2, Q) |
| `rectified_check.png` | rectify된 이미지 쌍 + 수평선 (정렬 확인용) |

## 구조

```
src/
├── acquire_image.py    # 촬영
├── calibrate.py        # 캘리브레이션 실행
├── calibration/        # Calibration, MonoCalibration, StereoCalibration
└── utils/              # 보드 검출, 세션 관리, 지표, 그리기, 상수
```
