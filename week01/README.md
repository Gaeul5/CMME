# Week 01: MNE 정방향 모델

[`mne_forward.py`](./mne_forward.py)는 MNE 샘플 데이터로 MEG 정방향 모델(forward solution)을 계산하고 `sample-fwd.fif`로 저장합니다.

## 실행 환경 준비

Homebrew Python을 사용하는 경우, 시스템 Python에 직접 패키지를 설치하는 대신 가상 환경을 사용합니다. 아래 명령은 저장소 루트(`CMME`)에서 실행합니다. MNE가 이미 Homebrew Python에 설치되어 있다면 `--system-site-packages` 옵션으로 해당 설치를 가상 환경에서 사용할 수 있습니다.

```bash
/opt/homebrew/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install nibabel
python week01/mne_forward.py
```

`nibabel`은 FreeSurfer 표면 파일을 읽을 때 필요합니다. 가상 환경에 MNE가 없다면 `python -m pip install mne nibabel`로 두 패키지를 설치하세요. 첫 실행 시에는 MNE 샘플 데이터 다운로드가 필요할 수 있습니다.

## 코드 해석

1. `mne.datasets.sample.data_path()`로 샘플 데이터 위치를 찾고, 해부학 자료(`subjects`)와 MEG 측정 자료(`MEG/sample`) 경로를 만듭니다.
2. `mne.setup_source_space()`가 `sample` 피험자의 양쪽 피질 표면에 `oct6` 간격의 소스 후보 위치를 배치합니다. `add_dist=False`는 이때 소스 간 거리 행렬을 만들지 않는 설정입니다.
3. `mne.make_bem_model()`이 전도도 `0.3 S/m`인 균질 단층 BEM 모델을 만들고, `mne.make_bem_solution()`이 정방향 계산에 사용할 BEM 해를 계산합니다. `ico=4`는 BEM 표면의 삼각형 메시 해상도를 지정합니다.
4. `mne.io.read_raw_fif()`가 샘플 측정 파일에서 채널 및 센서 정보 등을 읽습니다. `preload=False`이므로 원시 신호 전체를 메모리에 미리 읽지 않으며, 이 코드에서는 `raw.info`를 정방향 계산에 전달합니다.
5. `mne.make_forward_solution()`이 MRI-머리 좌표 변환(`sample_audvis_raw-trans.fif`), 소스 공간, BEM, 센서 정보를 사용해 MEG 정방향 모델을 계산합니다. `meg=True`, `eeg=False`는 MEG만 계산한다는 뜻이고, `mindist=5.0`은 두개골 안쪽 경계에서 5 mm보다 가까운 소스 후보를 제외합니다.
6. `mne.write_forward_solution()`이 계산 결과 `fwd_own`을 현재 작업 디렉터리의 `sample-fwd.fif` 파일에 저장합니다. 파일이 이미 있으면 덮어씁니다.

## 실행 결과 해석

성공적으로 실행되면 로그에 `Finished.`가 표시되고 `sample-fwd.fif`가 생성됩니다. 샘플 데이터에서 확인된 주요 결과는 다음과 같습니다.

- 양쪽 피질에 각각 4,098개씩, 총 8,196개의 소스 후보 위치가 만들어졌습니다.
- 두개골 안쪽 표면 밖에 있거나 `mindist=5.0` 조건에 걸린 위치를 제외한 뒤, 총 7,498개 위치에서 계산했습니다.
- 측정 정보에서 MEG 채널 306개를 읽었습니다. 계산은 각 위치에서 전류 방향을 자유롭게 두는 `Free source orientations` 방식으로 수행했습니다.
- MRI와 머리 좌표가 변환 파일을 통해 정렬되고 BEM 모델이 설정된 뒤, 소스 위치에서 각 MEG 센서에 생기는 자기장을 계산했습니다.

정방향 모델은 각 소스 위치의 전류가 센서에 어떤 신호를 만들지 나타내는 모델입니다. 따라서 이 결과만으로 실제 뇌 활동 위치를 찾은 것은 아니며, 역해(inverse solution)나 신호 분석은 별도로 수행해야 합니다.

## 깊이에 따른 신호 크기 시각화

정방향 모델을 만든 뒤 저장소 루트에서 다음 명령을 실행하면, 각 소스 위치의 머리 중심으로부터의 거리와 MEG 리드필드 크기를 산점도로 표시합니다.

```bash
python week01/mne_depth.py
```

이 스크립트는 저장소 루트의 `sample-fwd.fif`를 읽습니다. 먼저 위의 `mne_forward.py`를 실행해 해당 파일을 만들어야 합니다.
