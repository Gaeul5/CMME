"""Week 5-2: 머리(MRI)가 여러 개일 때 쓰는 forward 계산 도구. 11/4 미팅 메모의
"다른 MRI 데이터로 해 보기 (50개 규모) / MRI에서 리드필드를 얻어 입력으로 쓰기" 준비용.

피험자 한 명의 FreeSurfer 결과(subjects_dir/<subject>)에서
  - BEM 표면(없으면 make_watershed_bem으로 만듦, FreeSurfer 필요)
  - 피질 source space (oct-6)
  - 주어진 전극 이름·위치(머리 좌표계)에 대한 EEG leadfield (채널, source, [접선1, 접선2, 수직])
을 만들어 npz로 저장한다. 전극 위치는 피험자마다 다를 수 있으므로 인자로 받는다.

주의 (이 코드가 하지 않는 것)
  - T1 MRI에서 FreeSurfer 표면을 만드는 recon-all은 하지 않음 (피험자당 수 시간, 별도 설치 필요).
    IXI 같은 T1만 있는 데이터는 이 전처리가 먼저 필요함. 교수님이 주시는 MRI가 이미 처리된 것인지 확인할 것.
  - MRI-전극 좌표 정합(trans)은 인자로 받음. 피험자마다 정합 파일을 따로 만들어야 하고, 이 코드는 만들지 않음.
    표준 몽타주 위치를 그대로 쓰면 머리 크기가 다른 피험자에서는 전극이 두피에서 떠 있거나 파묻힘
    (Week 4는 sample 머리 크기에 맞춰 키웠음). 피험자마다 이 보정이 필요함.
  - 현재 sample 피험자 1명에서만 동작을 확인함. 다른 머리에서의 동작은 확인하지 않음.

실행 예:
  python week05/build_subject_forward.py --subjects-dir <dir> --subject sample --montage biosemi64 \
      --out week05/forward_sample.npz --bem-conductivity 0.3 0.006 0.3 --trans <sample-trans.fif>
"""
import argparse
import os
import warnings
import numpy as np
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)


def build(subjects_dir, subject, montage_name, trans, conductivity, out, spacing="oct6", mindist=5.0):
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    fifs = sorted(f for f in os.listdir(bem_dir) if f.endswith("-bem.fif")
                  and "sol" not in f) if os.path.isdir(bem_dir) else []
    has_surf = os.path.exists(os.path.join(bem_dir, "inner_skull.surf"))
    if has_surf:
        model = mne.make_bem_model(subject, subjects_dir=subjects_dir, conductivity=conductivity)
    elif fifs:
        # 표면 파일(.surf)은 없고 -bem.fif만 있는 경우 (MNE sample 배포본 등). 전도도는 표면 id로 지정
        model = mne.read_bem_surfaces(os.path.join(bem_dir, fifs[-1]))
        F = mne.io.constants.FIFF
        sigma = {F.FIFFV_BEM_SURF_ID_BRAIN: conductivity[0], F.FIFFV_BEM_SURF_ID_SKULL: conductivity[1],
                 F.FIFFV_BEM_SURF_ID_HEAD: conductivity[2]}
        for s in model:
            s["sigma"] = sigma[s["id"]]
    else:
        # FreeSurfer가 설치돼 있어야 함 (mri_watershed). 없으면 여기서 실패함.
        mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True)
        model = mne.make_bem_model(subject, subjects_dir=subjects_dir, conductivity=conductivity)
    bem = mne.make_bem_solution(model)
    src_file = os.path.join(bem_dir, f"{subject}-oct-6-src.fif")
    if os.path.exists(src_file):
        src = mne.read_source_spaces(src_file)          # 이미 만들어 둔 source space 재사용
    else:                                               # 피질 표면(surf/lh.white 등)이 있어야 함
        src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False)

    std = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(std.ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(std)
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=False, eeg=True, mindist=mindist)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)
    n_ch = fwd["sol"]["data"].shape[0]
    G = fwd["sol"]["data"].reshape(n_ch, -1, 3)
    np.savez_compressed(out, G=G.astype(np.float32), pos_mm=(fwd["source_rr"] * 1000).astype(np.float32),
                        ch_names=np.array(info.ch_names), subject=subject, conductivity=np.array(conductivity))
    print(f"{subject}: G {G.shape}, source {G.shape[1]}개 → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-dir", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--montage", default="biosemi64")
    ap.add_argument("--trans", required=True, help="MRI-머리 좌표 변환 파일 (-trans.fif)")
    ap.add_argument("--bem-conductivity", type=float, nargs=3, default=[0.3, 0.006, 0.3],
                    help="뇌 두개골 두피 순서 (MNE 기본값)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.subjects_dir, a.subject, a.montage, a.trans, tuple(a.bem_conductivity), a.out)
