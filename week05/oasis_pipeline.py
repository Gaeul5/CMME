"""Week 5-4: 실제 MRI(OASIS-3 등 FreeSurfer 처리된 피험자)에서 다중 머리 실험용 forward를 만든다.

연구실 서버에서 돌리는 것을 전제로 한 코드다. OASIS-3 데이터는 약관상 승인된 사용자만 접근할 수 있으므로
원본 MRI를 다른 곳(클라우드, GitHub 등)으로 옮기지 말 것. 이 코드의 출력(npz)에는 리드필드와 피질 위치만 들어가고
두피 표면은 저장하지 않는다 (두피 표면은 얼굴 복원에 쓰일 수 있어 약관상 공개 금지인 파생 데이터에 해당).

피험자 한 명 처리 순서
  1) BEM 표면: <subject>/bem/{inner_skull,outer_skull,outer_skin}.surf 가 있으면 사용,
     없으면 mne.bem.make_watershed_bem 로 생성 (FreeSurfer 설치 필요)
  2) source space: fsaverage ico-4(반구당 2,562곳, 총 5,124곳) 정점마다 구면 정합으로 피험자 피질 정점을 찾음
     → 모든 피험자에서 "같은 번호 = fsaverage 상의 같은 위치" 대응이 생김 (multihead.py가 이 대응을 씀)
  3) 전극 정합: EEG 측정 위치가 없으므로 표준 몽타주(biosemi64)를 MRI 두피에 자동 정합
     (MNI 변환으로 추정한 fiducial로 초기 정렬 → 두피 ICP). 전극은 forward 계산 때 두피에 투영됨
  4) forward 두 개: 모델 머리(두개골 0.006 S/m, 정합된 위치)와 정답 머리(두개골 0.005~0.015 무작위,
     전극 위치 오차 5 mm). week05/heads.py의 변형 머리와 같은 형식의 npz로 저장
  두개골 안쪽 면 밖으로 나간 source가 있으면 그 번호를 valid=False로 표시 (multihead.py가 공통 유효 번호만 씀)

실행 예 (연구실 서버):
  python week05/oasis_pipeline.py --subjects-dir $SUBJECTS_DIR --subject OAS30001_MR_d0129 \
      --fsaverage-dir $SUBJECTS_DIR --out heads_oasis/OAS30001.npz --seed 1

이 코드로 확인한 것: MNE sample 피험자 1명을 OASIS 피험자 대신 넣어 끝까지 실행되는 것 (README 참고).
OASIS-3 피험자로는 실행해 보지 않았다.
"""
import argparse
import os
import warnings
import numpy as np
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
F = mne.io.constants.FIFF
N_ICO4 = 2562  # fsaverage에서 번호 0~2561이 ico-4 정점 (ico-5 번호의 앞부분)


def bem_surfaces(subjects_dir, subject, ico=4):
    bem_dir = os.path.join(subjects_dir, subject, "bem")
    need = [os.path.join(bem_dir, f"{n}.surf") for n in ("inner_skull", "outer_skull", "outer_skin")]
    if not all(os.path.exists(p) for p in need):
        mne.bem.make_watershed_bem(subject=subject, subjects_dir=subjects_dir, overwrite=True)  # FreeSurfer 필요
    # 전도도는 나중에 바꿔 끼우므로 여기서는 기본값으로 표면만 만든다
    return mne.make_bem_model(subject, ico=ico, subjects_dir=subjects_dir, conductivity=(0.3, 0.006, 0.3))


def morphed_source_space(subjects_dir, subject, fsaverage_dir):
    """fsaverage ico-4 정점 각각에 대해, 구면 정합(sphere.reg)에서 가장 가까운 피험자 피질 정점을 찾는다.

    반환: (피험자 source space, 반구별 대응표). 대응표[h][k] = fsaverage k번 정점에 대응하는 피험자 정점 번호.
    주의: mne.morph_source_spaces 결과는 정점 번호가 피험자 번호 순으로 정렬되어 fsaverage 순서와의 대응이
    사라진다 (sample로 확인: 같은 위치를 비교하면 최대 120 mm 차이). 그래서 대응표를 직접 만든다.
    """
    from scipy.spatial import cKDTree
    src = mne.setup_source_space(subject, spacing="all", subjects_dir=subjects_dir, add_dist=False)
    maps = []
    for h, hemi in enumerate(("lh", "rh")):
        fs_sph = mne.read_surface(os.path.join(fsaverage_dir, "fsaverage", "surf", f"{hemi}.sphere.reg"))[0]
        sub_sph = mne.read_surface(os.path.join(subjects_dir, subject, "surf", f"{hemi}.sphere.reg"))[0]
        fs_u = fs_sph[:N_ICO4] / np.linalg.norm(fs_sph[:N_ICO4], axis=1, keepdims=True)
        sub_u = sub_sph / np.linalg.norm(sub_sph, axis=1, keepdims=True)
        v = cKDTree(sub_u).query(fs_u)[1]
        maps.append(v)
        s = src[h]
        used = np.unique(v)
        s["inuse"] = np.zeros(s["np"], int)
        s["inuse"][used] = 1
        s["vertno"], s["nuse"] = used, len(used)
        s["use_tris"], s["nuse_tri"] = None, 0
    return src, maps


def standard_info(montage_name):
    mont = mne.channels.make_standard_montage(montage_name)
    info = mne.create_info(mont.ch_names, 1000.0, "eeg")
    info.set_montage(mont)
    return info


def auto_coreg(info, subjects_dir, subject):
    """표준 몽타주를 MRI 두피에 맞춤. 실제 측정 위치가 있으면 이 함수 대신 그 trans를 쓸 것."""
    co = mne.coreg.Coregistration(info, subject, subjects_dir=subjects_dir, fiducials="estimated")
    co.fit_fiducials(verbose=False)
    co.fit_icp(n_iterations=30, nasion_weight=2.0, verbose=False)
    d = co.compute_dig_mri_distances() * 1000
    return co.trans, float(np.median(d)), float(np.max(d))


def forward(info, trans, src, surfs, skull):
    model = [dict(s) for s in surfs]
    for s in model:
        s["sigma"] = {F.FIFFV_BEM_SURF_ID_BRAIN: 0.3, F.FIFFV_BEM_SURF_ID_SKULL: skull, F.FIFFV_BEM_SURF_ID_HEAD: 0.3}[s["id"]]
    bem = mne.make_bem_solution(model)
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=False, eeg=True, mindist=0.0)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=False)
    return fwd


def to_full(fwd, maps):
    """forward 결과를 fsaverage ico-4 순서(반구당 2,562곳)로 다시 배열. 두개골 밖이라 빠진 곳은 NaN, valid=False."""
    n_ch = fwd["sol"]["data"].shape[0]
    G = fwd["sol"]["data"].reshape(n_ch, -1, 3)
    n_full = sum(len(m) for m in maps)
    Gf = np.full((n_ch, n_full, 3), np.nan)
    pos = np.full((n_full, 3), np.nan)
    valid = np.zeros(n_full, bool)
    off_full, off_fwd = 0, 0
    for m, s_fwd in zip(maps, fwd["src"]):
        vno = s_fwd["vertno"]
        j = np.searchsorted(vno, m)
        ok = (j < len(vno)) & (vno[np.minimum(j, len(vno) - 1)] == m)
        k = np.arange(len(m))[ok] + off_full
        Gf[:, k, :] = G[:, off_fwd + j[ok], :]
        pos[k] = fwd["source_rr"][off_fwd + j[ok]] * 1000
        valid[k] = True
        off_full += len(m)
        off_fwd += len(vno)
    return Gf, pos, valid


def skull_thickness_mm(surfs):
    from scipy.spatial import cKDTree
    S = {s["id"]: s["rr"] for s in surfs}
    d, _ = cKDTree(S[F.FIFFV_BEM_SURF_ID_BRAIN]).query(S[F.FIFFV_BEM_SURF_ID_SKULL])
    return float(np.median(d) * 1000)


def build(subjects_dir, subject, out, fsaverage_dir=None, montage="biosemi64", seed=0, jitter_mm=5.0, trans=None):
    rng = np.random.default_rng(seed)
    fsaverage_dir = fsaverage_dir or subjects_dir
    surfs = bem_surfaces(subjects_dir, subject)
    src, maps = morphed_source_space(subjects_dir, subject, fsaverage_dir)
    info = standard_info(montage)
    if trans is None:
        trans, med, mx = auto_coreg(info, subjects_dir, subject)
        print(f"{subject}: 자동 정합 전극-두피 거리 중앙값 {med:.1f} mm, 최대 {mx:.1f} mm")
    else:
        trans = mne.read_trans(trans)
    fwd_m = forward(info, trans, src, surfs, 0.006)
    G_model, pos, valid_m = to_full(fwd_m, maps)

    # 정답 머리: 전극 위치 오차 + 두개골 전도도 무작위
    skull_true = float(rng.uniform(0.005, 0.015))
    pos_head = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["ch_names"]))])
    info_t = info.copy()
    jit = pos_head + rng.normal(scale=jitter_mm / 1000, size=pos_head.shape)
    info_t.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(info["ch_names"], jit)), coord_frame="head",
                                                     **{k: v for k, v in _fids(info).items()}))
    fwd_t = forward(info_t, trans, src, surfs, skull_true)
    G_true, _, valid_t = to_full(fwd_t, maps)

    valid = valid_m & valid_t
    inner = [s for s in surfs if s["id"] == F.FIFFV_BEM_SURF_ID_BRAIN][0]["rr"]
    size_mm = float(np.cbrt(np.prod(inner.max(0) - inner.min(0))) * 1000)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, G_true=G_true.astype(np.float32), G_model=G_model.astype(np.float32), pos=pos,
                        valid=valid, map_lh=maps[0], map_rh=maps[1],
                        subject=subject, skull_true=skull_true, thick_mm=skull_thickness_mm(surfs),
                        scale=size_mm, elec_mm=pos_head * 1000)
    print(f"{subject}: source {valid.sum()}/{len(valid)}곳 유효, 두개골 두께 중앙값 {skull_thickness_mm(surfs):.1f} mm, "
          f"정답 전도도 {skull_true:.4f} → {out}")


def _fids(info):
    d = {}
    for dig in info["dig"]:
        if dig["kind"] == F.FIFFV_POINT_CARDINAL:
            key = {F.FIFFV_POINT_NASION: "nasion", F.FIFFV_POINT_LPA: "lpa", F.FIFFV_POINT_RPA: "rpa"}[dig["ident"]]
            d[key] = dig["r"]
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects-dir", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--fsaverage-dir", default=None, help="fsaverage가 들어 있는 subjects_dir (기본: --subjects-dir)")
    ap.add_argument("--montage", default="biosemi64")
    ap.add_argument("--trans", default=None, help="실제 정합 파일이 있으면 지정 (없으면 자동 정합)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.subjects_dir, a.subject, a.out, a.fsaverage_dir, a.montage, a.seed, trans=a.trans)
