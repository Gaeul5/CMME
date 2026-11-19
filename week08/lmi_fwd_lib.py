"""Localize-MI 피험자 머리로 순방향 모델 다시 만들기 (전도도·전극 위치를 바꿔 보기 위함)

표면: derivatives/sourcemodelling/sub-XX/anat의 inner_skull, outer_skull, outer_skin (FreeSurfer 표면 좌표, mm)
source: 데이터의 순방향 모델에 들어 있는 source 공간 (위치·법선 그대로)
좌표: 머리 좌표 = T1w = 표면 좌표 + mri_head_t 이동량. 전극은 T1w(머리 좌표)
"""
import copy
import os
import numpy as np
import nibabel as nib
import mne

mne.set_log_level("ERROR")
F = mne.io.constants.FIFF


def setup(root, sub):
    fwd = mne.read_forward_solution(os.path.join(root, f"derivatives/sourcemodelling/{sub}/fwd/{sub}_fwd.fif"))
    trans = fwd["mri_head_t"]
    surfs = []
    for name, sid in (("outer_skin", F.FIFFV_BEM_SURF_ID_HEAD), ("outer_skull", F.FIFFV_BEM_SURF_ID_SKULL), ("inner_skull", F.FIFFV_BEM_SURF_ID_BRAIN)):
        g = nib.load(os.path.join(root, f"derivatives/sourcemodelling/{sub}/anat/{sub}_{name}.surf.gii"))
        rr = g.darrays[0].data.astype(np.float64) / 1000.0
        tris = g.darrays[1].data.astype(int)
        s = dict(id=sid, rr=rr, tris=tris, ntri=len(tris), np=len(rr), coord_frame=F.FIFFV_COORD_MRI, sigma=1.0)
        s = mne.bem._ico_downsample(s, 4)              # 10,242점 → 2,562점 (삼각형 5,120개, Mikulan 외와 같은 해상도. 메모리 때문)
        s.update(id=sid, coord_frame=F.FIFFV_COORD_MRI, sigma=1.0)
        surfs.append(mne.surface.complete_surface_info(s, verbose=False))
    src = copy.deepcopy(fwd["src"])
    t = trans["trans"][:3, 3]                                    # mri → head 이동량 (회전 없음)
    assert np.allclose(trans["trans"][:3, :3], np.eye(3), atol=1e-6)
    for s in src:
        s["rr"] = s["rr"] - t
        s["coord_frame"] = F.FIFFV_COORD_MRI
    return dict(fwd=fwd, trans=trans, surfs=surfs, src=src, names=fwd["info"]["ch_names"])


_BEM = {}


def forward(S, E_head, skull=0.006, scalp=0.3, brain=0.3):
    key = (id(S), skull, scalp, brain)
    if key not in _BEM:
        _BEM.clear()                                   # BEM 해는 하나에 약 0.5 GB라 하나만 둠
        surfs = copy.deepcopy(S["surfs"])
        for s in surfs:
            s["sigma"] = {F.FIFFV_BEM_SURF_ID_BRAIN: brain, F.FIFFV_BEM_SURF_ID_SKULL: skull, F.FIFFV_BEM_SURF_ID_HEAD: scalp}[s["id"]]
        _BEM[key] = mne.make_bem_solution(surfs)
    bem = _BEM[key]
    info = mne.create_info(S["names"], 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(S["names"], E_head)), coord_frame="head"))
    fwd = mne.make_forward_solution(info, trans=S["trans"], src=S["src"], bem=bem, meg=False, eeg=True, mindist=0.0)
    return fwd["sol"]["data"].reshape(len(S["names"]), -1, 3)


def bem_solution(S, skull=0.006, scalp=0.3, brain=0.3):
    surfs = copy.deepcopy(S["surfs"])
    for s in surfs:
        s["sigma"] = {F.FIFFV_BEM_SURF_ID_BRAIN: brain, F.FIFFV_BEM_SURF_ID_SKULL: skull, F.FIFFV_BEM_SURF_ID_HEAD: scalp}[s["id"]]
    return mne.make_bem_solution(surfs)


def dipole_fields(S, E_head, bem, pos_head, ori):
    """임의 위치(머리 좌표, m)·방향의 쌍극자가 만드는 전극 값 (n, 채널). 피질 위가 아니어도 됨"""
    info = mne.create_info(S["names"], 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(S["names"], E_head)), coord_frame="head"))
    n = len(pos_head)
    dip = mne.Dipole(np.arange(n) * 1e-3, pos_head, np.ones(n) * 1e-8, ori, np.ones(n))
    fwd, stc = mne.make_forward_dipole(dip, bem, info, trans=S["trans"], verbose=False)
    G = fwd["sol"]["data"]                                     # (채널, n) — 방향 고정
    return G.T
