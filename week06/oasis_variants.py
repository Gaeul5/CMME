"""Week 6-7d 준비: OASIS 평가 머리 4명의 '후보 전도도별 리드필드'와 '전극 위치 오차별 정답 리드필드'

week05/heads_variants.py와 같은 내용을 OASIS 템플릿 변형 머리에 대해 만든다 (5-5 실험을 OASIS 머리로 다시 하기 위함).
  - 후보 전도도 6개(0.004~0.015)의 모델 리드필드 (전극은 모델 위치)
  - 그 머리의 진짜 전도도 + 전극 위치 오차 0/5/10/15 mm의 정답 리드필드 (오차 방향은 머리마다 고정)
평가 머리는 고정: OAS30004, 30006, 30008, 30010 (스캐너가 다른 30006, 30010 포함). 나머지 6명이 학습 머리.
표면은 저장하지 않으므로 정합부터 다시 한다 (random_seed=0이라 cache_oasis와 같은 머리가 나옴. pos로 확인).

실행: python week06/oasis_variants.py <T1 폴더> <MNE-sample-data 경로>
"""
import copy
import glob
import os
import sys
import time
import warnings
import numpy as np
import nibabel as nib
import ants
import mne

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "week05"))
import heads as H  # noqa: E402
from oasis_warp_heads import AntsWarp  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
F = mne.io.constants.FIFF
CANDS = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
JITTERS = [0.0, 5.0, 10.0, 15.0]
TEST = ["OAS30004", "OAS30006", "OAS30008", "OAS30010"]
OUT = os.path.join(HERE, "cache_oasis_variants")


def main(t1_dir, data_path):
    os.makedirs(OUT, exist_ok=True)
    P = H.paths(data_path)
    simg = nib.load(os.path.join(data_path, "subjects", "sample", "mri", "T1.mgz"))
    c_ras = simg.affine[:3, 3] - simg.header.get_vox2ras_tkr()[:3, 3]
    tmp = os.path.join(OUT, "_sample_T1.nii.gz")
    nib.save(nib.Nifti1Image(np.asanyarray(simg.dataobj).astype(np.float32), simg.affine), tmp)
    moving = ants.resample_image(ants.image_read(tmp), (2, 2, 2), False, 0)
    os.remove(tmp)
    z0 = np.load(os.path.join(H.CACHE0, "head000.npz"))
    geom0 = H.make_geometry(0, P, [z0["vert_lh"], z0["vert_rh"]])
    for sub in TEST:
        out = os.path.join(OUT, f"head{int(sub[3:]) % 1000:03d}.npz")     # adapt 코드가 head### 이름을 읽음
        if os.path.exists(out):
            continue
        t = time.time()
        f = glob.glob(os.path.join(t1_dir, f"{sub}*T1w.nii.gz"))[0]
        fixed = ants.resample_image(ants.image_read(f), (2, 2, 2), False, 0)
        W = AntsWarp(ants.registration(fixed, moving, type_of_transform="SyN", random_seed=0), c_ras)
        skull_true = np.random.default_rng(9000 + int(sub[3:])).uniform(0.005, 0.015)
        new = []
        for s0 in geom0["surfs"]:
            s = copy.deepcopy(s0)
            s["rr"] = W(s0["rr"])
            new.append(mne.surface.complete_surface_info(s, verbose=False))
        src = copy.deepcopy(geom0["src"])
        for s in src:
            s["nn"] = W.normals(s["rr"], s["nn"])
            s["rr"] = W(s["rr"])
        trans = geom0["trans"]
        h2m = trans["trans"] if trans["from"] == F.FIFFV_COORD_HEAD else np.linalg.inv(trans["trans"])
        E = (W(geom0["E"] @ h2m[:3, :3].T + h2m[:3, 3]) - h2m[:3, 3]) @ np.linalg.inv(h2m[:3, :3]).T
        geom = dict(surfs=new, src=src, trans=trans, names=geom0["names"], mindist=0.0)
        Gc = []
        for c in CANDS:
            G, _, pos = H.forward_for(geom, H.bem_for(geom, c), E)
            Gc.append(G)
        ref = np.load(os.path.join(HERE, "cache_oasis", f"{sub}.npz"))
        d = np.abs(pos - ref["pos"]).max()
        bem_t = H.bem_for(geom, skull_true)
        Z = np.random.default_rng(7000 + int(sub[3:])).normal(size=E.shape)
        Gj = [H.forward_for(geom, bem_t, E + Z * j / 1000)[0] for j in JITTERS]
        np.savez_compressed(out, G_cand=np.array(Gc, np.float32), cands=CANDS, G_jit=np.array(Gj, np.float32),
                            jitters=JITTERS, skull_true=skull_true, pos=pos)
        print(f"{sub}: 진짜 전도도 {skull_true:.4f}, 저장된 머리와 source 위치 차이 최대 {d:.3f} mm, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
