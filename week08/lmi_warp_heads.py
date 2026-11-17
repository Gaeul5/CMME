"""Week 8-2 준비 (경로 1): Localize-MI 7명에 대해 '템플릿 변형 머리' 만들기 (OASIS와 같은 방식, week06 oasis_warp_heads.py)

1) sample T1 → 피험자 T1 비선형 정합 (ANTs SyN, 2 mm)
2) sample의 BEM 표면 3개와 피질 7,498곳을 그 변형으로 옮김 (source 번호가 OASIS 머리와 같아 모델 출력 좌표를 그대로 쓸 수 있음)
3) 전극은 옮기지 않고 그 피험자의 실제 256개 전극 위치(T1w 좌표)를 씀
4) forward: 두개골 0.006 (Mikulan 외와 같은 값)
품질 확인: 실제 전극이 옮긴 두피 표면에서 얼마나 떨어져 있는지 (mm)
저장: week08/cache_lmi/sub-XX.npz (리드필드, source 위치(T1w mm), 전극 이름). GitHub에는 올리지 않음.

실행: python week08/lmi_warp_heads.py "<Localize MI 폴더>" <MNE-sample-data 경로>
"""
import copy
import os
import sys
import time
import warnings
import numpy as np
import nibabel as nib
import ants
import mne
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "week05"))
sys.path.insert(0, os.path.join(HERE, "..", "week06"))
sys.path.insert(0, HERE)
import heads as H  # noqa: E402
from oasis_warp_heads import AntsWarp  # noqa: E402
import lmi_common as C  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
F = mne.io.constants.FIFF
OUT = os.path.join(HERE, "cache_lmi")


def main(root, data_path):
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
    trans = geom0["trans"]
    h2m = trans["trans"] if trans["from"] == F.FIFFV_COORD_HEAD else np.linalg.inv(trans["trans"])
    to_head = lambda X: (X - h2m[:3, 3]) @ np.linalg.inv(h2m[:3, :3]).T
    to_mri = lambda X: X @ h2m[:3, :3].T + h2m[:3, 3]
    for s in range(1, 8):
        sub = f"sub-0{s}"
        out = os.path.join(OUT, f"{sub}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        f = os.path.join(root, sub, "anat", f"{sub}_T1w.nii")
        fixed = ants.resample_image(ants.image_read(f), (2, 2, 2), False, 0)
        W = AntsWarp(ants.registration(fixed, moving, type_of_transform="SyN", random_seed=0), c_ras)
        new = []
        for s0 in geom0["surfs"]:
            sf = copy.deepcopy(s0)
            sf["rr"] = W(s0["rr"])
            new.append(mne.surface.complete_surface_info(sf, verbose=False))
        src = copy.deepcopy(geom0["src"])
        for sp in src:
            sp["nn"] = W.normals(sp["rr"], sp["nn"])
            sp["rr"] = W(sp["rr"])
        names, E = C.scalp_xyz(root, sub)                       # T1w(스캐너) m
        scalp = [x for x in new if x["id"] == F.FIFFV_BEM_SURF_ID_HEAD][0]["rr"]
        d = cKDTree(scalp).query(E)[0] * 1000
        geom = dict(surfs=new, src=src, trans=trans, names=names, mindist=0.0)
        G, vertno, pos = H.forward_for(geom, H.bem_for(geom, 0.006), to_head(E))
        pos_t1 = to_mri(pos / 1000.0) * 1000.0                   # 머리 좌표 mm → T1w mm
        np.savez_compressed(out, G=G.astype(np.float32), pos_t1w_mm=pos_t1, names=np.array(names))
        print(f"{sub}: 전극-두피 거리 중앙값 {np.median(d):.1f} mm (95% {np.percentile(d, 95):.1f}), source {len(pos)}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
