"""Week 9-1: Parmigiani 외 2022 피험자 36명의 템플릿 변형 머리 (week08 lmi_warp_heads.py와 같은 방식)

sample T1 → 피험자 T1 비선형 정합(ANTs SyN, 2 mm) → sample BEM 표면과 피질 7,498곳을 옮김 → 피험자의 실제 두피 전극 185개로 forward (두개골 0.006)
좌표: 전극 파일은 T1w 좌표 mm, SEEG 접점 파일은 T1w 좌표 m. T1w = T1 영상의 스캐너 좌표로 가정하고, 전극이 두피 표면 근처에 오는지로 확인
저장: week09/cache_par/sub-XX.npz (GitHub에 올리지 않음)

실행: python week09/par_warp_heads.py <reduced 폴더> <MNE-sample-data 경로>
"""
import copy
import csv
import glob
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
import heads as H  # noqa: E402
from oasis_warp_heads import AntsWarp, head_mask, scalp_distance  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
F = mne.io.constants.FIFF
OUT = os.path.join(HERE, "cache_par")


def main(red, data_path):
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
    for d in sorted(glob.glob(os.path.join(red, "sub-*"))):
        sub = os.path.basename(d)
        out = os.path.join(OUT, f"{sub}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        f = os.path.join(d, f"{sub}_T1w.nii")
        img = nib.load(f)
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
        with open(glob.glob(os.path.join(d, "*_task-ccepcoreg_electrodes.tsv"))[0]) as fh:
            rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["x"] not in ("n/a", "")]
        names = [r["name"] for r in rows]
        E = np.array([[float(r[k]) for k in "xyz"] for r in rows]) / 1000.0           # T1w(스캐너) m
        scalp = [x for x in new if x["id"] == F.FIFFV_BEM_SURF_ID_HEAD][0]["rr"]
        d_warp = cKDTree(scalp).query(E)[0] * 1000
        d_mri = scalp_distance(img, head_mask(img), E * 1000)
        geom = dict(surfs=new, src=src, trans=trans, names=names, mindist=0.0)
        try:
            G, vertno, pos = H.forward_for(geom, H.bem_for(geom, 0.006), to_head(E))
        except Exception as e:
            print(f"{sub}: forward 실패 ({type(e).__name__}: {str(e)[:100]})", flush=True)
            continue
        np.savez_compressed(out, G=G.astype(np.float32), pos_t1w_mm=to_mri(pos / 1000.0) * 1000.0, names=np.array(names),
                            elec_scalp_mm=d_mri)
        print(f"{sub}: 전극 {len(names)}개, 전극-옮긴 두피 {np.median(d_warp):.1f} mm, 전극-MRI 머리 경계 {np.median(d_mri):.1f} mm (95% {np.percentile(d_mri, 95):.1f}), {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
