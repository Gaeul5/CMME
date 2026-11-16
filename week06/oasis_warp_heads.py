"""Week 6-6: OASIS-3 T1 10명으로 '진짜 머리 모양' 리드필드 만들기 (FreeSurfer 없이, 템플릿 변형 방식)

왜: 6-5에서 "자기 머리 리드필드 = 템플릿 머리 리드필드"가 나왔지만, 변형 머리끼리 너무 비슷했다 (상관 0.991).
    FreeSurfer 결과는 연구실 컴퓨터에 있어서 여기서는 쓸 수 없다. 대신 T1만으로 할 수 있는 방법을 쓴다.

방법 (템플릿 변형, template warping)
  1) sample 머리의 T1을 각 OASIS 피험자의 T1에 비선형 정합 (ANTs SyN, 2 mm 해상도)
  2) 그 변형으로 sample의 BEM 표면 3개(두피·두개골 바깥·두개골 안쪽), 피질 source 7,498곳(위치·법선), 전극을 옮김
  3) 옮긴 머리로 forward 계산. week05 heads.py와 같은 형식 (모델: 두개골 0.006, 정답: 0.005~0.015 무작위 + 전극 5 mm 오차)
  → 머리 크기·모양·두피와 두개골 위치는 그 사람의 MRI를 따르고, 피질 주름 모양은 sample을 옮긴 것이다.
    source 번호가 모든 머리에서 같아서 "같은 번호 = 같은 해부학적 위치" 대응이 그대로 유지된다.

품질 확인
  - 옮긴 두피 점이 그 사람 T1의 실제 머리 경계에서 얼마나 떨어져 있는지 (중앙값, 95% 값, mm)
  - 겹쳐 그린 그림 (스크래치 폴더, MRI 단면이라 GitHub에는 올리지 않음)
OASIS-3 약관: 결과 npz에는 리드필드와 피질 위치만 저장 (두피 표면 저장 안 함). npz는 .gitignore 대상.

실행: python week06/oasis_warp_heads.py <T1 폴더> <MNE-sample-data 경로> <출력 폴더> [<그림 폴더>]
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
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week05"))
import heads as H  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
F = mne.io.constants.FIFF
SKULL_MODEL, JITTER_MM = 0.006, 5.0


class AntsWarp:
    """sample MRI 좌표(m, FreeSurfer tkr) -> 피험자 스캐너 좌표(m). heads.Warp와 같은 사용법."""

    def __init__(self, reg, c_ras):
        self.tf, self.inv, self.c = reg["invtransforms"], [True, False], c_ras

    def __call__(self, x):
        p = x * 1000.0 + self.c                         # tkr(m) -> sample 스캐너 RAS(mm)
        lps = p * np.array([-1, -1, 1])                 # ANTs는 LPS
        import pandas as pd
        df = pd.DataFrame(lps, columns=["x", "y", "z"])
        out = ants.apply_transforms_to_points(3, df, self.tf, whichtoinvert=self.inv).values
        return out * np.array([-1, -1, 1]) / 1000.0     # 피험자 스캐너 RAS(m)

    def normals(self, x, n, h=1e-3):
        a, b = self(x - h * n), self(x + h * n)
        # 법선을 그대로 옮기면 안 되지만(J^{-T}), 이웃 점이 필요 없는 근사로 '법선 방향 짧은 선분'을 옮긴 방향을 쓴다.
        m = b - a
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def head_mask(img):
    d = img.get_fdata()
    thr = 0.08 * np.percentile(d, 99)                # 배경 잡음 위, 두피 아래 (OAS30001에서 부피의 35%)
    m = ndimage.binary_closing(d > thr, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return m


def scalp_distance(img, mask, pts_mm):
    """점들(스캐너 RAS mm)과 머리 마스크 경계 사이 거리 (mm). 영상 범위 밖(목 아래 등)에 떨어지는 점은 뺀다."""
    edge = mask ^ ndimage.binary_erosion(mask)
    vox = np.array(np.nonzero(edge)).T
    xyz = nib.affines.apply_affine(img.affine, vox)
    pv = nib.affines.apply_affine(np.linalg.inv(img.affine), pts_mm)
    inside = np.all((pv > 3) & (pv < np.array(img.shape) - 4), axis=1)
    from scipy.spatial import cKDTree
    return cKDTree(xyz).query(pts_mm[inside])[0]


def main(t1_dir, data_path, out_dir, fig_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    P = H.paths(data_path)
    smri = os.path.join(data_path, "subjects", "sample", "mri", "T1.mgz")
    simg = nib.load(smri)
    c_ras = simg.affine[:3, 3] - simg.header.get_vox2ras_tkr()[:3, 3]
    tmp = os.path.join(out_dir, "_sample_T1.nii.gz")
    nib.save(nib.Nifti1Image(np.asanyarray(simg.dataobj).astype(np.float32), simg.affine), tmp)
    moving = ants.resample_image(ants.image_read(tmp), (2, 2, 2), False, 0)
    os.remove(tmp)
    z0 = np.load(os.path.join(H.CACHE0, "head000.npz"))
    base = [z0["vert_lh"], z0["vert_rh"]]
    geom0 = H.make_geometry(0, P, base)                  # 원래 sample 머리 (변형 없음)
    scalp0 = [s for s in geom0["surfs"] if s["id"] == F.FIFFV_BEM_SURF_ID_HEAD][0]["rr"]

    for f in sorted(glob.glob(os.path.join(t1_dir, "*T1w.nii.gz"))):
        sub = os.path.basename(f)[:8]
        out = os.path.join(out_dir, f"{sub}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        img = nib.load(f)
        fixed = ants.resample_image(ants.image_read(f), (2, 2, 2), False, 0)
        reg = ants.registration(fixed, moving, type_of_transform="SyN", random_seed=0)
        W = AntsWarp(reg, c_ras)
        t_reg = time.time() - t

        # 품질 확인: 옮긴 두피가 실제 머리 경계에 놓이는가
        sc = W(scalp0) * 1000.0
        dist = scalp_distance(img, head_mask(img), sc)

        # 기하 만들기 (heads.make_geometry와 같은 순서, Warp만 교체)
        rng = np.random.default_rng(9000 + int(sub[3:]))
        skull_true = rng.uniform(0.005, 0.015)
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
        to_mri = lambda X: X @ h2m[:3, :3].T + h2m[:3, 3]
        to_head = lambda X: (X - h2m[:3, 3]) @ np.linalg.inv(h2m[:3, :3]).T
        E = to_head(W(to_mri(geom0["E"])))
        E_true = E + rng.normal(scale=JITTER_MM / 1000, size=E.shape)
        geom = dict(surfs=new, src=src, trans=trans, names=geom0["names"], mindist=0.0)
        try:
            Gm, vertno, pos = H.forward_for(geom, H.bem_for(geom, SKULL_MODEL), E)
            Gt, _, _ = H.forward_for(geom, H.bem_for(geom, skull_true), E_true)
        except Exception as e:  # 표면이 겹치는 등
            print(f"{sub}: forward 실패 ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        n = sum(len(v) for v in vertno)
        inner = [s for s in new if s["id"] == F.FIFFV_BEM_SURF_ID_BRAIN][0]["rr"]
        size_mm = float(np.cbrt(np.prod(inner.max(0) - inner.min(0))) * 1000)
        np.savez_compressed(out, G_true=Gt.astype(np.float32), G_model=Gm.astype(np.float32), pos=pos,
                            vert_lh=vertno[0], vert_rh=vertno[1], valid=np.ones(len(pos), bool),
                            skull_true=skull_true, thick_mm=0.0, scale=size_mm, elec_mm=E * 1000)
        print(f"{sub}: 정합 {t_reg:.0f} s, 두피 거리 중앙값 {np.median(dist):.1f} mm (95% {np.percentile(dist, 95):.1f}), "
              f"source {n}, 두개골 안쪽 크기 {size_mm:.0f} mm, 정답 전도도 {skull_true:.4f}, 전체 {time.time() - t:.0f} s", flush=True)

        if fig_dir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(fig_dir, exist_ok=True)
            d = img.get_fdata()
            inv = np.linalg.inv(img.affine)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            for k, (axis, name) in enumerate(((0, "sagittal"), (1, "coronal"))):
                mid = d.shape[axis] // 2
                sl = np.take(d, mid, axis=axis)
                ax[k].imshow(sl.T, cmap="gray", origin="lower", vmax=np.percentile(d, 99.5))
                for s, col in zip(new, ("y", "c", "m")):
                    v = nib.affines.apply_affine(inv, s["rr"] * 1000)
                    near = np.abs(v[:, axis] - mid) < 1.5
                    keep = [i for i in range(3) if i != axis]
                    ax[k].plot(v[near, keep[0]], v[near, keep[1]], ".", ms=1.5, color=col)
                ax[k].set_title(f"{sub} {name}")
                ax[k].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{sub}_qc.png"), dpi=70)
            plt.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
