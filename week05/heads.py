"""Week 5-3 준비: sample 머리를 변형해 '서로 다른 머리' 여러 개를 만들고 forward를 계산해 저장.

왜: 11/4 미팅 메모 "MRI 50개 규모, 학습에 안 쓴 새 머리에서 확인 / 리드필드를 입력으로" 를
    실제 MRI가 오기 전에 같은 실험 틀로 미리 돌려 보기 위함. 실제 MRI가 오면 이 파일 대신
    build_subject_forward.py 의 결과를 같은 형식(npz)으로 넣으면 된다.

변형 머리 하나 = sample의 BEM 표면 3개, 피질 source 위치·방향, 전극 위치에 같은 공간 변형을 적용한 것
  1) 축별 크기 0.88~1.12배 + 약한 기울임(shear, 최대 0.04)
  2) 매끄러운 국소 변형: 가우시안 혹 6개 (중심은 뇌 주변, 폭 35 mm, 최대 이동 8 mm)
  3) 두개골 두께: 바깥 두개골 면을 법선 방향으로 -1.5~+2.0 mm 이동 (원래 두께 3.5~ mm, 중앙값 5.6 mm)
  4) 정답 머리의 두개골 전도도 0.005~0.015 S/m 무작위, 전극 위치 오차 5 mm (모델 머리는 0.006, 오차 없음)
피질 vertex 번호는 모든 머리에서 같다 (같은 7,498곳). 그래서 "같은 vertex = 같은 해부학적 위치"로 대응됨.

한계: 이 변형은 크기·모양·두께만 바꾼다. 실제 사람 사이의 차이(주름 모양, 뇌 크기 대비 피질 배치)는
      훨씬 크고 vertex 대응도 이렇게 깔끔하지 않다. 결과는 '실제 다중 MRI의 쉬운 버전'으로 읽어야 함.

실행: python week05/heads.py <MNE-sample-data 경로> <머리 개수>   (head000 = 변형 없는 원래 머리)
"""
import os
import sys
import time
import copy
import warnings
import numpy as np
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
STRONG = os.environ.get("STRONG") == "1"   # 1이면 더 크게 변형한 머리 (일반화 한계 확인용, 학습에는 안 씀)
CACHE0 = os.path.join(HERE, "cache_heads")
CACHE = os.path.join(HERE, "cache_heads_strong") if STRONG else CACHE0
SCALE_RANGE = (0.80, 1.20) if STRONG else (0.88, 1.12)
BUMP_MAX, BUMP_SIG = (0.014, 0.030) if STRONG else (0.008, 0.035)
THICK_RANGE = (-0.0015, 0.0030) if STRONG else (-0.0015, 0.0020)
SKULL_MODEL, JITTER_MM = 0.006, 5.0
F = mne.io.constants.FIFF


def paths(data_path):
    j = lambda p: os.path.join(data_path, p)
    return dict(bem=j("subjects/sample/bem/sample-5120-5120-5120-bem.fif"),
                src=j("subjects/sample/bem/sample-oct-6-src.fif"),
                raw=j("MEG/sample/sample_audvis_raw.fif"),
                trans=j("MEG/sample/sample_audvis_raw-trans.fif"))


def biosemi_on_sample(raw_file):
    """Week 4와 같은 방식: Biosemi 64 표준 위치를 sample 머리 크기에 맞춤 (머리 좌표계, m)."""
    std = mne.channels.make_standard_montage("biosemi64")
    P = np.array([std.get_positions()["ch_pos"][n] for n in std.ch_names])
    info = mne.io.read_info(raw_file)
    idx = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
    samp = np.array([info["chs"][i]["loc"][:3] for i in idx])
    sc = np.median(np.linalg.norm(samp - samp.mean(0), axis=1)) / np.median(np.linalg.norm(P - P.mean(0), axis=1))
    return std.ch_names, (P - P.mean(0)) * sc + samp.mean(0)


class Warp:
    """x(m, MRI 좌표) -> 변형된 x. 법선은 야코비안의 역전치로 변환."""

    def __init__(self, rng, center, identity=False):
        self.c = center
        if identity:
            self.A = np.eye(3)
            self.mu, self.amp = np.zeros((0, 3)), np.zeros((0, 3))
            self.sig = 0.04
            return
        S = np.diag(rng.uniform(*SCALE_RANGE, 3))
        Sh = np.eye(3) + np.triu(rng.uniform(-0.04, 0.04, (3, 3)), 1)
        self.A = S @ Sh
        d = rng.normal(size=(6, 3))
        self.mu = center + 0.07 * d / np.linalg.norm(d, axis=1, keepdims=True)
        a = rng.normal(size=(6, 3))
        self.amp = a / np.linalg.norm(a, axis=1, keepdims=True) * rng.uniform(0, BUMP_MAX, (6, 1))
        self.sig = BUMP_SIG

    def __call__(self, x):
        y = self.c + (x - self.c) @ self.A.T
        if len(self.mu):
            w = np.exp(-((x[:, None, :] - self.mu[None]) ** 2).sum(-1) / (2 * self.sig**2))  # (n, 6)
            y = y + w @ self.amp
        return y

    def normals(self, x, n, h=1e-4):
        J = np.zeros((len(x), 3, 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            J[:, :, k] = (self(x + e) - self(x - e)) / (2 * h)
        m = np.einsum("nji,nj->ni", np.linalg.inv(J), n)  # J^{-T} n
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def make_head(h, P, base_vertno=None):
    """머리 h의 (정답 forward, 모델 forward, source 위치 mm, 정보) 반환. h=0은 변형 없는 원래 머리."""
    rng = np.random.default_rng((5000 if STRONG else 1000) + h)
    surfs = {s["id"]: s for s in mne.read_bem_surfaces(P["bem"])}
    src0 = mne.read_source_spaces(P["src"])
    trans = mne.read_trans(P["trans"])
    names, E_head = biosemi_on_sample(P["raw"])
    head_to_mri = trans["trans"] if trans["from"] == F.FIFFV_COORD_HEAD else np.linalg.inv(trans["trans"])
    to_mri = lambda X: X @ head_to_mri[:3, :3].T + head_to_mri[:3, 3]
    to_head = lambda X: (X - head_to_mri[:3, 3]) @ np.linalg.inv(head_to_mri[:3, :3]).T

    W = Warp(rng, surfs[F.FIFFV_BEM_SURF_ID_BRAIN]["rr"].mean(0), identity=(h == 0))
    thick = 0.0 if h == 0 else rng.uniform(*THICK_RANGE)
    skull_true = 0.010 if h == 0 else rng.uniform(0.005, 0.015)

    new = []
    for sid in (F.FIFFV_BEM_SURF_ID_HEAD, F.FIFFV_BEM_SURF_ID_SKULL, F.FIFFV_BEM_SURF_ID_BRAIN):
        s = copy.deepcopy(surfs[sid])
        s["rr"] = W(s["rr"])
        s = mne.surface.complete_surface_info(s, verbose=False)
        if sid == F.FIFFV_BEM_SURF_ID_SKULL and thick != 0:
            s["rr"] = s["rr"] + thick * s["nn"]
            s = mne.surface.complete_surface_info(s, verbose=False)
        new.append(s)

    src = copy.deepcopy(src0)
    for k, s in enumerate(src):
        s["nn"] = W.normals(s["rr"], s["nn"])
        s["rr"] = W(s["rr"])
        if base_vertno is not None:  # 모든 머리에서 같은 vertex만 사용
            s["inuse"] = np.zeros(s["np"], int)
            s["inuse"][base_vertno[k]] = 1
            s["vertno"] = np.array(base_vertno[k])
            s["nuse"] = len(base_vertno[k])
    E = to_head(W(to_mri(E_head)))
    E_true = E + np.random.default_rng((6000 if STRONG else 2000) + h).normal(scale=JITTER_MM / 1000, size=E.shape)

    out = {}
    for tag, sk, pos_e in (("true", skull_true, E_true), ("model", SKULL_MODEL, E)):
        surfs_c = copy.deepcopy(new)
        for s in surfs_c:
            s["sigma"] = {F.FIFFV_BEM_SURF_ID_BRAIN: 0.3, F.FIFFV_BEM_SURF_ID_SKULL: sk, F.FIFFV_BEM_SURF_ID_HEAD: 0.3}[s["id"]]
        bem = mne.make_bem_solution(surfs_c)
        info = mne.create_info(names, 1000.0, "eeg")
        info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, pos_e)), coord_frame="head"))
        fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=False, eeg=True,
                                        mindist=0.0 if base_vertno is not None else 5.0)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=False)  # 모든 머리에서 같은 기준(vertex 법선)
        G = fwd["sol"]["data"].reshape(len(names), -1, 3)
        out[tag] = G
        out["vertno"] = [s["vertno"] for s in fwd["src"]]
        out["pos"] = fwd["source_rr"] * 1000
    scale = np.cbrt(abs(np.linalg.det(W.A)))
    return out, dict(skull_true=skull_true, thick_mm=thick * 1000, scale=scale, elec_mm=E * 1000)


def build(data_path, n_heads, start=1):
    os.makedirs(CACHE, exist_ok=True)
    P = paths(data_path)
    f0 = os.path.join(CACHE0, "head000.npz")   # 기준 머리는 항상 기본 폴더에 둠
    if not os.path.exists(f0):
        o, meta = make_head(0, P)
        np.savez_compressed(f0, G_true=o["true"].astype(np.float32), G_model=o["model"].astype(np.float32),
                            pos=o["pos"], vert_lh=o["vertno"][0], vert_rh=o["vertno"][1], **meta)
    z = np.load(f0)
    base = [z["vert_lh"], z["vert_rh"]]
    for h in range(start, n_heads):
        f = os.path.join(CACHE, f"head{h:03d}.npz")
        if os.path.exists(f):
            continue
        t = time.time()
        o, meta = make_head(h, P, base)
        n = sum(len(v) for v in o["vertno"])
        if n != len(z["pos"]):
            print(f"head{h:03d}: source {n}개 (기준 {len(z['pos'])}) → 건너뜀")
            continue
        np.savez_compressed(f, G_true=o["true"].astype(np.float32), G_model=o["model"].astype(np.float32),
                            pos=o["pos"], vert_lh=o["vertno"][0], vert_rh=o["vertno"][1], **meta)
        print(f"head{h:03d}: 크기 {meta['scale']:.3f}배, 두개골 두께 {meta['thick_mm']:+.1f} mm, "
              f"정답 전도도 {meta['skull_true']:.4f}, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
