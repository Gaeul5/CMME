"""Week 8-2: Localize-MI 실제 기록에서 학습 모델 두 경로 비교 (피험자 7명, run 61개)

경로 1 (정합 + 범용 어댑터): 피험자 T1에 sample 머리를 정합한 '템플릿 변형 머리'(lmi_warp_heads.py)를 쓴다.
  - 학습: OASIS 머리 10명(64채널 안 무작위 배치) + '다른' Localize-MI 피험자 6명의 변형 머리(256채널 안 무작위 배치). 실제 기록은 학습에 안 씀
  - 평가: 남은 1명의 실제 기록 (피험자 하나씩 빼며 7번 반복)
  - 출력: 기준 머리 좌표 → 가장 가까운 기준 source 번호 → 그 번호의 피험자 머리 안 위치 (번호가 모든 변형 머리에서 같음)
경로 2 (피험자 전용 모델): 데이터에 들어 있는 그 피험자의 순방향 모델(8,196곳, 자유 방향)로 가짜 신호를 만들어 그 사람 전용 모델을 학습
  - 입력 두 가지: 어댑터(그 사람 source 공간에서 만든 PCA 256), 센서만(그 사람의 모든 run에서 좋은 채널만)
  - 출력: 그 사람 머리 좌표 → 가장 가까운 source
  - 실제 기록의 정답은 학습에 안 씀
비교: 같은 머리 모델 위의 쌍극자 스캔, sLORETA (경로 1은 변형 머리, 경로 2는 데이터의 순방향 모델)
신호: 자극 펄스 ±2 ms에서 GFP가 가장 큰 한 시점 (Mikulan 외와 같은 방식), 백색화 없음 (8-1에서 백색화가 오히려 나빴음)
학습 신호: 쌍극자 1개, 방향 무작위(0~90도 기울기, 부호 무작위), 흰 잡음 10~40 dB (자극 펄스는 신호가 매우 큼)
오차: 추정 위치 → 자극 접점 쌍의 중점 (Mikulan 외와 같은 정의)

실행: SEED=0 python week08/lmi_routes.py "<Localize MI 폴더>"
"""
import csv
import glob
import os
import sys
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmi_common as C  # noqa: E402

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(12000 + SEED)
root = sys.argv[1]
N_TR = 12000 if QUICK else 60000
N_PCA = 256
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=30 if QUICK else 80, early_stopping=True, random_state=SEED)
SUBS = [f"sub-0{s}" for s in range(1, 8)]
ONLY = os.environ.get("ROUTE")                     # "1", "2" 또는 둘 다


def ref_(X):
    return X - X.mean(-1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-30)


def slor_kernel(Gn, lam2=1.0 / 9.0):
    """Gn (k, V) 한 방향 리드필드 → sLORETA 커널 (V, k), 평균 기준"""
    k = Gn.shape[0]
    R = np.eye(k) - 1.0 / k
    L = R @ Gn
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K = Lw.T @ A @ W
    return K / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


def dirs(n, local):
    """local=True: 피질 법선 기준 좌표(마지막 축이 법선)에서 0~90도. False: 완전 무작위 방향"""
    v = rng.normal(size=(n, 3))
    v = unit(v)
    if local:
        v[:, 2] = np.abs(v[:, 2])
    return v


def simulate(G3, n, local):
    idx = rng.integers(0, G3.shape[1], n)
    clean = (rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n))[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], dirs(n, local))
    r = ref_(clean)
    snr = rng.uniform(10, 40, (n, 1))
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (snr / 10)), idx


def scan_idx(u, G3):
    A = np.transpose(G3, (1, 0, 2))
    Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
    return int(((np.einsum("vkj,k->vj", Q, u)) ** 2).sum(1).argmax())


def signal(Y, good):
    Yg = ref_(Y[good].T).T[:, C.tidx(*C.WIN)]
    u = Yg[:, np.argmax((Yg**2).sum(0))]
    return u / np.linalg.norm(u)


RUNS = C.runs(root)
FWD = {s: C.load_fwd(root, s) for s in SUBS}
CX = {s: C.contacts_xyz(root, s) for s in SUBS}
GOOD = {}
for r in RUNS:
    G, pos, names, bads = FWD[r["sub"]]
    GOOD[(r["sub"], r["run"])] = np.array([n not in (bads | C.bad_channels(root, r["sub"], r["run"])) for n in names])
MID = {(r["sub"], r["run"]): (CX[r["sub"]][r["contacts"][0]] + CX[r["sub"]][r["contacts"][1]]) / 2 * 1000 for r in RUNS}   # mm
rows = []
t0 = time.time()

# ------------------------------------------------------------ 경로 1
if ONLY in (None, "1"):
    oas = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "..", "week06", "cache_oasis", "OAS*.npz")))]
    lmi = {s: np.load(os.path.join(HERE, "cache_lmi", f"{s}.npz")) for s in SUBS}
    z0 = np.load(os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz"))
    pos0, G0 = z0["pos"], z0["G_model"].astype(np.float64)
    tree0 = cKDTree(pos0)
    maps = []
    for _ in range(40 if QUICK else 150):
        sel = np.sort(rng.choice(64, rng.integers(8, 65), replace=False))
        Y, _ = simulate(G0[sel], 40, True)
        maps.append(unit(ref_(Y) @ slor_kernel(G0[sel][:, :, 2]).T))
    B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
    del maps
    heads_tr = [(z["G_model"].astype(np.float64), 64, 8) for z in oas]
    print(f"[경로 1] 기준 PCA {time.time() - t0:.0f} s", flush=True)
    for test in SUBS:
        X, I = [], []
        hs = heads_tr + [(lmi[s]["G"].astype(np.float64), 256, 32) for s in SUBS if s != test]
        per = N_TR // len(hs)
        for G3, nch, kmin in hs:
            Y, idx = simulate(G3, per, True)
            for _ in range(10):                                      # 머리마다 무작위 배치 10개
                ii = np.arange(_, per, 10)
                sel = np.sort(rng.choice(nch, rng.integers(kmin, nch + 1), replace=False))
                X.append(unit(ref_(Y[ii][:, sel]) @ (B @ slor_kernel(G3[sel][:, :, 2])).T))
                I.append(idx[ii])
        m = MLPRegressor(**MLP_KW).fit(np.vstack(X), pos0[np.concatenate(I)] / 100.0)
        del X
        Gw = lmi[test]["G"].astype(np.float64)
        pw = lmi[test]["pos_t1w_mm"]
        for r in [x for x in RUNS if x["sub"] == test]:
            good = GOOD[(test, r["run"])]
            u = signal(C.evoked(root, test, r["run"]), good)
            Ks = slor_kernel(Gw[good][:, :, 2])
            v_ad = tree0.query(m.predict(unit((u @ (B @ Ks).T)[None]))[0] * 100.0)[1]
            ests = {"경로 1: 어댑터 (변형 머리)": v_ad, "쌍극자 스캔 (변형 머리)": scan_idx(u, Gw[good]),
                    "sLORETA (변형 머리)": int(np.abs(Ks @ u).argmax())}
            for k, v in ests.items():
                rows.append([SEED, test, r["run"], "-".join(r["contacts"]), r["mA"], k, f"{np.linalg.norm(pw[v] - MID[(test, r['run'])]):.2f}"])
        rr = [x for x in rows if x[1] == test]
        print(f"[경로 1] {test}: " + "  ".join(f"{k} {np.median([float(x[6]) for x in rr if x[5] == k]):.1f}" for k in dict.fromkeys(x[5] for x in rr)) + f"  ({time.time() - t0:.0f} s)", flush=True)

# ------------------------------------------------------------ 경로 2
if ONLY in (None, "2"):
    for sub in SUBS:
        G, pos, names, bads = FWD[sub]
        pos_mm = pos * 1000
        tree = cKDTree(pos_mm)
        keys = [k for k in GOOD if k[0] == sub]
        always = np.all([GOOD[k] for k in keys], axis=0)             # 그 사람의 모든 run에서 좋은 채널
        allg = np.array([n not in bads for n in names])
        maps = []
        for _ in range(40 if QUICK else 150):
            sel = np.where(allg)[0]
            sel = np.sort(rng.choice(sel, rng.integers(len(sel) // 2, len(sel) + 1), replace=False))
            Y, _ = simulate(G[sel], 40, False)
            Gn = np.linalg.norm(G[sel], axis=2) * 0 + G[sel][:, :, 2]
            maps.append(unit(ref_(Y) @ slor_kernel(Gn).T))
        Bs = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
        del maps
        Y, idx = simulate(G, N_TR, False)
        Xa, Xs = [], []
        for j in range(10):
            ii = np.arange(j, N_TR, 10)
            sel = np.where(allg)[0]
            sel = np.sort(rng.choice(sel, rng.integers(len(sel) // 2, len(sel) + 1), replace=False))
            Xa.append((ii, unit(ref_(Y[ii][:, sel]) @ (Bs @ slor_kernel(G[sel][:, :, 2])).T)))
        Xa_ = np.vstack([x for _, x in Xa]); Ia = np.concatenate([idx[i] for i, _ in Xa])
        m_ad = MLPRegressor(**MLP_KW).fit(Xa_, pos_mm[Ia] / 100.0)
        m_se = MLPRegressor(**MLP_KW).fit(unit(ref_(Y[:, always])), pos_mm[idx] / 100.0)
        del Y, Xa, Xa_
        for r in [x for x in RUNS if x["sub"] == sub]:
            good = GOOD[(sub, r["run"])]
            Yr = C.evoked(root, sub, r["run"])
            u = signal(Yr, good)
            ua = signal(Yr, always)
            Ks = slor_kernel(G[good][:, :, 2])
            ests = {"경로 2: 어댑터 (전용 모델)": tree.query(m_ad.predict(unit((u @ (Bs @ Ks).T)[None]))[0] * 100.0)[1],
                    "경로 2: 센서만 (전용 모델)": tree.query(m_se.predict(ua[None])[0] * 100.0)[1],
                    "쌍극자 스캔 (데이터 머리 모델)": scan_idx(u, G[good])}
            for k, v in ests.items():
                rows.append([SEED, sub, r["run"], "-".join(r["contacts"]), r["mA"], k, f"{np.linalg.norm(pos_mm[v] - MID[(sub, r['run'])]):.2f}"])
        rr = [x for x in rows if x[1] == sub and x[5].startswith(("경로 2", "쌍극자 스캔 (데이터"))]
        print(f"[경로 2] {sub}: " + "  ".join(f"{k} {np.median([float(x[6]) for x in rr if x[5] == k]):.1f}" for k in dict.fromkeys(x[5] for x in rr)) + f"  ({time.time() - t0:.0f} s)", flush=True)

out = os.path.join(HERE, f"lmi_routes_seed{SEED}{'_r' + ONLY if ONLY else ''}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "피험자", "run", "접점", "mA", "방법", "오차mm"])
    w.writerows(rows)
print("저장:", out)
