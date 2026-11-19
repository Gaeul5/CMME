"""Week 9-3: Parmigiani 실제 기록에서 학습 모델(8-3의 합친 방법)과 쌍극자 스캔 비교

모델: 8-3과 같은 범용 어댑터 (위치마다 쌍극자 적합도 지도 → 기준 머리 7,498곳 → 공유 PCA 256 → MLP → 기준 좌표 → 그 사람 머리로)
머리 모델: 모두 템플릿 변형 머리라 source 번호가 기준 머리와 같음. 좌표 변환은 변형 머리 source ↔ 기준 source로 맞춘 아핀
학습: 가짜 신호만. OASIS 변형 머리 28명(64채널 안 무작위 배치) + Parmigiani 절반(185채널 안 무작위 배치)
평가: 나머지 절반의 실제 기록. 두 묶음을 바꿔 한 번 더 (실제 기록과 정답은 학습에 안 씀)
  Localize-MI에서 학습한 것이 없으므로, 두 데이터에 겹치는 환자 문제는 생기지 않음
구간: par_eval.py와 같은 4개. 오차 = 추정 위치 → 자극 접점 쌍의 중점

실행: SEED=0 python week09/par_hybrid.py <reduced 폴더>
"""
import csv
import glob
import os
import re
import sys
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get("SEED", 0))
rng = np.random.default_rng(19000 + SEED)
red = sys.argv[1]
N_TR, N_PCA = 60000, 256
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)
WINDOWS = {"자극 전 -5~-1": (-5, -1), "자극 직후 0~5": (0, 5), "자극 잡음 최대 5~12": (5, 12), "초기 반응 15~40": (15, 40)}
LMI_SAME = {"sub-21", "sub-10", "sub-17", "sub-09", "sub-27", "sub-28"}

z0 = np.load(os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz"))
pos0, G0 = z0["pos"], z0["G_model"].astype(np.float64)
tree0 = cKDTree(pos0)


def ref_(X):
    return X - X.mean(-1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-30)


def affine_fit(P, Q):
    return np.linalg.lstsq(np.c_[P, np.ones(len(P))], Q, rcond=None)[0]


def apply(A, P):
    return np.c_[P, np.ones(len(P))] @ A


def apply_inv(A, Q):
    return (Q - A[3]) @ np.linalg.inv(A[:3])


def gof(G3sel, Yr):
    A = np.transpose(G3sel, (1, 0, 2))
    Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
    Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, G3sel.shape[0])
    return ((Yr @ Q2.T) ** 2).reshape(len(Yr), -1, 3).sum(2) / np.maximum((Yr**2).sum(1, keepdims=True), 1e-30)


def dirs(n):
    return unit(rng.normal(size=(n, 3)))


def simulate(G3, n):
    idx = rng.integers(0, G3.shape[1], n)
    clean = (rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n))[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], dirs(n))
    r = ref_(clean)
    snr = rng.uniform(5, 30, (n, 1))
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (snr / 10)), idx


maps = []
for _ in range(150):
    sel = np.sort(rng.choice(64, rng.integers(8, 65), replace=False))
    Y, _ = simulate(G0[sel], 40)
    maps.append(unit(gof(G0[sel], ref_(Y)) ** 4))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps


def feats(G3, Y, sel):
    return unit(gof(G3[sel], ref_(Y[:, sel])) ** 4) @ B.T


heads = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "week06", "cache_oasis", "OAS*.npz"))):
    z = np.load(f)
    heads[os.path.basename(f)[:8]] = dict(G=z["G_model"].astype(np.float64), A=affine_fit(z["pos"], pos0), pos=z["pos"], nch=64, kmin=8)
PAR = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(HERE, "cache_par", "sub-*.npz")))
for s in PAR:
    z = np.load(os.path.join(HERE, "cache_par", f"{s}.npz"))
    heads[s] = dict(G=z["G"].astype(np.float64), A=affine_fit(z["pos_t1w_mm"], pos0), pos=z["pos_t1w_mm"], names=list(z["names"]),
                    nch=z["G"].shape[0], kmin=16)
perm = np.random.default_rng(0).permutation(PAR)                      # 묶음 나누기는 시드와 무관하게 고정
FOLDS = [sorted(perm[: len(PAR) // 2]), sorted(perm[len(PAR) // 2:])]


def tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


rows = []
t0 = time.time()
for fi, test in enumerate(FOLDS):
    train = [k for k in heads if k.startswith("OAS")] + [k for k in PAR if k not in test]
    per = N_TR // len(train)
    X, T = [], []
    for k in train:
        h = heads[k]
        Y, idx = simulate(h["G"], per)
        for j in range(10):
            ii = np.arange(j, per, 10)
            sel = np.sort(rng.choice(h["nch"], rng.integers(h["kmin"], h["nch"] + 1), replace=False))
            X.append(feats(h["G"], Y[ii], sel))
            T.append(apply(h["A"], h["pos"][idx[ii]]))
    m = MLPRegressor(**MLP_KW).fit(np.vstack(X), np.vstack(T) / 100.0)
    del X, T
    print(f"묶음 {fi} 학습 {time.time() - t0:.0f} s", flush=True)
    for s in test:
        h = heads[s]
        tree = cKDTree(h["pos"])
        contacts = {r["name"]: np.array([float(r[c]) for c in "xyz"]) * 1000
                    for r in tsv(glob.glob(os.path.join(red, s, "*space-T1w_electrodes.tsv"))[0]) if r["x"] not in ("n/a", "")}
        for sf in sorted(glob.glob(os.path.join(red, s, f"{s}_run-*.npz"))):
            S = np.load(sf)
            mm = re.match(r"Stimulation of (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)", str(S["description"]))
            if not mm or mm.group(1) not in contacts:
                continue
            gt = contacts[mm.group(1)]
            st = dict(zip(S["eeg_names"], S["eeg_status"]))
            sel = np.array([i for i, n in enumerate(h["names"]) if st.get(n) == "good"])
            rowi = [list(S["eeg_names"]).index(h["names"][i]) for i in sel]
            Yc = S["eeg_mean"][rowi].astype(np.float64)
            Yc = Yc - Yc.mean(0, keepdims=True)
            t = (np.arange(Yc.shape[1]) / float(S["sfreq"]) - float(S["zero_time"])) * 1000
            for wn, (a, b) in WINDOWS.items():
                w = (t >= a) & (t <= b)
                seg = Yc[:, w]
                u = seg[:, np.argmax((seg**2).sum(0))]
                Yf = np.zeros((1, h["nch"]))
                Yf[0, sel] = u / np.linalg.norm(u)
                M = gof(h["G"][sel], ref_(Yf[:, sel]))[0]
                v_scan = int(M.argmax())
                v_hy = tree.query(apply_inv(h["A"], m.predict(unit(M[None] ** 4) @ B.T) * 100.0))[1][0]
                for meth, v in (("합친 방법", v_hy), ("쌍극자 스캔", v_scan)):
                    rows.append([SEED, s, os.path.basename(sf)[len(s) + 5:-4], mm.group(2), mm.group(3), mm.group(5), mm.group(6),
                                 "예" if s in LMI_SAME else "", wn, meth, f"{np.linalg.norm(h['pos'][v] - gt):.2f}"])
        r_ = [x for x in rows if x[1] == s and x[8] == "초기 반응 15~40"]
        print(f"  {s}: 15~40 ms 합친 {np.median([float(x[10]) for x in r_ if x[9] == '합친 방법']):.1f} / 스캔 {np.median([float(x[10]) for x in r_ if x[9] == '쌍극자 스캔']):.1f} ({time.time() - t0:.0f} s)", flush=True)
with open(os.path.join(HERE, f"par_hybrid_seed{SEED}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "피험자", "run", "세기", "펄스", "각도", "회백질백질", "LocalizeMI같은환자", "구간", "방법", "오차mm"])
    w.writerows(rows)
print("저장")
