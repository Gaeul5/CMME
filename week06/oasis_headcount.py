"""Week 8-8: 학습 머리 수 곡선 (OASIS-3 변형 머리 28명)

질문: 학습에 쓰는 MRI 수를 늘리면 처음 보는 머리에서 오차가 줄어드나 (11/17 미팅 "MRI 10 → 100명")
설정
  머리: OASIS-3 28명 (기존 10 + 추가 18). 시드마다 평가 6명을 무작위로 떼고, 나머지 22명에서 1, 3, 6, 12, 22명을 골라 학습
  모델: 범용 어댑터 하나 (그 머리 리드필드 → sLORETA → 공유 PCA 256 → MLP), 무작위 배치 8~64채널, 쌍극자 1개, 학습 샘플 30,000개 고정
  평가: 평가 6명의 '정답' 리드필드(전도도와 전극 위치가 모델과 다름), 표준 64채널과 처음 보는 무작위 16채널, SNR 10 dB
  비교: sLORETA, 쌍극자 스캔 (자기 머리 모델 리드필드)
주의: 변형 머리는 모두 sample 피질 주름을 늘린 것이라, 머리 수를 늘려도 '피질 모양'의 다양성은 늘지 않는다 (머리 크기·모양만 늘어남)

실행: SEED=0 python week06/oasis_headcount.py
"""
import csv
import glob
import os
import time
import warnings
import numpy as np
import mne
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
rng = np.random.default_rng(18000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR, N_TE, N_PCA = 30000, 600, 256
SNR_DB, AMP = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
COUNTS = [1, 3, 6, 12, 22]
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

z0 = np.load(os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz"))
pos0, G0 = z0["pos"], z0["G_model"].astype(np.float64)
tree0 = cKDTree(pos0)
files = sorted(glob.glob(os.path.join(HERE, "cache_oasis", "OAS*.npz")))
assert len(files) == 28, len(files)
heads = [dict(Gt=np.load(f)["G_true"].astype(np.float64), Gm=np.load(f)["G_model"].astype(np.float64), pos=np.load(f)["pos"]) for f in files]
NAMES = [os.path.basename(f)[:8] for f in files]
perm = rng.permutation(len(heads))
TEST, POOL = sorted(perm[:6].tolist()), perm[6:].tolist()                  # POOL 순서대로 앞에서 n명 → 학습 집합이 서로 포함 관계
N_SRC = len(pos0)


def simulate(G3, n, tilt):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt, n)); ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    clean = rng.uniform(*AMP, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    r = clean - clean.mean(1, keepdims=True)
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (SNR_DB / 10)), idx


def ref_(X):
    return X - X.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def slor(G3, sel, lam2=1.0 / 9.0):
    k = len(sel); R = np.eye(k) - 1.0 / k
    L = R @ G3[sel, :, 2]
    e, V = np.linalg.eigh(R @ R.T); keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L; Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    return (Lw.T @ A @ W) / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


def scan(G3, sel, Y):
    A = np.transpose(G3[sel], (1, 0, 2)); Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
    Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, len(sel))
    return ((ref_(Y) @ Q2.T) ** 2).reshape(len(Y), -1, 3).sum(2).argmax(1)


def rmont(k=None):
    return np.sort(rng.choice(64, rng.integers(8, 65) if k is None else k, replace=False))


t0 = time.time()
maps = []
for j in range(150):
    sel = np.arange(64) if j % 5 == 0 else rmont()
    Y, _ = simulate(G0, 40, TRAIN_TILT)
    maps.append(unit(ref_(Y[:, sel]) @ slor(G0, sel).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps
n64 = mne.channels.make_standard_montage("biosemi64").ch_names
EVAL = {"표준 64채널": np.arange(64), "처음 보는 16채널": rmont(16)}
DATA = {h: simulate(heads[h]["Gt"], N_TE, TEST_TILT) for h in TEST}
rows = []
for cond, sel in EVAL.items():
    for h in TEST:
        Y, it = DATA[h]
        P = heads[h]["pos"]
        for m, v in (("sLORETA (자기 머리)", np.abs(ref_(Y[:, sel]) @ slor(heads[h]["Gm"], sel).T).argmax(1)),
                     ("쌍극자 스캔 (자기 머리)", scan(heads[h]["Gm"], sel, Y[:, sel]))):
            rows.append([SEED, cond, m, 0, NAMES[h], np.linalg.norm(P[v] - P[it], axis=1).mean()])
for n in COUNTS:
    hs = POOL[:n]
    per = N_TR // n
    X, I = [], []
    for h in hs:
        Y, idx = simulate(heads[h]["Gm"], per, TRAIN_TILT)
        for j in range(20):
            ii = np.arange(j, per, 20); sel = rmont()
            X.append(unit(ref_(Y[ii][:, sel]) @ (B @ slor(heads[h]["Gm"], sel)).T)); I.append(idx[ii])
    m = MLPRegressor(**MLP_KW).fit(np.vstack(X), pos0[np.concatenate(I)] / 100.0)
    for cond, sel in EVAL.items():
        for h in TEST:
            Y, it = DATA[h]
            v = tree0.query(m.predict(unit(ref_(Y[:, sel]) @ (B @ slor(heads[h]["Gm"], sel)).T)) * 100.0)[1]
            rows.append([SEED, cond, "범용 어댑터", n, NAMES[h], np.linalg.norm(heads[h]["pos"][v] - heads[h]["pos"][it], axis=1).mean()])
    print(f"학습 머리 {n}명: " + "  ".join(f"{c} {np.mean([r[5] for r in rows if r[1] == c and r[3] == n]):.1f}" for c in EVAL) + f" ({time.time() - t0:.0f} s)", flush=True)
with open(os.path.join(HERE, f"oasis_headcount_seed{SEED}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "배치", "방법", "학습머리수", "머리", "평균mm"])
    for r in rows:
        w.writerow(r[:5] + [f"{r[5]:.2f}"])
print("저장")
