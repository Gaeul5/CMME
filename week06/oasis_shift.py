"""Week 6-7b-2: 머리 분포가 바뀔 때(변형 머리 ↔ OASIS) 자기 머리 리드필드가 템플릿 머리보다 나은가.
6-7b와 같은 틀에서 학습 묶음 2개(변형 머리 12개, OASIS 6명)와 입력 3가지(센서만, 어댑터 자기 머리, 어댑터 템플릿 머리)만 돌린다.
아래는 6-7b 설명 그대로.

Week 6-7b: 학습 머리를 무엇으로, 몇 명으로 해야 하나 (OASIS-3 머리 10명 + sample 변형 머리)

질문 (11/4 미팅 "MRI 50개 규모", 11/10 "OASIS 이용"에 대한 작은 버전)
  1) 학습 머리 수: OASIS 1명 / 3명 / 6명으로 학습하면 처음 보는 OASIS 머리에서 얼마나 달라지나
  2) 가짜 머리로 충분한가: sample 변형 머리 12개로만 학습한 모델이 OASIS 머리에서 통하나 (반대 방향도)
  3) 섞으면 나아지나: 변형 머리 12개 + OASIS 6명

설정: 64채널, 쌍극자 1개 위치 예측, MLP (256, 128), 학습 샘플 총 30,000개 고정 (머리 수와 무관)
  입력 두 가지: 센서만 / 어댑터 (자기 머리 리드필드 → sLORETA → PCA 256)
  평가: 처음 보는 OASIS 4명 (시드마다 다른 4명), 변형 머리 평가용 6개(head012~017). 정답 forward, SNR 10 dB
  OASIS 1명·3명은 학습 6명 중 앞에서부터 고름 (시드마다 학습 6명이 달라지므로 고르는 사람도 달라짐)

실행: SEED=0 python week06/oasis_transfer.py
"""
import csv
import glob
import os
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(8000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 6000 if QUICK else 30000
N_TE = 200 if QUICK else 1000
SNR_DB, AMP_RANGE = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA = 256
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

fake = sorted(glob.glob(os.path.join(HERE, "..", "week05", "cache_heads", "head*.npz")))[:18]
oas = sorted(glob.glob(os.path.join(HERE, "cache_oasis", "OAS*.npz")))
assert len(fake) == 18 and len(oas) == 10
files = fake + oas
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"]))
NAMES = [os.path.basename(f).replace(".npz", "")[:8] for f in files]
FAKE_TRAIN, FAKE_TEST = list(range(12)), list(range(12, 18))
perm = np.random.default_rng(4000 + SEED).permutation(np.arange(18, 28))   # 다른 6-6, 6-7 실험과 같은 분할
O_TEST, O_TRAIN = sorted(perm[:4].tolist()), sorted(perm[4:].tolist())
print("OASIS 학습", [NAMES[i] for i in O_TRAIN], "평가", [NAMES[i] for i in O_TEST], flush=True)
pos0 = heads[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)


def simulate(G3, n, tilt_max):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    clean = rng.uniform(*AMP_RANGE, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    ref = clean - clean.mean(1, keepdims=True)
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (SNR_DB / 10))
    return clean + rng.normal(size=clean.shape) * sigma, idx


def ref_(Y):
    return Y - Y.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def slor_kernel(G3, lam2=1.0 / 9.0):
    k = G3.shape[0]
    R = np.eye(k) - 1.0 / k
    L = R @ G3[:, :, 2]
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K = Lw.T @ A @ W
    return K / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


maps = []
for _ in range(40 if QUICK else 150):
    Y, _ = simulate(heads[0]["Gm"], 40, TRAIN_TILT)
    maps.append(unit(ref_(Y) @ slor_kernel(heads[0]["Gm"]).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps
PROJ = {h: B @ slor_kernel(heads[h]["Gm"]) for h in range(len(heads))}      # 머리마다 256 x 64
KS = {h: slor_kernel(heads[h]["Gm"]) for h in O_TEST + FAKE_TEST}

FEATS = {"센서만": lambda Y, h: unit(ref_(Y)), "어댑터 (자기 머리)": lambda Y, h: unit(ref_(Y) @ PROJ[h].T),
         "어댑터 (템플릿 머리)": lambda Y, h: unit(ref_(Y) @ PROJ[0].T)}
SETS = {"변형 머리 12개": FAKE_TRAIN, "OASIS 6명": O_TRAIN}
rows = []
t0 = time.time()
# 평가 데이터는 한 번만 만들어 모든 모델에 똑같이 씀
TESTS = {"OASIS 평가 4명": O_TEST, "변형 머리 평가 6개": FAKE_TEST}
DATA = {h: simulate(heads[h]["Gt"], N_TE, TEST_TILT) for h in O_TEST + FAKE_TEST}


def err(h, j, it):
    return np.linalg.norm(heads[h]["pos"][j] - heads[h]["pos"][it], axis=1)


for tname, hs in TESTS.items():
    for h in hs:
        Y, it = DATA[h]
        rows.append((SEED, "sLORETA (자기 머리)", "-", tname, NAMES[h], err(h, np.abs(ref_(Y) @ KS[h].T).argmax(1), it).mean()))
for sname, hs in SETS.items():
    per = N_TR // len(hs)
    X = {k: [] for k in FEATS}
    I = []
    for h in hs:
        Y, idx = simulate(heads[h]["Gm"], per, TRAIN_TILT)
        for k, f in FEATS.items():
            X[k].append(f(Y, h))
        I.append(idx)
    I = np.concatenate(I)
    line = f" [{sname}] "
    for k in FEATS:
        m = MLPRegressor(**MLP_KW).fit(np.vstack(X[k]), pos0[I] / 100.0)
        for tname, ths in TESTS.items():
            es = []
            for h in ths:
                Y, it = DATA[h]
                e = err(h, tree0.query(m.predict(FEATS[k](Y, h)) * 100.0)[1], it).mean()
                rows.append((SEED, k, sname, tname, NAMES[h], e))
                es.append(e)
            line += f"{k} → {tname} {np.mean(es):.1f}  "
    print(line + f"({time.time() - t0:.0f} s)", flush=True)

out = os.path.join(HERE, f"oasis_shift_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "방법", "학습 머리", "평가 머리 묶음", "머리", "평균mm"])
    for r in rows:
        w.writerow(list(r[:5]) + [f"{r[5]:.2f}"])
print("저장:", out)
