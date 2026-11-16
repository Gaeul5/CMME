"""Week 6-6: 6-5 반전 실험을 OASIS-3 머리 10명으로 다시 (템플릿 변형 리드필드, oasis_warp_heads.py)

6-5와 다른 점
  - 머리: sample 변형 머리 대신 OASIS-3 10명의 T1에 맞춰 옮긴 머리 (머리 크기·모양·두피·두개골 위치가 실제 사람)
  - 시드마다 10명을 무작위로 학습 6명 / 평가 4명으로 나눔 (같은 사람이 학습과 평가에 같이 들어가지 않음)
  - B(템플릿 머리)는 옮기기 전 원래 sample 머리(head000). 실제로 "MRI 없이 표준 머리를 쓰는" 경우와 같음
  - 학습 샘플 수는 6-5와 같음 (머리 수가 줄어 머리당 샘플은 늘어남)
아래는 6-5의 설명 그대로.

Week 6-5: 반전 실험 — 어댑터의 이득은 정말 '물리(리드필드)'에서 오는가?

지금까지의 주장: "리드필드로 만든 역연산자를 거쳐 넣으면, 처음 보는 머리와 배치에서도 잘 된다."
반전 가설: 이득의 대부분은 물리가 아니라 "배치가 바뀌어도 같은 공간으로 보내 주는 고정된 선형 변환"에서 올 수 있다.
그렇다면 아무 의미 없는 무작위 '가짜 리드필드'로 같은 변환을 만들어도 비슷하게 잘 될 것이다.

학습 조건 (모델은 모두 같은 MLP (256,128), 머리 12개 x 무작위 배치 200개, 60,000개, 쌍극자 1개 위치 예측)
  A 자기 머리 리드필드 (지금까지의 어댑터)
  B 템플릿 머리 리드필드 (모든 머리에 head000의 것)
  C 무작위 가짜 리드필드 (64 x 7,498 정규분포, 모든 머리 공통. 물리 정보 0, 배치 정보만 있음)
  D 마스크 (변환 없이 64칸 + 마스크, 참고)
시험 때 일부러 틀린 물리를 주기 (A 모델은 다시 학습하지 않음)
  A1 다른 사람 리드필드: 평가 머리끼리 리드필드가 가장 다른 짝의 것을 씀
  A2 전극 좌우 뒤바뀜: 실제 측정은 그대로인데, 각 채널이 좌우 대칭 위치에 있다고 잘못 알고 있음 (흔한 실수)
  A3 전극 순서 뒤섞임: 채널 위치를 완전히 무작위로 잘못 앎
  같은 실수를 sLORETA에도 줌
평가: 학습에 안 쓴 머리 6개(정답 forward), 표준 64/32/16채널, 처음 보는 무작위 배치 16/32채널(각 2개), SNR 10 dB

실행: SEED=0 python week06/physics_ablation.py
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
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(5000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 6000 if QUICK else 30000
N_TE = 60 if QUICK else 250
SNR_DB, AMP_RANGE = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA = 256
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

files = [os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz")] + \
    sorted(glob.glob(os.path.join(HERE, "cache_oasis", "OAS*.npz")))
assert len(files) >= 8, "OASIS 머리가 부족함. oasis_warp_heads.py를 먼저 실행"
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"]))
_perm = np.random.default_rng(4000 + SEED).permutation(np.arange(1, len(heads)))
N_TEST = 4
TEST_HEADS, TRAIN_HEADS = sorted(_perm[:N_TEST].tolist()), sorted(_perm[N_TEST:].tolist())
print("학습 머리", [os.path.basename(files[i])[:8] for i in TRAIN_HEADS], "평가 머리", [os.path.basename(files[i])[:8] for i in TEST_HEADS], flush=True)
pos0 = heads[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)
ALL = np.arange(64)
G_RAND = np.random.default_rng(123).normal(size=(64, N_SRC))     # 가짜 리드필드 (시드와 무관하게 고정)

# 좌우 대칭 채널 (biosemi64 위치에서 x를 뒤집어 가장 가까운 채널)
mont = mne.channels.make_standard_montage("biosemi64")
n64 = mont.ch_names
xyz = np.array([mont.get_positions()["ch_pos"][c] for c in n64])
MIRROR = cKDTree(xyz).query(xyz * np.array([-1, 1, 1]))[1]

# 평가 머리 중 리드필드가 가장 다른 짝
def corr(a, b):
    a, b = a[:, :, 2], b[:, :, 2]
    a, b = a - a.mean(0), b - b.mean(0)
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


C = {(i, j): corr(heads[i]["Gm"], heads[j]["Gm"]) for i in TEST_HEADS for j in TEST_HEADS if i != j}
WRONG = {i: min((j for j in TEST_HEADS if j != i), key=lambda j: C[(i, j)]) for i in TEST_HEADS}


def simulate(G3, n, tilt_max):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    clean = rng.uniform(*AMP_RANGE, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    ref = clean - clean.mean(1, keepdims=True)
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (SNR_DB / 10))
    return clean + rng.normal(size=clean.shape) * sigma, idx


def ref_(X):
    return X - X.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def slor_kernel(G, sel, lam2=1.0 / 9.0):
    """G: 64 x N (수직 성분)"""
    k = len(sel)
    R = np.eye(k) - 1.0 / k
    L = R @ G[sel]
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K = Lw.T @ A @ W
    res = np.einsum("ij,ji->i", Lw.T @ A, Lw)
    return K / np.sqrt(res)[:, None]


def random_montage(k=None):
    k = rng.integers(3, 65) if k is None else k
    return np.sort(rng.choice(64, k, replace=False))


def Gfor(cond, h):
    return {"A": heads[h]["Gm"][:, :, 2], "B": heads[0]["Gm"][:, :, 2], "C": G_RAND}[cond]


t0 = time.time()
BASES = {}
for cond in ("A", "C"):          # B는 A와 같은 기저 (둘 다 head000으로 계산)
    maps = []
    for j in range(60 if QUICK else 150):
        sel = ALL if j % 5 == 0 else random_montage()
        Y, _ = simulate(heads[0]["Gm"], 40, TRAIN_TILT)
        maps.append(unit(ref_(Y[:, sel]) @ slor_kernel(Gfor(cond, 0), sel).T))
    BASES[cond] = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
BASES["B"] = BASES["A"]


def feat(cond, Y, h, sel, Gk=None, data_sel=None):
    """sel: 모델이 '있다고 믿는' 채널, data_sel: 실제 값을 가져오는 채널 (전극 실수 실험용)"""
    data_sel = sel if data_sel is None else data_sel
    if cond == "D":
        X = np.zeros((len(Y), 128))
        X[:, sel] = unit(ref_(Y[:, data_sel]))
        X[:, 64 + np.asarray(sel)] = 1.0
        return X
    G = Gfor(cond, h) if Gk is None else Gk
    return unit(ref_(Y[:, data_sel]) @ (BASES[cond] @ slor_kernel(G, sel)).T)


CONDS = {"A 자기 머리 리드필드": "A", "B 템플릿 머리 리드필드": "B", "C 무작위 가짜 리드필드": "C", "D 마스크 (변환 없음)": "D"}
pool = [random_montage() for _ in range(60 if QUICK else 200)]
pool_keys = {tuple(s) for s in pool}
per = 2 * N_TR // len(TRAIN_HEADS)
X = {k: [] for k in CONDS}
I = []
for h in TRAIN_HEADS:
    Y, idx = simulate(heads[h]["Gm"], per, TRAIN_TILT)
    a = rng.integers(0, len(pool), per)
    for m in np.unique(a):
        ii = np.where(a == m)[0]
        for k, c in CONDS.items():
            X[k].append(feat(c, Y[ii], h, pool[m]))
        I.append(idx[ii])
I = np.concatenate(I)
models = {}
for k in CONDS:
    t1 = time.time()
    models[k] = MLPRegressor(**MLP_KW).fit(np.vstack(X[k]), pos0[I] / 100.0)
    print(f"{k} 학습 {time.time() - t1:.0f} s", flush=True)
del X

EVAL = {"표준 64채널": ALL}
for name, std in (("표준 32채널", "biosemi32"), ("표준 16채널", "biosemi16")):
    EVAL[name] = np.array([n64.index(c) for c in mne.channels.make_standard_montage(std).ch_names])
for c in (32, 16):
    for r in range(2):
        s = random_montage(c)
        while tuple(s) in pool_keys:
            s = random_montage(c)
        EVAL[f"처음 보는 배치 {c}채널 #{r + 1}"] = s

rows = []


def err(h, j, it):
    return np.linalg.norm(heads[h]["pos"][j] - heads[h]["pos"][it], axis=1)


for cond_name, sel in EVAL.items():
    for h in TEST_HEADS:
        Y, it = simulate(heads[h]["Gt"], N_TE, TEST_TILT)
        res = {}
        for k, c in CONDS.items():
            res[k] = tree0.query(models[k].predict(feat(c, Y, h, sel)) * 100.0)[1]
        # 시험 때 틀린 물리 (A 모델, 재학습 없음)
        mir = np.array([MIRROR[s] for s in sel])
        shuf = rng.permutation(sel)
        mA = models["A 자기 머리 리드필드"]
        res["A1 시험 때 다른 사람 리드필드"] = tree0.query(mA.predict(feat("A", Y, h, sel, Gk=heads[WRONG[h]]["Gm"][:, :, 2])) * 100.0)[1]
        res["A2 시험 때 전극 좌우 뒤바뀜"] = tree0.query(mA.predict(feat("A", Y, h, sel, data_sel=mir)) * 100.0)[1]
        res["A3 시험 때 전극 순서 뒤섞임"] = tree0.query(mA.predict(feat("A", Y, h, sel, data_sel=shuf)) * 100.0)[1]
        # sLORETA (정상, 같은 실수)
        Gh = heads[h]["Gm"][:, :, 2]
        K = slor_kernel(Gh, sel)
        res["sLORETA 정상"] = np.abs(ref_(Y[:, sel]) @ K.T).argmax(1)
        res["sLORETA 다른 사람 리드필드"] = np.abs(ref_(Y[:, sel]) @ slor_kernel(heads[WRONG[h]]["Gm"][:, :, 2], sel).T).argmax(1)
        res["sLORETA 전극 좌우 뒤바뀜"] = np.abs(ref_(Y[:, mir]) @ K.T).argmax(1)
        res["sLORETA 전극 순서 뒤섞임"] = np.abs(ref_(Y[:, shuf]) @ K.T).argmax(1)
        for k, j in res.items():
            e = err(h, j, it)
            rows.append((SEED, cond_name, k, os.path.basename(files[h])[:8], len(sel), e.mean(), np.median(e), 100 * np.mean(e <= 10)))
    print(f" [{cond_name}] " + "  ".join(f"{k.split(' ')[0]}:{np.mean([x[5] for x in rows if x[1] == cond_name and x[2] == k]):.1f}"
                                       for k in dict.fromkeys(x[2] for x in rows)), flush=True)
print(f"전체 {time.time() - t0:.0f} s")
out = os.path.join(HERE, f"oasis_ablation_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "조건", "방법", "머리", "채널수", "평균mm", "중앙값mm", "10mm이내%"])
    for r in rows:
        w.writerow(list(r[:5]) + [f"{r[5]:.2f}", f"{r[6]:.2f}", f"{r[7]:.1f}"])
print("저장:", out)
