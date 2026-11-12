"""Week 6-2: 모델 하나로 채널 수 4~64개, 처음 보는 배치, 처음 보는 머리 (선행 연구와의 차별점 확인용)

선행 연구 정리에서 나온 질문
  - DeepSIF 계열(Rong 2025)은 채널 수(16/21/32/64/75)마다 모델을 따로 학습했다
  - 3D-PIUNet은 61채널에서 최대 11개를 빼는 것까지만 시험했다 (결과 숫자 없음)
  → "모델 1개가 채널 수가 크게 다른 처음 보는 배치를, 처음 보는 머리에서 받는다"가 실제로 되는지,
    그리고 채널 수별로 따로 학습한 모델과 비교해 얼마나 손해인지를 잰다.

설정 (week05 multihead.py E2와 같은 틀, 요약 벡터는 뺌)
  학습: 머리 12개(head000~011) x 무작위 채널 부분집합 200개(3~64채널), 샘플 60,000개 → 범용 모델 2개
        (a) 마스크: 64칸 값 + 64칸 마스크      (b) 어댑터: 그 머리·그 배치의 sLORETA 역연산자 → 공유 PCA 256
  비교: 채널 수별 전용 모델 (DeepSIF 계열 방식): 64/32/16채널 표준 배치마다 센서 입력 MLP를 따로 학습 (머리 12개, 30,000개)
        sLORETA (자기 머리 리드필드)
  평가: 학습에 안 쓴 머리 6개(head012~017)의 '정답' forward (전도도·전극 위치가 모델과 다름), SNR 10 dB
        E6a 채널 수 곡선: 4, 6, 8, 12, 16, 24, 32, 48, 53, 64채널. 채널 수마다 학습에 없던 무작위 배치 6개 (64채널은 1개)
                          53채널 = 64개 중 11개를 뺀 것 (3D-PIUNet의 센서 빼기 실험과 같은 비율은 아니고 같은 개수)
        E6b 표준 배치 64/32/16: 범용 모델 vs 채널 수별 전용 모델
  시드마다 학습 데이터, 무작위 배치, 평가 샘플이 모두 바뀐다. 머리별 평균을 따로 저장해 머리 간·시드 간 흩어짐을 본다.

한계: 머리는 sample 변형 머리(진짜 다른 사람 아님), 신호원은 쌍극자 1개, 표준 배치는 biosemi 64 안의 부분집합.

실행: SEED=0 python week06/channel_curve.py     (week05/cache_heads/head000~017.npz 필요)
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
rng = np.random.default_rng(1000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 6000 if QUICK else 30000
N_TE = 60 if QUICK else 250            # 평가: 머리 1개 x 배치 1개당
N_MONT_EVAL = 2 if QUICK else 6
SNR_DB, AMP_RANGE = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA = 256
COUNTS = [4, 6, 8, 12, 16, 24, 32, 48, 53, 64]
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

files = sorted(glob.glob(os.path.join(HERE, "..", "week05", "cache_heads", "head*.npz")))[:18]
assert len(files) == 18
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"]))
TRAIN_HEADS, TEST_HEADS = list(range(12)), list(range(12, 18))
pos0 = heads[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)
ALL = np.arange(64)


def simulate(G3, n, tilt_max):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    clean = rng.uniform(*AMP_RANGE, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    ref = clean - clean.mean(1, keepdims=True)
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (SNR_DB / 10))
    return clean + rng.normal(size=clean.shape) * sigma, idx


def pick_ref(Y, sel):
    X = Y[:, sel]
    return X - X.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def slor_kernel(G3, sel, lam2=1.0 / 9.0):
    k = len(sel)
    R = np.eye(k) - 1.0 / k
    L = R @ G3[sel, :, 2]
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


t0 = time.time()
maps = []
for j in range(60 if QUICK else 150):
    sel = ALL if j % 5 == 0 else random_montage()
    Y, _ = simulate(heads[0]["Gm"], 40, TRAIN_TILT)
    maps.append(unit(pick_ref(Y, sel) @ slor_kernel(heads[0]["Gm"], sel).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps


def feat_mask(Y, h, sel, K=None):
    X = np.zeros((len(Y), 128))
    X[:, sel] = unit(pick_ref(Y, sel))
    X[:, 64 + np.asarray(sel)] = 1.0
    return X


def feat_adapter(Y, h, sel, K=None):
    K = slor_kernel(heads[h]["Gm"], sel) if K is None else K
    return unit(pick_ref(Y, sel) @ (B @ K).T)


def feat_sensor(Y, h, sel, K=None):
    return unit(pick_ref(Y, sel))


def build(head_ids, n, montages, feats):
    per = n // len(head_ids)
    X = {k: [] for k in feats}
    I = []
    for h in head_ids:
        Y, idx = simulate(heads[h]["Gm"], per, TRAIN_TILT)
        a = rng.integers(0, len(montages), per)
        for m in np.unique(a):
            ii = np.where(a == m)[0]
            for k, f in feats.items():
                X[k].append(f(Y[ii], h, montages[m]))
            I.append(idx[ii])
    return {k: np.vstack(v) for k, v in X.items()}, np.concatenate(I)


def fit(X, I):
    return MLPRegressor(**MLP_KW).fit(X, pos0[I] / 100.0)


# 범용 모델 2개
pool = [random_montage() for _ in range(60 if QUICK else 200)]
pool_keys = {tuple(s) for s in pool}
UNI = {"범용 1개: 마스크": feat_mask, "범용 1개: 어댑터": feat_adapter}
X, I = build(TRAIN_HEADS, 2 * N_TR, pool, UNI)
uni = {k: fit(X[k], I) for k in UNI}
del X
print(f"범용 모델 학습 {time.time() - t0:.0f} s (seed {SEED})", flush=True)

# 채널 수별 전용 모델 (64/32/16 표준 배치)
n64 = mne.channels.make_standard_montage("biosemi64").ch_names
NAMED = {"64채널": ALL}
for name, std in (("32채널", "biosemi32"), ("16채널", "biosemi16")):
    NAMED[name] = np.array([n64.index(c) for c in mne.channels.make_standard_montage(std).ch_names])
t1 = time.time()
ded = {}
for name, sel in NAMED.items():
    X, I = build(TRAIN_HEADS, N_TR, [sel], {"s": feat_sensor})
    ded[name] = fit(X["s"], I)
print(f"전용 모델 3개 학습 {time.time() - t1:.0f} s", flush=True)

rows = []


def evaluate(exp, cond, sel, models):
    """models: {이름: (모델, 특징 함수)}. 머리별 평균 오차를 rows에 추가."""
    for h in TEST_HEADS:
        Y, it = simulate(heads[h]["Gt"], N_TE, TEST_TILT)
        K = slor_kernel(heads[h]["Gm"], sel)
        res = {"sLORETA (자기 머리)": np.abs(pick_ref(Y, sel) @ K.T).argmax(1)}
        for k, (m, f) in models.items():
            res[k] = tree0.query(m.predict(f(Y, h, sel, K)) * 100.0)[1]
        for k, j in res.items():
            e = np.linalg.norm(heads[h]["pos"][j] - heads[h]["pos"][it], axis=1)
            rows.append((SEED, exp, cond, k, h, len(sel), e.mean(), np.median(e), 100 * np.mean(e <= 10)))


t2 = time.time()
um = {k: (uni[k], UNI[k]) for k in UNI}
for c in COUNTS:
    mont = [ALL] if c == 64 else []
    while len(mont) < (1 if c == 64 else N_MONT_EVAL):
        s = random_montage(c)
        if tuple(s) not in pool_keys:          # 학습에 쓴 배치는 제외
            mont.append(s)
    for j, sel in enumerate(mont):
        evaluate("E6a", f"{c}채널", sel, um)
    r = [x for x in rows if x[1] == "E6a" and x[2] == f"{c}채널"]
    print(f" {c:>2}채널: " + "  ".join(f"{k} {np.mean([x[6] for x in r if x[3] == k]):5.1f}" for k in ["sLORETA (자기 머리)"] + list(UNI)), flush=True)
for name, sel in NAMED.items():
    evaluate("E6b", name, sel, dict(um, **{"채널 수별 전용 모델": (ded[name], feat_sensor)}))
    r = [x for x in rows if x[1] == "E6b" and x[2] == name]
    print(f" [표준 {name}] " + "  ".join(f"{k} {np.mean([x[6] for x in r if x[3] == k]):5.1f}" for k in sorted({x[3] for x in r})), flush=True)
print(f"평가 {time.time() - t2:.0f} s, 전체 {time.time() - t0:.0f} s")

out = os.path.join(HERE, f"channel_curve_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "실험", "조건", "방법", "머리", "채널수", "평균mm", "중앙값mm", "10mm이내%"])
    for r in rows:
        w.writerow(list(r[:6]) + [f"{r[6]:.2f}", f"{r[7]:.2f}", f"{r[8]:.1f}"])
print("저장:", out)
