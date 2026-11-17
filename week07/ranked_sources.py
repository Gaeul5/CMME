"""Week 7-1: 가장 센 신호원부터 순위대로 찾기 (시간 축 사용, 신호원 1~3개)

11/17 미팅: "일반화 성능, one, two source에서 제일 센 신호 찾고 순위별로 찾기", "분포로 봐야 됨"
6-4의 문제: 한 시점만 보면 신호원 2~3개일 때 모든 방법이 가장 센 하나만 찾았다.

방법: 순차 제거 (찾고 → 지우고 → 다음 것 찾기)
  1) 시간 창(50개 시점) 전체에서 가장 두드러진 센서 무늬 하나를 뽑는다 (SVD 첫 성분. 시간 정보를 이렇게 쓴다)
  2) 그 무늬로 위치 하나를 찾는다 (sLORETA 또는 학습 모델)
  3) 찾은 위치의 리드필드(3방향)로 설명되는 부분을 기록에서 지운다
  4) 남은 기록으로 1)부터 반복 → 찾은 순서가 곧 순위
비교
  한 시점, sLORETA 봉우리      : 6-4와 같은 방식 (시간·제거 없음). 센서 파워가 가장 큰 시점의 지도에서 봉우리 K개
  sLORETA 순차 제거             : 위 절차, 위치 찾기는 sLORETA
  어댑터 순차 제거 (일반 학습)  : 위치 찾기는 6-2의 범용 어댑터 모델과 같은 방식으로 학습한 모델
  어댑터 순차 제거 (제거 학습)  : 학습 때도 "다른 신호원을 지운 뒤의 무늬"를 섞어 보여 준 모델
  마스크 순차 제거 (제거 학습)  : 리드필드 변환 없이 같은 학습
설정
  머리: OASIS-3 10명 (week06 oasis_warp_heads.py). 시드마다 학습 6명 / 평가 4명. 기준 좌표는 sample 원래 머리
  학습: 무작위 배치 200개(8~64채널), 60,000개, 쌍극자 1개, 부호 무작위, 잡음 5~30 dB. MLP (256,128), 출력은 위치
  평가: 신호원 K = 1, 2, 3개 (중심끼리 30 mm 이상), 방향 0~30도, 세기 0.5~1.5
        신호원마다 다른 파형 (6~15 Hz, 창 안에서 다른 시각에 나타났다 사라짐), 250 Hz, 0.2초
        정답 forward(전도도·전극 위치 다름), SNR 10 dB. 표준 64/32/16채널
  신호원 개수 K는 안다고 가정 (한계)
지표: 찾은 위치와 정답을 짝지은 거리(mm), 20 mm 안에 못 찾은 비율, 첫 번째로 찾은 것이 가장 센 신호원인 비율,
      순위별(가장 센 것, 둘째, 셋째) 오차. 오차 분포(히스토그램)도 저장

추가 (Week 7-4): 학습 없는 기준 두 개. SCAN_ONLY=1이면 학습을 건너뛰고 학습 없는 방법만 평가해 scan_ranked_*.csv에 저장
  쌍극자 스캔 순차 제거 : 가장 두드러진 무늬 하나를 가장 잘 설명하는 위치(GOF 최대)를 고르고 지움
  MUSIC 순차 제거       : 남은 신호원 수만큼의 무늬(SVD 성분)가 만드는 공간과 가장 잘 겹치는 위치를 고르고 지움
                          (RAP-MUSIC과 같은 생각이지만 리드필드 쪽은 다시 투영하지 않고 기록 쪽만 지우는 단순한 형태)

실행: SEED=0 python week07/ranked_sources.py
"""
import csv
import glob
import os
import time
import warnings
import numpy as np
import mne
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
SCAN_ONLY = os.environ.get("SCAN_ONLY") == "1"
rng = np.random.default_rng(9000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
W6 = os.path.join(HERE, "..", "week06")
N_TR = 6000 if QUICK else 30000
N_TE = 40 if QUICK else 150            # 머리 x 배치 x K 마다
SNR_DB, AMP = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA, T, FS = 256, 50, 250.0
MIN_SEP, PEAK_SEP, HIT = 30.0, 20.0, 20.0
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

files = [os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz")] + sorted(glob.glob(os.path.join(W6, "cache_oasis", "OAS*.npz")))
assert len(files) == 11
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"]))
NAMES = [os.path.basename(f)[:8] for f in files]
perm = np.random.default_rng(4000 + SEED).permutation(np.arange(1, 11))
TEST_HEADS, TRAIN_HEADS = sorted(perm[:4].tolist()), sorted(perm[4:].tolist())
print("학습", [NAMES[i] for i in TRAIN_HEADS], "평가", [NAMES[i] for i in TEST_HEADS], flush=True)
pos0 = heads[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)
ALL = np.arange(64)


def ref_(X):
    return X - X.mean(-1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-30)


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
    return K / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


def dirs(n, tilt):
    th = np.deg2rad(rng.uniform(0, tilt, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)


def qbasis(G3, sel, v):
    """정점 v들의 3방향 리드필드(평균 기준)의 정규직교 기저. 반환 (n, k, 3)"""
    A = ref_(np.transpose(G3[sel][:, v, :], (1, 2, 0)))        # (n, 3, k)
    return np.linalg.qr(np.transpose(A, (0, 2, 1)))[0]


def remove(Y, Q):
    """Y (n, k) 또는 (n, T, k) 에서 Q (n, k, 3) 가 펼치는 부분을 지움"""
    if Y.ndim == 2:
        return Y - np.einsum("nkj,nj->nk", Q, np.einsum("nkj,nk->nj", Q, Y))
    return Y - np.einsum("nkj,ntj->ntk", Q, np.einsum("nkj,ntk->ntj", Q, Y))


def random_montage():
    return np.sort(rng.choice(64, rng.integers(8, 65), replace=False))


t0 = time.time()
maps = []
for j in range(40 if QUICK else 150):
    sel = ALL if j % 5 == 0 else random_montage()
    idx = rng.integers(0, N_SRC, 40)
    y = np.einsum("cnk,nk->nc", heads[0]["Gm"][:, idx, :], dirs(40, TRAIN_TILT))[:, sel]
    maps.append(unit(ref_(y) @ slor_kernel(heads[0]["Gm"], sel).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps


def feat_adapter(X, h, sel, K=None):
    K = slor_kernel(heads[h]["Gm"], sel) if K is None else K
    return unit(X @ (B @ K).T)


def feat_mask(X, h, sel, K=None):
    Z = np.zeros((len(X), 128))
    Z[:, sel] = unit(X)
    Z[:, 64 + np.asarray(sel)] = 1.0
    return Z


def train_data(deflate):
    """쌍극자 1개 무늬(부호 무작위, 잡음 포함). deflate면 60%에 대해 다른 위치 1~2곳을 지운 무늬로 바꿈"""
    Xa, Xm, I = [], [], []
    per = 2 * N_TR // len(TRAIN_HEADS)
    for h in TRAIN_HEADS:
        G = heads[h]["Gm"]
        idx = rng.integers(0, N_SRC, per)
        amp = rng.uniform(*AMP, per) * rng.choice([-1, 1], per)
        clean = amp[:, None] * np.einsum("cnk,nk->nc", G[:, idx, :], dirs(per, TRAIN_TILT))
        r = ref_(clean)
        snr = rng.uniform(5, 30, (per, 1))           # 시간 창에서 뽑은 무늬는 잡음이 줄어 있으므로 학습 잡음 수준을 넓게 (5~30 dB)
        Y = clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (snr / 10))
        a = rng.integers(0, len(pool), per)
        for m in np.unique(a):
            ii = np.where(a == m)[0]
            sel = pool[m]
            X = ref_(Y[ii][:, sel])
            if deflate:
                for _ in range(2):
                    jj = np.where(rng.random(len(ii)) < (0.6 if _ == 0 else 0.4))[0] if _ == 0 else jj[rng.random(len(jj)) < 0.4]
                    if len(jj) == 0:
                        break
                    v = rng.integers(0, N_SRC, len(jj))
                    near = np.linalg.norm(pos0[v] - pos0[idx[ii][jj]], axis=1) < MIN_SEP
                    v[near] = rng.integers(0, N_SRC, near.sum())
                    X[jj] = remove(X[jj], qbasis(G, sel, v))
            K = slor_kernel(G, sel)
            Xa.append(feat_adapter(X, h, sel, K))
            Xm.append(feat_mask(X, h, sel))
            I.append(idx[ii])
    return np.vstack(Xa), np.vstack(Xm), np.concatenate(I)


pool = [random_montage() for _ in range(60 if QUICK else 200)]
def fit(X, I):
    return None if SCAN_ONLY else MLPRegressor(**MLP_KW).fit(X, pos0[I] / 100.0)


def eigmax3(S):
    """대칭 3x3 행렬 묶음 (..., 3, 3)의 가장 큰 고윳값 (닫힌 식)"""
    q = np.trace(S, axis1=-2, axis2=-1) / 3.0
    Bm = S - q[..., None, None] * np.eye(3)
    p = np.sqrt(np.maximum((Bm**2).sum((-2, -1)) / 6.0, 1e-30))
    r = np.clip(np.linalg.det(Bm) / (2.0 * p**3), -1.0, 1.0)
    return q + 2.0 * p * np.cos(np.arccos(r) / 3.0)


Xa, Xm, I = train_data(False)
m_plain = fit(Xa, I)
Xa, Xm, I = train_data(True)
m_defl = fit(Xa, I)
m_mask = fit(Xm, I)
del Xa, Xm
print(f"학습 {time.time() - t0:.0f} s", flush=True)
LEARNED = {"어댑터 순차 제거 (일반 학습)": (m_plain, feat_adapter), "어댑터 순차 제거 (제거 학습)": (m_defl, feat_adapter),
           "마스크 순차 제거 (제거 학습)": (m_mask, feat_mask)}


def make_events(h, K):
    """반환: Y (n, T, 64), 정답 정점 (n, K) — 센 순서로 정렬"""
    G = heads[h]["Gt"]
    idx = np.zeros((N_TE, K), int)
    for i in range(N_TE):
        c = []
        while len(c) < K:
            v = rng.integers(N_SRC)
            if all(np.linalg.norm(pos0[v] - pos0[u]) >= MIN_SEP for u in c):
                c.append(v)
        idx[i] = c
    t = np.arange(T) / FS
    Y = np.zeros((N_TE, T, 64))
    strength = np.zeros((N_TE, K))
    for k in range(K):
        g = rng.uniform(*AMP, N_TE)[:, None] * np.einsum("cnk,nk->nc", G[:, idx[:, k], :], dirs(N_TE, TEST_TILT))
        f, ph = rng.uniform(6, 15, N_TE), rng.uniform(0, 2 * np.pi, N_TE)
        tc, sg = rng.uniform(0.04, 0.16, N_TE), rng.uniform(0.02, 0.05, N_TE)
        s = np.sin(2 * np.pi * f[:, None] * t[None] + ph[:, None]) * np.exp(-((t[None] - tc[:, None]) ** 2) / (2 * sg[:, None] ** 2))
        Y += s[:, :, None] * g[:, None, :]
        strength[:, k] = np.linalg.norm(ref_(g), axis=1) * np.sqrt(np.mean(s**2, 1))
    sig = np.sqrt(np.mean(ref_(Y) ** 2, axis=(1, 2), keepdims=True) / 10 ** (SNR_DB / 10))
    Y += rng.normal(size=Y.shape) * sig
    order = np.argsort(-strength, axis=1)
    return Y, np.take_along_axis(idx, order, 1)


def peaks(score, posh, k):
    out = []
    for v in np.argsort(-score):
        if all(np.linalg.norm(posh[v] - posh[u]) >= PEAK_SEP for u in out):
            out.append(v)
            if len(out) == k:
                break
    return out


rows, hist = [], {}
BINS = np.arange(0, 102, 2)


def score(cond, method, h, K, found, truth):
    """found (n, K) 찾은 순서, truth (n, K) 센 순서"""
    posh = heads[h]["pos"]
    E = np.zeros((len(found), K))
    top1 = np.zeros(len(found), bool)
    for i in range(len(found)):
        D = np.linalg.norm(posh[truth[i]][:, None] - posh[found[i]][None], axis=2)   # (정답 순위, 찾은 순서)
        r, c = linear_sum_assignment(D)
        E[i, r] = D[r, c]
        top1[i] = D[0, 0] <= HIT                    # 첫 번째로 찾은 것이 가장 센 신호원인가
    rk = [E[:, j].mean() for j in range(K)] + [float("nan")] * (3 - K)
    rows.append((SEED, cond, method, NAMES[h], K, E.mean(), 100 * np.mean(E > HIT), 100 * top1.mean(), *rk))
    key = (cond, method, K)
    hist[key] = hist.get(key, 0) + np.histogram(np.minimum(E.ravel(), 100), BINS)[0]


n64 = mne.channels.make_standard_montage("biosemi64").ch_names
EVAL = {"64채널": ALL}
for name, std in (("32채널", "biosemi32"), ("16채널", "biosemi16")):
    EVAL[name] = np.array([n64.index(c) for c in mne.channels.make_standard_montage(std).ch_names])
t1 = time.time()
for cond, sel in EVAL.items():
    for h in TEST_HEADS:
        G = heads[h]["Gm"]
        Ks = slor_kernel(G, sel)
        Q2 = np.transpose(qbasis(G, sel, np.arange(N_SRC)), (0, 2, 1)).reshape(-1, len(sel))     # (source*3, k)
        for K in (1, 2, 3):
            Y, truth = make_events(h, K)
            R0 = ref_(Y[:, :, sel])
            # 한 시점, 봉우리
            tmax = np.argmax((R0**2).sum(2), axis=1)
            M = np.abs(R0[np.arange(N_TE), tmax] @ Ks.T)
            score(cond, "한 시점, sLORETA 봉우리", h, K, np.array([peaks(M[i], heads[h]["pos"], K) for i in range(N_TE)]), truth)
            # 순차 제거
            methods = {"sLORETA 순차 제거": None, "쌍극자 스캔 순차 제거": "scan", "MUSIC 순차 제거": "music", **({} if SCAN_ONLY else LEARNED)}
            for name, ml in methods.items():
                R = R0.copy()
                found = np.zeros((N_TE, K), int)
                for r in range(K):
                    Vh = np.linalg.svd(R, full_matrices=False)[2]
                    u = Vh[:, 0, :]                                                  # (n, k) 가장 두드러진 무늬
                    if ml is None:
                        v = np.abs(u @ Ks.T).argmax(1)
                    elif ml == "scan":
                        v = ((u @ Q2.T) ** 2).reshape(N_TE, -1, 3).sum(2).argmax(1)
                    elif ml == "music":
                        C = np.einsum("nmk,vk->nvm", Vh[:, :K - r, :], Q2).reshape(N_TE, N_SRC, 3, K - r)
                        v = eigmax3(C @ np.swapaxes(C, -1, -2)).argmax(1)
                    else:
                        v = tree0.query(ml[0].predict(ml[1](u, h, sel, Ks)) * 100.0)[1]
                    found[:, r] = v
                    R = remove(R, qbasis(G, sel, v))
                score(cond, name, h, K, found, truth)
    for K in (1, 2, 3):
        print(f" [{cond}, 신호원 {K}개] " + "  ".join(f"{m.split(' (')[0][:12]}{'(' + m.split('(')[1] if '(' in m else ''}: {np.mean([x[5] for x in rows if x[1] == cond and x[2] == m and x[4] == K]):.1f}"
                                                  for m in dict.fromkeys(x[2] for x in rows)), flush=True)
print(f"평가 {time.time() - t1:.0f} s, 전체 {time.time() - t0:.0f} s")

tag = f"seed{SEED}{'_quick' if QUICK else ''}"
PRE = "scan_ranked" if SCAN_ONLY else "ranked"
with open(os.path.join(HERE, f"{PRE}_{tag}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "배치", "방법", "머리", "신호원수", "평균mm", "놓침%", "첫번째가가장센것%", "순위1mm", "순위2mm", "순위3mm"])
    for r in rows:
        w.writerow(list(r[:5]) + [f"{x:.2f}" for x in r[5:]])
with open(os.path.join(HERE, f"{PRE}_hist_{tag}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["배치", "방법", "신호원수"] + [f"{b}" for b in BINS[:-1]])
    for (c, m, K), v in hist.items():
        w.writerow([c, m, K] + list(v))
print("저장: ranked_" + tag + ".csv")
