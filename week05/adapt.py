"""Week 5-5: 라벨 없는 '1분 기록'으로 그 사람에 맞추기 + 전극 위치 오차에 대한 민감도.

11/4 미팅 메모: "환자마다 다를 수 있으니 그 사람의 EEG 1분 레코딩으로 빠르게 어댑테이션"
여기서 맞추는 대상은 파형이 아니라 '머리 모델의 두개골 전도도'다. 전도도는 사람마다 다르고(문헌 범위가 넓음)
MRI로는 알 수 없어서, 실제 데이터에서 오차의 큰 원인이 될 수 있다.

E4 (적응)
  1) 평가 머리마다 정답 위치를 모르는 사건 N개(기본 200개, '1분 기록'으로 가정)를 받는다
  2) 후보 전도도 6개(0.004~0.015) 각각의 리드필드로 "쌍극자 하나로 얼마나 잘 설명되는가"(GOF)를 계산
       GOF(y) = max_i ||Q_i^T y||^2 / ||y||^2,  Q_i = i번째 위치의 3방향 리드필드 열의 정규직교 기저
  3) 평균 GOF가 가장 높은 전도도를 고른다 (라벨 불필요)
  4) 고른 전도도의 리드필드로 sLORETA와 '자기 머리 어댑터'를 다시 만든다. 어댑터 모델은 다시 학습하지 않는다
     (입력이 역연산자를 거치므로 리드필드만 바꿔 끼우면 됨. 5-3의 어댑터 구조가 이걸 가능하게 함)
  비교: 기본 전도도(0.006, 지금까지 모든 실험의 모델 머리) / 적응해서 고른 전도도 / 후보 중 진짜에 가장 가까운 값(상한)
E5 (전극 위치 오차)
  정답 데이터를 만드는 전극 위치를 0/5/10/15 mm 흔들고, 모델은 표준 위치를 그대로 쓸 때의 오차.
  OASIS 파이프라인의 자동 정합 오차(sample에서 전극-두피 거리 중앙값 5.6 mm)가 어느 정도 영향을 주는지 보기 위함

머리: heads.py의 변형 머리. 학습 head000~011, 평가 head012~017 (heads_variants.py로 후보 리드필드 생성 필요)
실행: SEED=0 python week05/adapt.py
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
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = os.environ.get("QUICK") == "1"
N_TR, N_TE = (6000, 200) if QUICK else (30000, 1000)
SNR_DB, AMP_RANGE = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA = 256
N_UNLAB = [10, 50, 200]
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

train = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "cache_heads", "head0*.npz")))[:12]]
Gm_train = [z["G_model"].astype(np.float64) for z in train]
pos0 = train[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)
var = {}
for f in sorted(glob.glob(os.path.join(os.environ.get("VAR_DIR", os.path.join(HERE, "cache_variants")), "head*.npz"))):
    z = np.load(f)
    var[int(os.path.basename(f)[4:7])] = dict(Gc=z["G_cand"].astype(np.float64), cands=z["cands"], Gj=z["G_jit"].astype(np.float64),
                                              jit=z["jitters"], skull=float(z["skull_true"]), pos=z["pos"])
TEST = sorted(var)
CANDS = var[TEST[0]]["cands"]
I_DEF = int(np.argmin(np.abs(CANDS - 0.006)))
print(f"학습 머리 {len(train)}개, 평가 머리 {len(TEST)}개, 후보 전도도 {list(CANDS)}, seed {SEED}")


def simulate(G3, n, tilt_max):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    clean = rng.uniform(*AMP_RANGE, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    ref = clean - clean.mean(1, keepdims=True)
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (SNR_DB / 10))
    return clean + rng.normal(size=clean.shape) * sigma, idx


def ref(Y):
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


# 어댑터: 기준 머리 피질 지도로 PCA 기저 (5-3과 같은 방식, 64채널)
maps = []
for _ in range(150):
    Y, _ = simulate(Gm_train[0], 40, TRAIN_TILT)
    maps.append(unit(ref(Y) @ slor_kernel(Gm_train[0]).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps


def adapter_feat(Y, K):
    return unit(ref(Y) @ (B @ K).T)


# 학습 (5-3 E1의 K=12와 같은 설정): 머리 12개의 모델 forward(전도도 0.006)
t = time.time()
Xs, Xa, I = [], [], []
for G in Gm_train:
    Y, idx = simulate(G, N_TR // len(Gm_train), TRAIN_TILT)
    Xs.append(unit(ref(Y)))
    Xa.append(adapter_feat(Y, slor_kernel(G)))
    I.append(idx)
I = np.concatenate(I)
m_sens = MLPRegressor(**MLP_KW).fit(np.vstack(Xs), pos0[I] / 100.0)
m_adap = MLPRegressor(**MLP_KW).fit(np.vstack(Xa), pos0[I] / 100.0)
print(f"학습 {time.time() - t:.0f} s")

rows = []


def err(h, j, idx):
    return np.linalg.norm(var[h]["pos"][j] - var[h]["pos"][idx], axis=1)


def log(exp, cond, method, e):
    rows.append((exp, cond, method, e.mean(), np.median(e), 100 * np.mean(e <= 10)))
    print(f"  {method:<30}: 평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}  10mm 이내 {100 * np.mean(e <= 10):5.1f}%", flush=True)


def gof_scores(Gc, Y):
    """후보 리드필드마다 평균 GOF. Gc: (후보, 채널, source, 3), Y: (사건, 채널)."""
    Yr = ref(Y)
    nrm = (Yr**2).sum(1)
    out = []
    R = np.eye(Gc.shape[1]) - 1.0 / Gc.shape[1]
    for G in Gc:
        A = np.einsum("ij,njk->nik", R, np.transpose(G, (1, 0, 2)))  # (source, 채널, 3), 평균 기준
        Q = np.linalg.qr(A)[0]
        p = np.einsum("nck,sc->snk", Q, Yr)
        out.append(((p**2).sum(2).max(1) / nrm).mean())
    return np.array(out)


# ---------------------------------------------------------------------------
# E4: 라벨 없는 기록으로 전도도 고르기
# ---------------------------------------------------------------------------
print("\n=== E4: 라벨 없는 기록으로 두개골 전도도 고르기 (전극 위치 오차 5 mm) ===")
J5 = int(np.where(var[TEST[0]]["jit"] == 5.0)[0][0])
picks = {n: [] for n in N_UNLAB}
E = {k: [] for k in ["sLORETA 기본(0.006)", "sLORETA 적응", "sLORETA 상한(진짜에 가장 가까운 후보)",
                     "MLP 센서만", "어댑터 기본(0.006)", "어댑터 적응", "어댑터 상한(진짜에 가장 가까운 후보)"]}
for h in TEST:
    v = var[h]
    Gt = v["Gj"][J5]
    i_true = int(np.argmin(np.abs(CANDS - v["skull"])))
    for n in N_UNLAB:
        Yu, _ = simulate(Gt, n, TEST_TILT)
        picks[n].append((h, v["skull"], i_true, int(np.argmax(gof_scores(v["Gc"], Yu)))))
    i_ad = picks[N_UNLAB[-1]][-1][3]
    Yt, it = simulate(Gt, N_TE, TEST_TILT)
    Ks = {i: slor_kernel(v["Gc"][i]) for i in {I_DEF, i_ad, i_true}}
    for name, i in (("기본(0.006)", I_DEF), ("적응", i_ad), ("상한(진짜에 가장 가까운 후보)", i_true)):
        E[f"sLORETA {name}"].append(err(h, np.abs(ref(Yt) @ Ks[i].T).argmax(1), it))
        E[f"어댑터 {name}"].append(err(h, tree0.query(m_adap.predict(adapter_feat(Yt, Ks[i])) * 100)[1], it))
    E["MLP 센서만"].append(err(h, tree0.query(m_sens.predict(unit(ref(Yt))) * 100)[1], it))
for n in N_UNLAB:
    hit = np.mean([p[2] == p[3] for p in picks[n]])
    near = np.mean([abs(p[2] - p[3]) <= 1 for p in picks[n]])
    print(f" 사건 {n:>3}개: 진짜에 가장 가까운 후보를 고른 비율 {100 * hit:.0f}%, 한 칸 이내 {100 * near:.0f}%  "
          + ", ".join(f"h{p[0]}:{p[1]:.4f}->{CANDS[p[3]]:.3f}" for p in picks[n]))
    rows.append(("E4", f"선택 정확도 / 사건 {n}개", "정확히 맞힘 %", 100 * hit, 100 * near, float("nan")))
for k, e in E.items():
    log("E4", "평가 머리 6개, 사건 200개로 적응", k, np.concatenate(e))

# ---------------------------------------------------------------------------
# E5: 전극 위치 오차
# ---------------------------------------------------------------------------
print("\n=== E5: 전극 위치 오차 (모델은 표준 위치, 전도도 0.006) ===")
for jj, j in enumerate(var[TEST[0]]["jit"]):
    print(f" [오차 {j:.0f} mm]")
    E = {"sLORETA (자기 머리)": [], "MLP 센서만": [], "어댑터 (자기 머리)": []}
    for h in TEST:
        v = var[h]
        Yt, it = simulate(v["Gj"][jj], N_TE, TEST_TILT)
        K = slor_kernel(v["Gc"][I_DEF])
        E["sLORETA (자기 머리)"].append(err(h, np.abs(ref(Yt) @ K.T).argmax(1), it))
        E["MLP 센서만"].append(err(h, tree0.query(m_sens.predict(unit(ref(Yt))) * 100)[1], it))
        E["어댑터 (자기 머리)"].append(err(h, tree0.query(m_adap.predict(adapter_feat(Yt, K)) * 100)[1], it))
    for k, e in E.items():
        log("E5", f"전극 오차 {j:.0f} mm", k, np.concatenate(e))

out = os.path.join(HERE, f"adapt_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["실험", "조건", "방법", "평균mm(선택실험은 정확%)", "중앙값mm(선택실험은 한칸이내%)", "10mm이내%"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.1f}"])
print("결과 저장:", out)
