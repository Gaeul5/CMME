"""Week 5-3: 여러 머리로 학습하고, 학습에 안 쓴 머리(hold-out)에서 평가.

11/4 미팅 메모 대응
  - "MRI 50개 규모, 50개 밖의 새 데이터에서도 잘 뽑는지 확인" → 학습 머리 수 K = 1, 3, 6, 12, 평가는 따로 둔 머리 6개
  - "MRI에서 리드필드를 얻어 입력으로 넣는 게 더 잘 되는 것 같다" → 리드필드를 쓰는 방법 두 가지를 비교
      (a) 머리 벡터: 리드필드를 16차원으로 줄인 '머리 요약 벡터'를 센서 입력 옆에 붙임
          (세미나에서 본 global MRI 벡터와 같은 계열. 학습 머리를 외우는 지름길이 될 수 있음)
      (b) 자기 머리 어댑터: 그 머리의 리드필드로 만든 sLORETA 역연산자로 센서 → 피질 지도로 바꾼 뒤 입력
          (5-1의 몽타주 어댑터를 머리에도 적용. 피질 vertex 번호가 머리 사이에 대응하므로 입력 공간이 같아짐)
  - "몽타주와 머리를 같이" → 실험 E2: 무작위 몽타주 + 여러 머리로 모델 1개를 학습하고 새 머리 + 새 몽타주로 평가

머리는 heads.py로 만든 sample 변형 머리다 (실제 다른 사람 MRI가 아님. 한계는 README 참고).
출력은 모두 '기준 머리(head000)의 좌표'로 예측하고, 가장 가까운 vertex를 찾아 그 머리에서의 실제 위치로 오차를 잰다.
sLORETA도 vertex를 고르므로 같은 방식으로 채점된다.

실행: SEED=0 python week05/multihead.py   (week05/cache_heads/head000~017.npz 필요)
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
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 8000 if QUICK else 30000         # 학습 샘플 총수 (머리 수와 무관하게 고정 → 데이터 양이 아니라 머리 다양성의 효과)
N_TE = 300 if QUICK else 1000           # 평가 머리 1개당
SNR_DB, AMP_RANGE = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA, N_DESC = 256, 16
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)
SPLIT = os.environ.get("SPLIT")          # 실제 MRI용: {"train": [npz...], "test": [npz...]} (oasis_split.py가 만듦)


def load(f):
    z = np.load(f)
    return dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"],
                scale=float(z["scale"]), thick=float(z["thick_mm"]), skull=float(z["skull_true"]),
                valid=z["valid"] if "valid" in z.files else np.ones(len(z["pos"]), bool),
                name=os.path.basename(f))


STRONG_HEADS = []
if SPLIT:
    import json
    sp = json.load(open(SPLIT))
    heads = [load(f) for f in sp["train"]] + [load(f) for f in sp["test"]]
    N_TRAIN_POOL = len(sp["train"])
    TEST_HEADS = list(range(N_TRAIN_POOL, len(heads)))
    K_LIST = sorted({k for k in [1, 3, 6, 12, 25, 50] if k < N_TRAIN_POOL} | {N_TRAIN_POOL})
else:  # 변형 머리 (heads.py): head000~011 학습 후보, head012~017 평가, cache_heads_strong은 추가 평가
    files = sorted(glob.glob(os.path.join(os.environ.get("HEADS_DIR", os.path.join(HERE, "cache_heads")), "head*.npz")))
    assert len(files) >= 18, f"머리가 {len(files)}개뿐. heads.py로 18개를 먼저 만들 것"
    heads = [load(f) for f in files[:18]]
    N_TRAIN_POOL, TEST_HEADS = 12, list(range(12, 18))
    K_LIST = [1, 3, 6] if QUICK else [1, 3, 6, 12]
    for f in sorted(glob.glob(os.path.join(HERE, "cache_heads_strong", "head*.npz"))):
        STRONG_HEADS.append(len(heads))
        heads.append(load(f))

# 모든 머리에서 유효한 source만 사용 (실제 MRI에서는 두개골 밖으로 나간 정점이 머리마다 다를 수 있음)
common = np.logical_and.reduce([h["valid"] for h in heads])
for h in heads:
    h["Gt"], h["Gm"], h["pos"] = h["Gt"][:, common], h["Gm"][:, common], h["pos"][common]
pos0 = heads[0]["pos"]          # 기준 좌표 = 첫 번째 학습 머리
tree0 = cKDTree(pos0)
N_SRC = len(pos0)
print(f"머리 {len(heads)}개 (학습 후보 {N_TRAIN_POOL}, 평가 {len(TEST_HEADS)}, 강한 변형 {len(STRONG_HEADS)}), "
      f"공통 source {N_SRC}개, seed {SEED}{' (QUICK)' if QUICK else ''}")
d = [np.linalg.norm(heads[i]["pos"] - pos0, axis=1).mean() for i in range(1, len(heads))]
if d:
    print(f"기준 머리와 source 위치 차이: 평균 {np.mean(d):.1f} mm (범위 {np.min(d):.1f}~{np.max(d):.1f})")


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


ALL = np.arange(64)
_kc = {}


def kernel(h, sel):
    key = (h, tuple(sel))
    if key not in _kc:
        _kc[key] = slor_kernel(heads[h]["Gm"], sel)
    return _kc[key]


# 공유 PCA 기저: 기준 머리의 피질 지도로 한 번 계산
def random_montage():
    return np.sort(rng.choice(64, rng.integers(3, 65), replace=False))


maps = []
for j in range(60 if QUICK else 150):
    sel = ALL if j % 5 == 0 else random_montage()
    Y, _ = simulate(heads[0]["Gm"], 40, TRAIN_TILT)
    maps.append(unit(pick_ref(Y, sel) @ slor_kernel(heads[0]["Gm"], sel).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps
_pc = {}


def projector(h, sel):
    key = (h, tuple(sel))
    if key not in _pc:   # 역연산자(7,498 x k)는 저장하지 않고 256 x k 투영만 저장 (메모리)
        _pc[key] = B @ slor_kernel(heads[h]["Gm"], sel)
    return _pc[key]


# 머리 요약 벡터: 모델 리드필드(수직 성분)를 고정된 무작위 투영으로 16차원으로 줄임 (기준 머리와의 차이)
R_desc = np.random.default_rng(7).normal(size=(N_DESC, 64 * N_SRC)) / np.sqrt(64 * N_SRC)
g0 = heads[0]["Gm"][:, :, 2].ravel()
desc = np.array([R_desc @ ((h["Gm"][:, :, 2].ravel() - g0) / np.linalg.norm(g0)) for h in heads])
desc = desc / (desc[1:N_TRAIN_POOL].std() + 1e-12) if N_TRAIN_POOL > 1 else desc
del R_desc

rows = []


def score(h, j, idx):
    return np.linalg.norm(heads[h]["pos"][j] - heads[h]["pos"][idx], axis=1)


def to_vertex(xyz):
    return tree0.query(xyz)[1]


def log(exp, cond, method, e):
    rows.append((exp, cond, method, e.mean(), np.median(e), 100 * np.mean(e <= 10)))
    print(f"  {method:<24}: 평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}  10mm 이내 {100 * np.mean(e <= 10):5.1f}%", flush=True)


def fit(X, idx):
    return MLPRegressor(**MLP_KW).fit(X, pos0[idx] / 100.0)


FEATS = {
    "MLP 센서만": lambda Y, h, sel: unit(pick_ref(Y, sel)),
    "MLP 센서 + 머리 벡터": lambda Y, h, sel: np.hstack([unit(pick_ref(Y, sel)), np.repeat(desc[h][None], len(Y), 0)]),
    "어댑터 (기준 머리 리드필드)": lambda Y, h, sel: unit(pick_ref(Y, sel) @ projector(0, sel).T),
    "어댑터 (자기 머리 리드필드)": lambda Y, h, sel: unit(pick_ref(Y, sel) @ projector(h, sel).T),
}


def build_inputs(head_ids, n, montage_list, feats):
    """머리별로 같은 수의 샘플, 샘플마다 몽타주 하나. 반환: {방법: X}, 정답 인덱스."""
    per = n // len(head_ids)
    X = {k: [] for k in feats}
    I = []
    for h in head_ids:
        Y, idx = simulate(heads[h]["Gm"], per, TRAIN_TILT)
        a = rng.integers(0, len(montage_list), per)
        for m in np.unique(a):
            ii = np.where(a == m)[0]
            sel = montage_list[m]
            for k in feats:
                X[k].append(FEATS[k](Y[ii], h, sel))
            I.append(idx[ii])
    return {k: np.vstack(v) for k, v in X.items()}, np.concatenate(I)


def evaluate(exp, cond, models, head_ids, sel, extra_slor=True):
    E = {k: [] for k in (["sLORETA (기준 머리)", "sLORETA (자기 머리)"] if extra_slor else []) + list(models)}
    for h in head_ids:
        Yt, it = simulate(heads[h]["Gt"], N_TE, TEST_TILT)   # 평가는 그 머리의 '정답' forward (전도도·전극 위치 다름)
        Xr = pick_ref(Yt, sel)
        if extra_slor:
            E["sLORETA (기준 머리)"].append(score(h, np.abs(Xr @ kernel(0, sel).T).argmax(1), it))
            E["sLORETA (자기 머리)"].append(score(h, np.abs(Xr @ kernel(h, sel).T).argmax(1), it))
        for k, m in models.items():
            p = m.predict(FEATS[k](Yt, h, sel)) * 100.0
            E[k].append(score(h, to_vertex(p), it))
    for k, v in E.items():
        log(exp, cond, k, np.concatenate(v))


# ---------------------------------------------------------------------------
# E1: 학습 머리 수 K에 따른 새 머리 성능 (64채널)
# ---------------------------------------------------------------------------
for K in K_LIST:
    t = time.time()
    train_ids = list(range(K))
    X, I = build_inputs(train_ids, N_TR, [ALL], list(FEATS))
    models = {k: fit(X[k], I) for k in FEATS}
    print(f"\n=== E1: 학습 머리 {K}개 (학습 {time.time() - t:.0f} s) ===")
    print(f" [새 머리 {len(TEST_HEADS)}개]")
    evaluate("E1", f"K={K} / 새 머리", models, TEST_HEADS, ALL)
    print(" [학습에 쓴 머리]")
    evaluate("E1", f"K={K} / 학습 머리", models, train_ids[:6], ALL)
    if K == K_LIST[-1] and STRONG_HEADS:
        print(f" [E3: 더 크게 변형한 새 머리 {len(STRONG_HEADS)}개, 학습 머리 {K}개 모델]")
        evaluate("E3", f"K={K} / 강한 변형 머리", models, STRONG_HEADS, ALL)

# ---------------------------------------------------------------------------
# E2: 머리 12개 + 무작위 몽타주로 모델 1개 학습 → 새 머리 + 처음 보는 몽타주
# ---------------------------------------------------------------------------
print(f"\n=== E2: 머리 {N_TRAIN_POOL}개 × 무작위 몽타주로 범용 모델 1개 ===")
pool = [random_montage() for _ in range(60 if QUICK else 200)]


def feat_mask(Y, h, sel):
    X = np.zeros((len(Y), 64))
    X[:, sel] = unit(pick_ref(Y, sel))
    M = np.zeros((len(Y), 64))
    M[:, sel] = 1.0
    return np.hstack([X, M])


FEATS["MLP 마스크"] = feat_mask
feats2 = ["MLP 마스크", "어댑터 (기준 머리 리드필드)", "어댑터 (자기 머리 리드필드)"]
t = time.time()
X, I = build_inputs(list(range(N_TRAIN_POOL)), 2 * N_TR, pool, feats2)
models2 = {k: fit(X[k], I) for k in feats2}
print(f" 학습 {time.time() - t:.0f} s")
NAMED = {"64채널": ALL,
         "32채널 (Biosemi32)": None, "16채널 (Biosemi16)": None}
try:
    import mne
    n64 = mne.channels.make_standard_montage("biosemi64").ch_names
    for name, std in (("32채널 (Biosemi32)", "biosemi32"), ("16채널 (Biosemi16)", "biosemi16")):
        NAMED[name] = np.array([n64.index(c) for c in mne.channels.make_standard_montage(std).ch_names])
except Exception:
    NAMED = {"64채널": ALL}
for name, sel in NAMED.items():
    print(f" [새 머리 {len(TEST_HEADS)}개 / {name}]")
    evaluate("E2", name, models2, TEST_HEADS, sel)

center = pos0.mean(0)
print(f"\n참고: 항상 중심을 답하면 평균 오차 약 {np.linalg.norm(pos0 - center, axis=1).mean():.1f} mm (기준 머리)")
out = os.path.join(HERE, f"multihead_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["실험", "조건", "방법", "평균mm", "중앙값mm", "10mm이내%"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.1f}"])
print("결과 저장:", out)
