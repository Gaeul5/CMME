"""Week 9-4: 실제 뇌 모양(FastSurfer) OASIS-3 28명으로 학습 머리 수 곡선 다시 그리기

week06/oasis_headcount.py(변형 머리)와 같은 설정. 다른 점: 머리가 각자의 실제 피질(fsaverage ico-4 대응, 5,124곳).
모델 출력 좌표는 학습 머리 중 첫 번째 머리(POOL[0])의 위치 → 같은 번호 → 평가 머리의 위치로 바꿔 오차 계산.
실행: SEED=0 python week09/real_headcount.py
"""
import csv, glob, os, time, warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
rng = np.random.default_rng(19000 + SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR, N_TE, N_PCA = 30000, 600, 256
SNR_DB, AMP = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
COUNTS = [1, 3, 6, 12, 22]
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)
files = sorted(glob.glob(os.path.join(HERE, "cache_real", "OAS*.npz")))
assert len(files) == 28, len(files)
heads = [dict(Gt=np.load(f)["G_true"].astype(np.float64), Gm=np.load(f)["G_model"].astype(np.float64), pos=np.load(f)["pos"] * (1000.0 if np.abs(np.load(f)["pos"]).max() < 1 else 1.0)) for f in files]
NAMES = [os.path.basename(f)[:8] for f in files]
perm = rng.permutation(len(heads))
TEST, POOL = sorted(perm[:6].tolist()), perm[6:].tolist()
REF = POOL[0]; pos0, G0 = heads[REF]["pos"], heads[REF]["Gm"]; tree0 = cKDTree(pos0)
N_SRC = len(pos0)

def simulate(G3, n, tilt):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt, n)); ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    clean = rng.uniform(*AMP, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    r = clean - clean.mean(1, keepdims=True)
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (SNR_DB / 10)), idx
def ref_(X): return X - X.mean(1, keepdims=True)
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
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
def rmont(k=None): return np.sort(rng.choice(64, rng.integers(8, 65) if k is None else k, replace=False))

t0 = time.time(); maps = []
for j in range(150):
    sel = np.arange(64) if j % 5 == 0 else rmont()
    Y, _ = simulate(G0, 40, TRAIN_TILT); maps.append(unit(ref_(Y[:, sel]) @ slor(G0, sel).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]; del maps
EVAL = {"표준 64채널": np.arange(64), "처음 보는 16채널": rmont(16)}
DATA = {h: simulate(heads[h]["Gt"], N_TE, TEST_TILT) for h in TEST}
rows = []
for cond, sel in EVAL.items():
    for h in TEST:
        Y, it = DATA[h]; P = heads[h]["pos"]
        for m, v in (("sLORETA (자기 머리)", np.abs(ref_(Y[:, sel]) @ slor(heads[h]["Gm"], sel).T).argmax(1)),
                     ("쌍극자 스캔 (자기 머리)", scan(heads[h]["Gm"], sel, Y[:, sel]))):
            rows.append([SEED, cond, m, 0, NAMES[h], np.linalg.norm(P[v] - P[it], axis=1).mean()])
for n in COUNTS:
    hs = POOL[:n]; per = N_TR // n; X, I = [], []
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
with open(os.path.join(HERE, f"real_headcount_seed{SEED}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["시드", "배치", "방법", "학습머리수", "머리", "평균mm"])
    for r in rows: w.writerow(r[:5] + [f"{r[5]:.2f}"])
print("저장")
