"""Week 6-7c: 6-4를 OASIS-3 머리 10명으로 다시 (시드마다 학습 6명 / 평가 4명). 아래는 6-4 설명 그대로.

Week 6-4: 신호원 여러 개, 넓게 퍼진 신호원 (선행 연구와 같은 과제로 넓히기)

선행 연구와 비교했을 때 가장 큰 약점: 지금 모델은 '쌍극자 1개의 위치' 하나만 낸다.
ConvDip, DeepSIF, 3D-PIUNet은 넓게 퍼진 신호원 여러 개의 '피질 지도'를 낸다.
→ 같은 과제에서도 어댑터(리드필드 → 공유 표현)가 통하는지 확인한다.

신호원 (한 시점)
  - 개수 1~3개, 중심끼리 30 mm 이상 떨어뜨림
  - 크기: 반지름 0~15 mm (0이면 정점 1곳, 15 mm면 지름 3 cm 정도의 피질 조각)
  - 조각 안 정점은 같은 세기, 조각마다 세기 0.5~1.5, 부호 무작위. 방향은 피질에 수직
  - 학습은 모델 리드필드(Gm), 평가는 정답 리드필드(Gt: 전도도·전극 위치가 모델과 다름), SNR 10 dB
출력 (DeepSIF처럼 영역 단위)
  - 기준 머리 피질을 k-means로 994개 영역으로 나누고, 영역마다 '켜진 정도'(영역 안 정점 중 켜진 비율)를 맞힌다
  - 정점 번호가 머리 사이에 대응하므로 영역도 모든 머리에 그대로 쓸 수 있다
모델 (같은 MLP: 입력 → 512 → 512 → 994, 출력 시그모이드, BCE 손실, PyTorch)
  - 범용 마스크: 64칸 값 + 64칸 마스크
  - 범용 어댑터: 그 머리·그 배치의 sLORETA 역연산자 → 공유 PCA 256
  - 둘 다 머리 12개 x 무작위 배치(3~64채널)로 학습한 모델 1개씩
  - 비교: sLORETA, MNE (자기 머리 리드필드, 정점 단위 |지도|)
평가: 학습에 안 쓴 머리 6개, 표준 배치 64/32/16 + 처음 보는 무작위 배치(채널 8/16/32)
지표 (선행 연구와 같은 종류)
  - 위치 오차: 지도에서 봉우리를 신호원 개수만큼 고르고(서로 20 mm 이상 떨어진 최댓값), 정답 중심과 짝지은 거리 평균 (mm)
    · 신호원 개수는 안다고 가정 (실제로는 모름. 한계)
  - AUC: 정점마다 '켜졌나'를 지도 값으로 얼마나 잘 가르는지 (1이면 완벽, 0.5면 무작위)
  - 놓친 비율: 정답 중심 20 mm 안에 봉우리가 없는 신호원의 비율

실행: SEED=0 python week06/extended_sources.py
"""
import csv
import glob
import os
import time
import warnings
import numpy as np
import mne
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(2000 + SEED)
torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count())
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 12000 if QUICK else 60000
N_TE = 40 if QUICK else 150             # 머리 1개 x 배치 1개당
N_PCA, N_PARC = 256, 994
SNR_DB = 10.0
MIN_SEP, MAX_R, PEAK_SEP, HIT_MM = 30.0, 15.0, 20.0, 20.0
EPOCHS = 8 if QUICK else 30

files = [os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz")] + \
    sorted(glob.glob(os.path.join(HERE, "cache_oasis", "OAS*.npz")))   # 0번은 기준 좌표용(sample), 학습·평가에는 안 씀
assert len(files) == 11
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"][:, :, 2].astype(np.float64), Gm=z["G_model"][:, :, 2].astype(np.float64),
                      Gm3=z["G_model"].astype(np.float64), pos=z["pos"].astype(np.float64)))
_perm = np.random.default_rng(4000 + SEED).permutation(np.arange(1, len(heads)))
TEST_HEADS, TRAIN_HEADS = sorted(_perm[:4].tolist()), sorted(_perm[4:].tolist())
NAMES = [os.path.basename(f)[:8] for f in files]
print("학습", [NAMES[i] for i in TRAIN_HEADS], "평가", [NAMES[i] for i in TEST_HEADS], flush=True)
pos0 = heads[0]["pos"]
N_SRC = len(pos0)
hemi = np.r_[np.zeros(3732, int), np.ones(N_SRC - 3732, int)]   # head000의 vert_lh 개수 (반구를 섞지 않음)
ALL = np.arange(64)

# 영역 994개 (반구별 k-means, 정점 수 비율로 나눔)
t0 = time.time()
parc = np.zeros(N_SRC, int)
off = 0
for hh in (0, 1):
    ii = np.where(hemi == hh)[0]
    k = int(round(N_PARC * len(ii) / N_SRC)) if hh == 0 else N_PARC - off
    parc[ii] = KMeans(k, n_init=1, random_state=0).fit_predict(pos0[ii]) + off
    off += k
P = np.zeros((N_PARC, N_SRC))
P[parc, np.arange(N_SRC)] = 1.0
P /= P.sum(1, keepdims=True)                     # 정점 → 영역 평균
cent = np.array([pos0[parc == j].mean(0) for j in range(N_PARC)])
dcent = np.linalg.norm(pos0 - cent[parc], axis=1)      # 영역 값은 영역 안에서 같으므로, 봉우리는 영역 중심에 가장 가까운 정점으로
trees = [cKDTree(h["pos"]) for h in heads]
tree0 = cKDTree(pos0)
print(f"영역 {N_PARC}개 (영역당 정점 평균 {N_SRC / N_PARC:.1f}), {time.time() - t0:.0f} s", flush=True)


def make_sources(n):
    """반환: 정점 세기 행렬 (n x N_SRC, 희소하게 채움), 중심 정점 목록, 켜진 정점 마스크"""
    S = np.zeros((n, N_SRC))
    centers = []
    for i in range(n):
        k = rng.integers(1, 4)
        c = []
        while len(c) < k:
            v = rng.integers(N_SRC)
            if all(np.linalg.norm(pos0[v] - pos0[u]) >= MIN_SEP for u in c):
                c.append(v)
        for v in c:
            r = rng.uniform(0, MAX_R)
            nb = [j for j in tree0.query_ball_point(pos0[v], r) if hemi[j] == hemi[v]] if r > 0 else [v]
            S[i, nb] = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
        centers.append(c)
    return S, centers


def simulate(G, S):
    clean = S @ G.T
    ref = clean - clean.mean(1, keepdims=True)
    sigma = np.sqrt(np.mean(ref**2, 1, keepdims=True) / 10 ** (SNR_DB / 10))
    return clean + rng.normal(size=clean.shape) * sigma


def pick_ref(Y, sel):
    X = Y[:, sel]
    return X - X.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def kernels(G, sel, lam2=1.0 / 9.0):
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
    return K, K / np.sqrt(res)[:, None]


def random_montage(k=None):
    k = rng.integers(3, 65) if k is None else k
    return np.sort(rng.choice(64, k, replace=False))


# 공유 PCA 기저 (기준 머리, 여러 신호원 지도로)
maps = []
for j in range(40 if QUICK else 120):
    sel = ALL if j % 5 == 0 else random_montage()
    S, _ = make_sources(40)
    maps.append(unit(pick_ref(simulate(heads[0]["Gm"], S), sel) @ kernels(heads[0]["Gm"], sel)[1].T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps


def feat_mask(Y, h, sel, K=None):
    X = np.zeros((len(Y), 128))
    X[:, sel] = unit(pick_ref(Y, sel))
    X[:, 64 + np.asarray(sel)] = 1.0
    return X


def feat_adapter(Y, h, sel, K=None):
    K = kernels(heads[h]["Gm"], sel)[1] if K is None else K
    return unit(pick_ref(Y, sel) @ (B @ K).T)


FEATS = {"범용 1개: 마스크": feat_mask, "범용 1개: 어댑터": feat_adapter}
pool = [random_montage() for _ in range(60 if QUICK else 200)]
pool_keys = {tuple(s) for s in pool}

t1 = time.time()
X = {k: [] for k in FEATS}
T = []
per = N_TR // len(TRAIN_HEADS)
for h in TRAIN_HEADS:
    S, _ = make_sources(per)
    Y = simulate(heads[h]["Gm"], S)
    a = rng.integers(0, len(pool), per)
    for m in np.unique(a):
        ii = np.where(a == m)[0]
        for k, f in FEATS.items():
            X[k].append(f(Y[ii], h, pool[m]))
        T.append(((S[ii] != 0).astype(float)) @ P.T)
X = {k: np.vstack(v).astype(np.float32) for k, v in X.items()}
T = np.vstack(T).astype(np.float32)
print(f"학습 데이터 {len(T)}개 만들기 {time.time() - t1:.0f} s", flush=True)


def train(Xn):
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6          # 입력 표준화 (크기 1로 맞춘 입력은 값이 작아 학습이 느림)
    Xn = (Xn - mu) / sd
    n_in = Xn.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.ReLU(),
                              torch.nn.Linear(512, N_PARC))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt, Tt = torch.from_numpy(Xn), torch.from_numpy(T)
    nv = len(Xt) // 10
    idx = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(SEED))
    va, tr = idx[:nv], idx[nv:]
    pw = torch.tensor(float((1 - T.mean()) / T.mean()) ** 0.5)   # 켜진 영역이 드물어서 양성에 가중치
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    best, state = 1e9, None
    for ep in range(EPOCHS):
        net.train()
        for b in tr[torch.randperm(len(tr))].split(256):
            opt.zero_grad()
            loss = lossf(net(Xt[b]), Tt[b])
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Xt[va]), Tt[va]).item()
        if v < best:
            best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    net.mu, net.sd = torch.from_numpy(mu), torch.from_numpy(sd)
    return net


models = {}
for k in FEATS:
    t2 = time.time()
    models[k] = train(X[k])
    print(f"{k} 학습 {time.time() - t2:.0f} s", flush=True)
del X, T


def peaks(score, posh, k):
    """서로 PEAK_SEP 이상 떨어진 최댓값 k개"""
    order = np.argsort(-score)
    out = []
    for v in order:
        if all(np.linalg.norm(posh[v] - posh[u]) >= PEAK_SEP for u in out):
            out.append(v)
            if len(out) == k:
                break
    return out


rows = []


def evaluate(cond, sel):
    for h in TEST_HEADS:
        S, centers = make_sources(N_TE)
        Y = simulate(heads[h]["Gt"], S)
        K_mne, K_slor = kernels(heads[h]["Gm"], sel)
        Xr = pick_ref(Y, sel)
        maps_ = {"MNE (자기 머리)": np.abs(Xr @ K_mne.T), "sLORETA (자기 머리)": np.abs(Xr @ K_slor.T)}
        for k, f in FEATS.items():
            with torch.no_grad():
                m_ = models[k]
                p = torch.sigmoid(m_((torch.from_numpy(f(Y, h, sel, K_slor).astype(np.float32)) - m_.mu) / m_.sd)).numpy()
            maps_[k] = p[:, parc] - 1e-9 * dcent          # 영역 값 → 정점 (같은 값이면 영역 중심 쪽이 봉우리)
        posh = heads[h]["pos"]
        for name, M in maps_.items():
            le, auc, miss = [], [], []
            for i in range(N_TE):
                c = centers[i]
                pk = peaks(M[i], posh, len(c))
                D = np.linalg.norm(posh[c][:, None] - posh[pk][None], axis=2)
                r_, c_ = linear_sum_assignment(D)
                d = D[r_, c_]
                le.append(d.mean())
                miss.append(np.mean(d > HIT_MM))
                auc.append(roc_auc_score(S[i] != 0, M[i]))
            n_src = np.array([len(c) for c in centers])
            for g, gm in (("전체", n_src > 0), ("신호원 1개", n_src == 1), ("신호원 2~3개", n_src > 1)):
                rows.append((SEED, cond, name, NAMES[h], g, np.mean(np.array(le)[gm]), np.mean(np.array(auc)[gm]),
                             100 * np.mean(np.array(miss)[gm])))


n64 = mne.channels.make_standard_montage("biosemi64").ch_names
EVAL = {"표준 64채널": ALL}
for name, std in (("표준 32채널", "biosemi32"), ("표준 16채널", "biosemi16")):
    EVAL[name] = np.array([n64.index(c) for c in mne.channels.make_standard_montage(std).ch_names])
for c in (8, 16, 32):
    s = random_montage(c)
    while tuple(s) in pool_keys:
        s = random_montage(c)
    EVAL[f"처음 보는 배치 {c}채널"] = s
t3 = time.time()
for cond, sel in EVAL.items():
    evaluate(cond, sel)
    r = [x for x in rows if x[1] == cond and x[4] == "전체"]
    print(f" [{cond}]", flush=True)
    for k in ["MNE (자기 머리)", "sLORETA (자기 머리)"] + list(FEATS):
        rr = [x for x in r if x[2] == k]
        print(f"   {k:<20} 위치 오차 {np.mean([x[5] for x in rr]):5.1f} mm  AUC {np.mean([x[6] for x in rr]):.3f}  "
              f"놓침 {np.mean([x[7] for x in rr]):4.1f}%", flush=True)
print(f"평가 {time.time() - t3:.0f} s, 전체 {time.time() - t0:.0f} s")

out = os.path.join(HERE, f"oasis_extended_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "조건", "방법", "머리", "신호원수", "위치오차mm", "AUC", "놓침%"])
    for r in rows:
        w.writerow(list(r[:5]) + [f"{r[5]:.2f}", f"{r[6]:.4f}", f"{r[7]:.1f}"])
print("저장:", out)
