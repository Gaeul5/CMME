"""Week 7-2: 정답 없는 신호로 추론 단계에서 적응하기 (test-time adaptation, TTA) — 시뮬레이션 시험

11/17 미팅: "Test time adaptation 한번 보고 인퍼런스 단계에서 적응 완료", "테스트 신호를 보고 오차를 줄일 수 있지 않을까"
상황: 변형 머리 12개로 학습한 모델을 처음 보는 종류의 머리(OASIS-3 10명)에 쓴다 (6-7b에서 오차가 커지던 경우).
      새 사람마다 정답 없는 기록 N개(50/200/1000)를 받아 적응하고, 따로 만든 1,000개로 평가한다.
입력 3가지: 센서만 / 어댑터 템플릿 머리 / 어댑터 자기 머리.  64채널, 쌍극자 1개, MLP (256,128), PyTorch
적응 방법 (모두 정답 위치를 쓰지 않음)
  없음
  입력 통계 맞추기 : 새 사람 입력의 평균·표준편차를 학습 때 값에 맞춤 (가장 단순한 방식)
  특징 정렬 (SSA)  : 중간 특징의 분포를 학습 때 분포에 맞추도록 모델을 조금 고침
                     (Adachi et al., "Test-time Adaptation for Regression by Subspace Alignment"의 방식을 단순하게 구현)
  물리 보정        : 모델이 찍은 위치 주변 20 mm 안에서, 그 사람 리드필드로 신호를 가장 잘 설명하는 곳(GOF 최대)으로 옮김.
                     학습 없이 신호마다 따로 함
  물리 자기학습    : 물리 보정으로 얻은 위치를 임시 정답으로 삼아 모델을 조금 고침 (그 뒤에는 보정 없이 모델만 사용)
참고
  정답으로 미세조정 (상한) : 같은 N개에 진짜 정답을 줬을 때. 적응으로 갈 수 있는 한계를 보려는 것
  sLORETA, 쌍극자 스캔(전체에서 GOF 최대) : 학습 없는 방법
한계: 하이퍼파라미터(학습률 1e-4, 100걸음, 반경 20 mm, 부분공간 32차원)는 한 번 정한 값이고 조정하지 않았다.

실행: SEED=0 python week07/tta.py
"""
import copy
import csv
import glob
import os
import time
import warnings
import numpy as np
import torch
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(11000 + SEED)
torch.manual_seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
N_TR = 6000 if QUICK else 30000
N_TE = 200 if QUICK else 1000
N_UNLAB = [50, 200] if QUICK else [50, 200, 1000]
SNR_DB, AMP = 10.0, (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
N_PCA, RADIUS, SUB = 256, 20.0, 32
EPOCHS, LR_TTA, STEPS = (8 if QUICK else 30), 1e-4, 100

fake = sorted(glob.glob(os.path.join(HERE, "..", "week05", "cache_heads", "head*.npz")))[:12]
oas = sorted(glob.glob(os.path.join(HERE, "..", "week06", "cache_oasis", "OAS*.npz")))
if QUICK:
    oas = oas[:1]
files = fake + oas
heads = []
for f in files:
    z = np.load(f)
    heads.append(dict(Gt=z["G_true"].astype(np.float64), Gm=z["G_model"].astype(np.float64), pos=z["pos"]))
NAMES = [os.path.basename(f).replace(".npz", "")[:8] for f in files]
TRAIN, TEST = list(range(12)), list(range(12, len(files)))
pos0 = heads[0]["pos"]
tree0 = cKDTree(pos0)
N_SRC = len(pos0)


def simulate(G3, n, tilt):
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    clean = rng.uniform(*AMP, n)[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    r = clean - clean.mean(1, keepdims=True)
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (SNR_DB / 10)), idx


def ref_(Y):
    return Y - Y.mean(-1, keepdims=True)


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


t0 = time.time()
maps = []
for _ in range(40 if QUICK else 150):
    Y, _ = simulate(heads[0]["Gm"], 40, TRAIN_TILT)
    maps.append(unit(ref_(Y) @ slor_kernel(heads[0]["Gm"]).T))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps
KS = [slor_kernel(h["Gm"]) for h in heads]
PROJ = [B @ K for K in KS]
FEATS = {"센서만": lambda Y, h: unit(ref_(Y)), "어댑터 템플릿 머리": lambda Y, h: unit(ref_(Y) @ PROJ[0].T),
         "어댑터 자기 머리": lambda Y, h: unit(ref_(Y) @ PROJ[h].T)}


class Net(torch.nn.Module):
    def __init__(self, n_in, mu, sd):
        super().__init__()
        self.mu, self.sd = torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU())
        self.head = torch.nn.Linear(128, 3)

    def feat(self, x):
        return self.body((x - self.mu) / self.sd)

    def forward(self, x):
        return self.head(self.feat(x))


def T_(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32)


def predict(net, X):
    net.eval()
    with torch.no_grad():
        return net(T_(X)).numpy() * 100.0


def finetune(net, X, target, steps=STEPS):
    """위치(기준 좌표, mm)를 목표로 모델 전체를 조금 고침"""
    net = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), LR_TTA)
    Xt, Yt = T_(X), T_(target / 100.0)
    g = torch.Generator().manual_seed(SEED)
    net.train()
    for _ in range(steps):
        b = torch.randint(0, len(Xt), (min(64, len(Xt)),), generator=g)
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(Xt[b]), Yt[b]).backward()
        opt.step()
    return net


def ssa(net, X, stats, steps=STEPS):
    """중간 특징을 학습 때의 주성분 부분공간에서 평균 0, 분산 λ에 맞춤 (양방향 KL, 출력에 영향 큰 방향에 가중치)"""
    mu_f, V, lam = stats
    net = copy.deepcopy(net)
    alpha = 1.0 + (net.head.weight.detach() @ V).abs().sum(0)
    opt = torch.optim.Adam(net.body.parameters(), LR_TTA)
    Xt = T_(X)
    g = torch.Generator().manual_seed(SEED)
    net.train()
    for _ in range(steps):
        b = torch.randint(0, len(Xt), (min(64, len(Xt)),), generator=g)
        z = (net.feat(Xt[b]) - mu_f) @ V
        m, v = z.mean(0), z.var(0) + 1e-6
        kl = 0.5 * ((v + m**2) / lam + (lam + m**2) / v - 2.0)
        opt.zero_grad()
        (alpha * kl).sum().backward()
        opt.step()
    return net


# 학습 (변형 머리 12개)
models, stats, in_stats = {}, {}, {}
per = N_TR // len(TRAIN)
data = [simulate(heads[h]["Gm"], per, TRAIN_TILT) for h in TRAIN]
I = np.concatenate([d[1] for d in data])
for k, f in FEATS.items():
    X = np.vstack([f(d[0], h) for d, h in zip(data, TRAIN)]).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    # 입력 크기 조절은 숫자 하나로만 한다. 성분마다 따로 나누면 분산이 작은 PCA 성분이 부풀어 처음 보는 머리에서 오차가 40 mm대로 커졌다
    net = Net(X.shape[1], np.zeros_like(mu), np.full_like(sd, X.std()))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt, Yt = T_(X), T_(pos0[I] / 100.0)
    for ep in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(Xt)).split(128):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xt[b]), Yt[b]).backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        Z = net.feat(Xt)
    mu_f = Z.mean(0)
    U, S, Vt = torch.linalg.svd(Z - mu_f, full_matrices=False)
    stats[k] = (mu_f, Vt[:SUB].T.contiguous(), (S[:SUB] ** 2 / (len(Z) - 1)) + 1e-6)
    models[k], in_stats[k] = net, (mu, sd)
del data
print(f"학습 {time.time() - t0:.0f} s (seed {SEED})", flush=True)

rows = []
for h in TEST:
    posh = heads[h]["pos"]
    A = np.transpose(ref_(np.transpose(heads[h]["Gm"], (1, 2, 0))), (0, 2, 1))     # (source, 64, 3) 평균 기준
    Q = np.linalg.qr(A)[0]
    Q2 = np.ascontiguousarray(np.transpose(Q, (0, 2, 1)).reshape(-1, Q.shape[1]))    # (source*3, 64)

    def gof(Y):
        Yr = ref_(Y)
        return ((Yr @ Q2.T) ** 2).reshape(len(Yr), -1, 3).sum(2) / (Yr**2).sum(1, keepdims=True)

    def refine(Y, v0, g=None):
        g = gof(Y) if g is None else g
        d2 = (posh**2).sum(1)[None, :] + (posh[v0] ** 2).sum(1)[:, None] - 2.0 * posh[v0] @ posh.T
        return np.where(d2 <= RADIUS**2, g, -1).argmax(1)

    def err(v, it):
        return np.linalg.norm(posh[v] - posh[it], axis=1).mean()

    Yt, it = simulate(heads[h]["Gt"], N_TE, TEST_TILT)
    g_te = gof(Yt)
    rows.append((SEED, "-", "sLORETA (자기 머리)", 0, NAMES[h], err(np.abs(ref_(Yt) @ KS[h].T).argmax(1), it)))
    rows.append((SEED, "-", "쌍극자 스캔 (자기 머리)", 0, NAMES[h], err(g_te.argmax(1), it)))
    U = {n: simulate(heads[h]["Gt"], n, TEST_TILT) for n in N_UNLAB}
    G_U = {n: gof(U[n][0]) for n in N_UNLAB}
    for k, f in FEATS.items():
        net, Xt = models[k], f(Yt, h).astype(np.float32)
        v0 = tree0.query(predict(net, Xt))[1]
        rows.append((SEED, k, "없음", 0, NAMES[h], err(v0, it)))
        rows.append((SEED, k, "물리 보정", 0, NAMES[h], err(refine(Yt, v0, g_te), it)))
        for n in N_UNLAB:
            Xu = f(U[n][0], h).astype(np.float32)
            # 입력 통계 맞추기
            mu, sd = in_stats[k]
            Xs = (Xt - Xu.mean(0)) / (Xu.std(0) + 1e-6) * sd + mu
            rows.append((SEED, k, "입력 통계 맞추기", n, NAMES[h], err(tree0.query(predict(net, Xs))[1], it)))
            # 특징 정렬
            rows.append((SEED, k, "특징 정렬 (SSA)", n, NAMES[h], err(tree0.query(predict(ssa(net, Xu, stats[k]), Xt))[1], it)))
            # 물리 자기학습
            vu = refine(U[n][0], tree0.query(predict(net, Xu))[1], G_U[n])
            net2 = finetune(net, Xu, pos0[vu])
            v2 = tree0.query(predict(net2, Xt))[1]
            rows.append((SEED, k, "물리 자기학습", n, NAMES[h], err(v2, it)))
            rows.append((SEED, k, "물리 자기학습 + 물리 보정", n, NAMES[h], err(refine(Yt, v2, g_te), it)))
            # 상한
            rows.append((SEED, k, "정답으로 미세조정 (상한)", n, NAMES[h], err(tree0.query(predict(finetune(net, Xu, pos0[U[n][1]]), Xt))[1], it)))
    r = [x for x in rows if x[4] == NAMES[h]]
    print(f" {NAMES[h]}: " + "  ".join(f"{x[1][:6]}/{x[2][:6]}{'' if not x[3] else x[3]}:{x[5]:.1f}" for x in r if x[3] in (0, N_UNLAB[-1])) + f"  ({time.time() - t0:.0f} s)", flush=True)

out = os.path.join(HERE, f"tta_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "입력", "적응 방법", "정답 없는 기록 수", "머리", "평균mm"])
    for r in rows:
        w.writerow(list(r[:5]) + [f"{r[5]:.2f}"])
print("저장:", out)
