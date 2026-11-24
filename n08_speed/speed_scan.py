"""노션 Week 8 · 목표 7: 쌍극자 스캔이 정말 빠른지 — 신호 1개당 계산 시간 측정

세 방법 (week09/real_headcount.py와 같은 계산):
  - sLORETA      : |ref(Y) K^T| 의 최댓값 위치 (K: 5,124곳 x C 역행렬)
  - 쌍극자 스캔   : 5,124곳 각각에서 3방향 쌍극자로 설명되는 에너지 ||Q_n^T ref(y)||^2 의 최댓값
  - 학습 모델     : unit(ref(Y) (B K)^T) -> MLP(256-128) -> 좌표 -> 가장 가까운 위치
조건: 채널 C = 16/64/256, 배치 = 1/100/1000, 장치 = CPU(numpy/sklearn) / GPU 0번(torch float32)
시간 = (Y가 CPU 메모리에 있는 상태 -> 위치 번호를 CPU로 받을 때까지) / 배치 크기.  GPU는 전송 포함.
역행렬·QR 등 배치(몽타주)마다 한 번 하는 준비 계산은 따로 잰다.

머리: week09/cache_real 의 OAS*.npz 첫 번째 머리 (G_model, 64채널).
  - 16채널: 64채널 중 무작위 16개
  - 256채널: 캐시에 없음 -> 64채널 열을 무작위 섞어 늘린 같은 크기 대체 행렬 (계산 시간은 크기에만 의존)
  - 캐시가 없으면 모든 채널 수에서 같은 크기의 무작위 리드필드를 쓴다 (출력 CSV '머리' 열에 표시)
학습 모델 가중치는 시간 측정용(크기만 같음). 정확도는 재지 않는다.
실행: CUDA_VISIBLE_DEVICES=0 python n08_speed/speed_scan.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # torch import 전에 GPU 0번만
import csv, glob, time, warnings
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "week09", "cache_real")
CHANS, BATCHES, N_PCA = [16, 64, 256], [1, 100, 1000], 256
rng = np.random.default_rng(0)
dev = torch.device("cuda")
DT = np.float32

# ---------- 머리 ----------
files = sorted(glob.glob(os.path.join(CACHE, "OAS*.npz")))
if files:
    d = np.load(files[0]); G64 = d["G_model"].astype(DT); pos = d["pos"].astype(DT)
    pos = pos * (1000.0 if np.abs(pos).max() < 1 else 1.0)
    HEAD = os.path.basename(files[0])[:8]
else:
    N = 5124; G64 = rng.normal(size=(64, N, 3)).astype(DT)
    pos = rng.normal(size=(N, 3)).astype(DT) * 50; HEAD = "대체(무작위, 캐시 없음)"
N_SRC = G64.shape[1]
def lead(C):
    if C <= G64.shape[0]: return G64[np.sort(rng.choice(G64.shape[0], C, replace=False))], HEAD
    # 256채널: 64채널 행을 섞어 늘림 + 약간의 잡음 (행렬 크기만 맞추는 대체)
    G = G64[rng.integers(0, G64.shape[0], C)]
    return (G + 0.05 * np.abs(G).mean() * rng.normal(size=G.shape)).astype(DT), HEAD + " 기반 대체(256ch 없음)"

# ---------- 준비 계산 (몽타주마다 1번) ----------
def slor_np(G3, lam2=1.0 / 9.0):
    k = G3.shape[0]; R = np.eye(k) - 1.0 / k; L = R @ G3[:, :, 2]
    e, V = np.linalg.eigh(R @ R.T); keep = e > 1e-6 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L; Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    return ((Lw.T @ A @ W) / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]).astype(DT)
def scan_np(G3):
    A = np.transpose(G3, (1, 0, 2)); Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
    return np.ascontiguousarray(np.transpose(Q, (0, 2, 1)).reshape(-1, G3.shape[0])).astype(DT)
def slor_t(G3, lam2=1.0 / 9.0):
    k = G3.shape[0]; R = torch.eye(k, device=dev) - 1.0 / k; L = R @ G3[:, :, 2]
    e, V = torch.linalg.eigh(R @ R.T); keep = e > 1e-6 * e.max()
    W = (V[:, keep] / torch.sqrt(e[keep])).T
    Lw = W @ L; Lw = Lw / torch.sqrt(torch.trace(Lw @ Lw.T) / keep.sum())
    A = torch.linalg.inv(Lw @ Lw.T + lam2 * torch.eye(int(keep.sum()), device=dev))
    return (Lw.T @ A @ W) / torch.sqrt(torch.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]
def scan_t(G3):
    A = G3.permute(1, 0, 2); Q = torch.linalg.qr(A - A.mean(1, keepdim=True))[0]
    return Q.permute(0, 2, 1).reshape(-1, G3.shape[0]).contiguous()

# ---------- 학습 모델 (크기만 같은 가중치) ----------
mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=1).fit(rng.normal(size=(300, N_PCA)), rng.normal(size=(300, 3)))
Wt = [torch.tensor(w, dtype=torch.float32, device=dev) for w in mlp.coefs_]
bt = [torch.tensor(b, dtype=torch.float32, device=dev) for b in mlp.intercepts_]
B = np.linalg.qr(rng.normal(size=(N_SRC, N_PCA)))[0].T.astype(DT)  # 256 x 5124 (PCA 기저 크기)
tree = cKDTree(pos); pos_t = torch.tensor(pos, device=dev)

# ---------- 적용 (신호마다) ----------
def ref_np(Y): return Y - Y.mean(1, keepdims=True)
def run_cpu(meth, Y, op):
    if meth == "sLORETA": return np.abs(ref_np(Y) @ op.T).argmax(1)
    if meth == "쌍극자 스캔": return ((ref_np(Y) @ op.T) ** 2).reshape(len(Y), -1, 3).sum(2).argmax(1)
    F = ref_np(Y) @ op.T; F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-30)
    return tree.query(mlp.predict(F) * 100.0)[1]
def run_gpu(meth, Y, op):
    Yt = torch.from_numpy(Y).to(dev, non_blocking=False); Yt = Yt - Yt.mean(1, keepdim=True)
    if meth == "sLORETA": out = (Yt @ op.T).abs().argmax(1)
    elif meth == "쌍극자 스캔": out = ((Yt @ op.T) ** 2).reshape(len(Y), -1, 3).sum(2).argmax(1)
    else:
        h = Yt @ op.T; h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-30)
        for i, (w, b) in enumerate(zip(Wt, bt)):
            h = h @ w + b
            if i < len(Wt) - 1: h = torch.relu(h)
        out = torch.cdist(h * 100.0, pos_t).argmin(1)
    return out.cpu().numpy()

def timeit(fn, min_rep=20, min_sec=0.5):
    for _ in range(3): fn()
    ts = []; t_end = time.perf_counter() + min_sec
    while len(ts) < min_rep or time.perf_counter() < t_end:
        if dev.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
        if len(ts) >= 2000: break
    return float(np.median(ts))

METHODS = ["sLORETA", "쌍극자 스캔", "학습 모델"]
rows, setup = [], []
print(f"머리: {HEAD} | GPU: {torch.cuda.get_device_name(0)} | CPU 스레드(OPENBLAS): {os.environ.get('OPENBLAS_NUM_THREADS', '기본')}", flush=True)
for C in CHANS:
    G, hname = lead(C); Gt = torch.tensor(G, device=dev)
    # 준비 계산 시간
    t_s_cpu = timeit(lambda: slor_np(G), 3, 0.2); t_q_cpu = timeit(lambda: scan_np(G), 3, 0.2)
    t_s_gpu = timeit(lambda: slor_t(Gt), 3, 0.2); t_q_gpu = timeit(lambda: scan_t(Gt), 3, 0.2)
    K = slor_np(G); Q2 = scan_np(G); BK = (B @ K).astype(DT)
    t_bk_cpu = timeit(lambda: B @ K, 3, 0.2)
    ops_cpu = {"sLORETA": K, "쌍극자 스캔": Q2, "학습 모델": BK}
    ops_gpu = {m: torch.tensor(v, device=dev) for m, v in ops_cpu.items()}
    setup.append([C, hname, t_s_cpu, t_s_gpu, t_q_cpu, t_q_gpu, t_bk_cpu])
    for nb in BATCHES:
        idx = rng.integers(0, N_SRC, nb); w = rng.normal(size=(nb, 3)).astype(DT)
        Y = np.einsum("cnk,nk->nc", G[:, idx, :], w); Y = (Y + 0.3 * Y.std() * rng.normal(size=Y.shape)).astype(DT)
        for m in METHODS:
            # 같은 답인지 확인 (CPU vs GPU, float32 반올림 차이로 몇 개는 다를 수 있음)
            agree = float(np.mean(run_cpu(m, Y, ops_cpu[m]) == run_gpu(m, Y, ops_gpu[m])))
            tc = timeit(lambda: run_cpu(m, Y, ops_cpu[m])) / nb
            tg = timeit(lambda: run_gpu(m, Y, ops_gpu[m])) / nb
            rows.append([C, nb, m, hname, tc * 1e3, tg * 1e3, agree])
            print(f"C={C:3d} 배치={nb:4d} {m:7s}  CPU {tc*1e3:9.4f} ms  GPU {tg*1e3:9.4f} ms  (CPU/GPU 답 일치 {agree:.3f})", flush=True)
with open(os.path.join(HERE, f"speed_scan{os.environ.get('TAG', '')}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.writer(f); wr.writerow(["채널", "배치", "방법", "머리", "CPU_ms_per_signal", "GPU_ms_per_signal", "CPU_GPU_일치율"])
    for r in rows: wr.writerow(r[:4] + [f"{r[4]:.5f}", f"{r[5]:.5f}", f"{r[6]:.3f}"])
with open(os.path.join(HERE, f"speed_scan_setup{os.environ.get('TAG', '')}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.writer(f); wr.writerow(["채널", "머리", "sLORETA_CPU_ms", "sLORETA_GPU_ms", "스캔QR_CPU_ms", "스캔QR_GPU_ms", "학습BK_CPU_ms"])
    for r in setup: wr.writerow(r[:2] + [f"{x*1e3:.3f}" for x in r[2:]])
print("저장")
