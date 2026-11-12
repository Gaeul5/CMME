"""Week 6-1: 지연 시간 (11/10 미팅 메모: "실시간성 보장되어야 함", "어댑터 연산량이 크지 않나?")

질문: 어댑터를 넣으면 실시간이 깨지는가?
나누어 재는 것
  (1) 사람마다 한 번: BEM + forward  → 이미 잰 값(week05 README: 머리당 2.2~3.2분)을 그대로 씀. 여기서는 다시 재지 않음
  (2) 몽타주가 바뀔 때 한 번: 역연산자 K 만들기, 어댑터 행렬 P = B K 만들기
  (3) 신호마다: 실제로 실시간이 걸린 부분
      - sLORETA        : K y (7,498 x k) → 절댓값 최대 위치
      - MLP 센서만     : 평균 기준 → 크기 1 → MLP
      - MLP 마스크     : 64칸 + 마스크 64칸 → MLP
      - 어댑터         : P y (256 x k) → 크기 1 → MLP
    MLP는 week05와 같은 크기 (입력 → 256 → 128 → 3).

핵심: 어댑터의 "7,498곳 지도 → PCA 256" 두 단계는 몽타주가 정해지면 P = B K 하나(256 x k)로 합쳐진다.
그래서 신호마다 하는 계산은 256 x k 행렬 곱 하나 + 작은 MLP뿐이다.

정확도와 무관한 측정이라 MLP 가중치와 PCA 기저는 무작위 값으로 만든다 (크기만 같으면 계산량이 같음).
리드필드는 week05 sample 머리(head000)의 실제 값.

실행: python week06/latency.py
"""
import os
import platform
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
G = np.load(os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz"))["G_model"][:, :, 2]   # 64 x 7498 (피질 수직 방향)
N_SRC = G.shape[1]
N_PCA = 256
H1, H2 = 256, 128


def inverse_kernel(sel, lam2=1.0 / 9.0):
    """week05 adapter.py의 inverse_kernels와 같은 계산 (sLORETA만)."""
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


def mlp(n_in):
    s = lambda a, b: rng.normal(size=(a, b)) / np.sqrt(a)
    return [s(n_in, H1), np.zeros(H1), s(H1, H2), np.zeros(H2), s(H2, 3), np.zeros(3)]


def mlp_fwd(w, X):
    h = np.maximum(X @ w[0] + w[1], 0)
    h = np.maximum(h @ w[2] + w[3], 0)
    return h @ w[4] + w[5]


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def ref(Y):
    return Y - Y.mean(1, keepdims=True)


def timeit(f, reps, warm=20):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return np.median(ts), np.percentile(ts, 99)


B = np.linalg.qr(rng.normal(size=(N_SRC, N_PCA)))[0].T      # 256 x 7498, 정규직교 (모양만 같은 가짜 PCA 기저)
W_sensor, W_adapt = {}, mlp(N_PCA)
W_mask = mlp(128)
MONT = {"64채널": np.arange(64), "32채널": np.sort(rng.choice(64, 32, replace=False)), "16채널": np.sort(rng.choice(64, 16, replace=False))}

print(f"환경: {platform.processor() or platform.machine()}, CPU {os.cpu_count()}코어, numpy {np.__version__}, GPU 없음")
print(f"리드필드 {G.shape[0]} 채널 x {N_SRC} 곳\n")

rows = []
for name, sel in MONT.items():
    k = len(sel)
    t_k, _ = timeit(lambda: inverse_kernel(sel), 20, warm=2)
    K = inverse_kernel(sel)
    t_p, _ = timeit(lambda: B @ K, 20, warm=2)
    P = B @ K
    W_sensor[k] = mlp(k)
    y1 = rng.normal(size=(1, 64))[:, sel]
    yb = rng.normal(size=(1000, 64))[:, sel]          # 1 kHz 기록 1초 분량

    def m_slor(y):
        return np.abs(ref(y) @ K.T).argmax(1)

    def m_sensor(y):
        return mlp_fwd(W_sensor[k], unit(ref(y)))

    def m_mask(y):
        X = np.zeros((len(y), 128))
        X[:, sel] = unit(ref(y))
        X[:, 64 + sel] = 1.0
        return mlp_fwd(W_mask, X)

    def m_adapt(y):
        return mlp_fwd(W_adapt, unit(ref(y) @ P.T))

    def m_adapt_unfused(y):                         # 비교용: K와 B를 따로 곱하는 경우 (7,498곳 지도를 실제로 만듦)
        return mlp_fwd(W_adapt, unit(unit(ref(y) @ K.T) @ B.T))

    print(f"[{name}] 몽타주가 바뀔 때 한 번: 역연산자 K {1e3 * t_k:.1f} ms, 어댑터 행렬 P = B K {1e3 * t_p:.1f} ms")
    for mname, f in (("sLORETA", m_slor), ("MLP 센서만", m_sensor), ("MLP 마스크", m_mask), ("어댑터", m_adapt), ("어댑터(합치지 않음)", m_adapt_unfused)):
        a1, p1 = timeit(lambda: f(y1), 2000)
        ab, _ = timeit(lambda: f(yb), 50, warm=5)
        rows.append((name, mname, 1e3 * t_k, 1e3 * t_p, 1e6 * a1, 1e6 * p1, 1e3 * ab))
        print(f"   {mname:<14} 신호 1개 {1e6 * a1:7.1f} us (99% {1e6 * p1:7.1f} us) | 1초 분량(1,000개) 한 번에 {1e3 * ab:6.2f} ms")
    print()

out = os.path.join(HERE, "latency.csv")
with open(out, "w", encoding="utf-8-sig") as f:
    f.write("몽타주,방법,K만들기_ms,P만들기_ms,신호1개_중앙값_us,신호1개_99퍼센트_us,1초분량1000개_ms\n")
    for r in rows:
        f.write(",".join([r[0], r[1]] + [f"{x:.2f}" for x in r[2:]]) + "\n")
print("저장:", out)
