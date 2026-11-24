"""Week 9-5: 변형 머리 vs 실제 뇌 모양 머리 — 사람 사이 리드필드가 얼마나 다른가 (같은 OASIS-3 28명)

1) 같은 번호 source의 리드필드 공간(64x3, 평균 기준) 유사도: 두 머리 기저의 주각 cos^2 평균 (1이면 같음)
2) 남의 머리로 풀었을 때 오차: 머리 a에서 잡음 없는 쌍극자 신호를 만들고 머리 b의 리드필드로 쌍극자 스캔 → 머리 a 위치에서 오차
   (자기 머리로 풀면 0 mm. 클수록 사람마다 머리가 다르다는 뜻 = 학습에 머리를 늘릴 이유)
"""
import glob, os, csv
import numpy as np
rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
def load(d):
    fs = sorted(glob.glob(os.path.join(HERE, "..", d, "OAS*.npz")) if d.startswith("week06") else glob.glob(os.path.join(HERE, d, "OAS*.npz")))
    return [os.path.basename(f)[:8] for f in fs], [np.load(f) for f in fs]
sets = {"변형 머리": load("week06/cache_oasis"), "실제 뇌 모양": load("cache_real")}
NPAIR, NSRC = 80, 300
pairs = [tuple(rng.choice(28, 2, replace=False)) for _ in range(NPAIR)]
rows = []
for name, (ids, zs) in sets.items():
    G = [z["G_model"].astype(np.float64) for z in zs]; P = [z["pos"] for z in zs]
    valid = np.all([z["valid"] for z in zs], 0) if "valid" in zs[0].files else np.ones(G[0].shape[1], bool)
    cand = np.flatnonzero(valid)
    def basis(Gh, i):
        A = Gh[:, i, :] - Gh[:, i, :].mean(0); return np.linalg.qr(A)[0]
    for a, b in pairs:
        src = rng.choice(cand, NSRC, replace=False)
        sim = np.mean([np.sum((basis(G[a], i).T @ basis(G[b], i)) ** 2) / 3 for i in src])
        w = rng.normal(size=(NSRC, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True)
        Y = np.einsum("cnk,nk->nc", G[a][:, src, :], w); Y -= Y.mean(1, keepdims=True)
        A = np.transpose(G[b][:, cand, :], (1, 0, 2)); A = A - A.mean(1, keepdims=True)
        Q = np.linalg.qr(A)[0]; Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, 64)
        v = cand[((Y @ Q2.T) ** 2).reshape(NSRC, -1, 3).sum(2).argmax(1)]
        err = np.linalg.norm(P[a][v] - P[a][src], axis=1)
        rows.append([name, ids[a], ids[b], sim, np.median(err), np.mean(err)])
    r = [x for x in rows if x[0] == name]
    print(f"{name}: 유사도 {np.mean([x[3] for x in r]):.3f}, 남의 머리로 풀 때 오차 중앙값 {np.median([x[4] for x in r]):.1f} mm (평균 {np.mean([x[5] for x in r]):.1f})", flush=True)
with open(os.path.join(HERE, "head_diversity.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["머리 종류", "머리 a", "머리 b", "리드필드 유사도", "오차 중앙값mm", "오차 평균mm"])
    for r in rows: w.writerow(r[:3] + [f"{r[3]:.4f}", f"{r[4]:.2f}", f"{r[5]:.2f}"])
