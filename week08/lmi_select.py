"""Week 8-4b: 머리 모델을 고르는 '정답 없는' 기준 비교 — 채널 교차 검증

8-4a의 기준(쌍극자로 설명되지 않고 남는 비율, 모든 채널로 맞추고 같은 채널로 채점)은 실패했다.
  → 같은 채널로 맞추고 채점하면 자유도가 큰 머리 모델이 유리해진다고 보고, 채점을 '맞출 때 안 쓴 채널'로 한다.
채널 교차 검증: 좋은 채널을 4묶음으로 나눔(머리 위에 고르게 섞이도록 무작위, 고정 시드).
  3묶음으로 쌍극자 스캔(위치 + 3방향 세기 최소제곱) → 남은 1묶음의 값을 예측 → 정규화한 예측 오차. 4번 돌려 평균, run 평균
단계: 1) 두개골 전도도 6개 중 기준이 가장 좋은 것  2) 그 전도도에서 전극 전체 이동 x,y,z 각 {−4, 0, 4} mm (27가지, 선형 근사)
     3) 고른 전극 이동으로 순방향 모델을 정확히 다시 계산
비교용으로 8-4a 기준(같은 채널로 채점)도 같은 후보에서 다시 계산
입력: cache_lmi/adapt_full_sub-XX.npz (lmi_headadapt.py가 만든 후보 리드필드)
출력: lmi_select_sub-XX.csv, cache_lmi/adapt_sub-XX.npz (고른 리드필드. 이전 파일을 덮어씀)

실행: python week08/lmi_select.py "<Localize MI 폴더>" sub-01
"""
import csv
import itertools
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmi_common as C  # noqa: E402
import lmi_fwd_lib as L  # noqa: E402
from lmi_routes_lib import signal  # noqa: E402

root, sub = sys.argv[1], sys.argv[2]
SHIFT = [-4.0, 0.0, 4.0]
t0 = time.time()
Z = np.load(os.path.join(HERE, "cache_lmi", f"adapt_full_{sub}.npz"))
CONDS = list(Z["conds"])
G_all, J = Z["G_all"], Z["J"]
names, E = C.scalp_xyz(root, sub)
import mne  # noqa: E402
fwd = mne.read_forward_solution(os.path.join(root, f"derivatives/sourcemodelling/{sub}/fwd/{sub}_fwd.fif"), verbose=False)
pos = np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]]) * 1000
bads = set(fwd["info"]["bads"])
runs = [r for r in C.runs(root) if r["sub"] == sub]
cx = C.contacts_xyz(root, sub)
U, GOODS, MIDS, FOLDS = [], [], [], []
frng = np.random.default_rng(0)
for r in runs:
    good = np.array([n not in (bads | C.bad_channels(root, sub, r["run"])) for n in names])
    U.append(signal(C.evoked(root, sub, r["run"]), good))
    GOODS.append(good)
    MIDS.append((cx[r["contacts"][0]] + cx[r["contacts"][1]]) / 2 * 1000)
    FOLDS.append(frng.permutation(good.sum()) % 4)


def ref(X):
    return X - X.mean(0, keepdims=True)


def scan(Gs, y):
    """Gs (k, V, 3) 평균 기준 전 리드필드, y (k,) → 가장 잘 맞는 위치, 그 위치의 3방향 세기, 남는 비율"""
    A = np.transpose(ref(Gs), (1, 0, 2))                                # (V, k, 3)
    Q, R = np.linalg.qr(A)
    yr = y - y.mean()
    c = np.einsum("vkj,k->vj", Q, yr)
    g = (c**2).sum(1) / (yr @ yr)
    v = int(g.argmax())
    m = np.linalg.solve(R[v], c[v])
    return v, m, 1 - g[v]


def evaluate(G):
    ins, cv, err = [], [], []
    for u, good, mid, fold in zip(U, GOODS, MIDS, FOLDS):
        Gg = G[good].astype(np.float64)
        v, m, res = scan(Gg, u)
        ins.append(res)
        err.append(np.linalg.norm(pos[v] - mid))
        e = []
        for f in range(4):
            tr, te = fold != f, fold == f
            vf, mf, _ = scan(Gg[tr], u[tr])
            pred = Gg[te][:, vf, :] @ mf
            # 평균 기준 맞추기: 학습 채널에서의 평균을 기준으로 둘 다 빼 준다
            yt = u[te] - u[tr].mean()
            pt = pred - (Gg[tr][:, vf, :] @ mf).mean()
            e.append(((yt - pt) ** 2).sum() / (yt**2).sum())
        cv.append(np.mean(e))
    return np.mean(ins), np.mean(cv), np.median(err), np.mean(err)


rows = []
for i, c in enumerate(CONDS):
    a, b, md, mn = evaluate(G_all[i])
    rows.append(["전도도", c, 0, 0, 0, a, b, md, mn])
    print(f"{sub} 전도도 {c}: 같은 채널 {a:.4f}  교차 검증 {b:.4f}  오차 중앙값 {md:.1f} ({time.time() - t0:.0f} s)", flush=True)
conds = [r for r in rows if r[0] == "전도도"]
c_cv = min(conds, key=lambda r: r[6])[1]
c_in = min(conds, key=lambda r: r[5])[1]
c_or = min(conds, key=lambda r: r[7])[1]
ic = CONDS.index(c_cv)
best = None
for t in itertools.product(SHIFT, SHIFT, SHIFT):
    G = G_all[ic].astype(np.float64) + sum(t[k] * J[k] for k in range(3))
    a, b, md, mn = evaluate(G)
    rows.append(["전극 이동(근사)", c_cv, *t, a, b, md, mn])
    if best is None or b < best[0]:
        best = (b, t)
t_star = np.array(best[1])
if np.any(t_star):
    S = L.setup(root, sub)
    G_final = L.forward(S, E + t_star / 1000, skull=c_cv)
else:
    G_final = G_all[ic].astype(np.float64)
a, b, md, mn = evaluate(G_final)
rows.append(["최종(정확히 다시 계산)", c_cv, *t_star, a, b, md, mn])
base = [r for r in conds if r[1] == 0.006][0]
print(f"{sub}: 교차 검증으로 고른 전도도 {c_cv} (같은 채널 기준 {c_in}, 정답 기준 {c_or}), 전극 이동 {t_star} mm | "
      f"쌍극자 스캔 오차 중앙값 기본 {base[7]:.1f} → 전도도 {[r for r in conds if r[1] == c_cv][0][7]:.1f} → 전도도·전극 {md:.1f} ({time.time() - t0:.0f} s)", flush=True)
np.savez_compressed(os.path.join(HERE, "cache_lmi", f"adapt_{sub}.npz"), G_final=G_final.astype(np.float32),
                    G_cond=G_all[ic].astype(np.float32), G_oracle=G_all[CONDS.index(c_or)].astype(np.float32),
                    c_star=c_cv, c_oracle=c_or, t_star=t_star)
with open(os.path.join(HERE, f"lmi_select_{sub}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["단계", "전도도", "x이동mm", "y이동mm", "z이동mm", "같은채널_남는비율", "교차검증_예측오차", "스캔오차중앙값mm", "스캔오차평균mm"])
    for r in rows:
        w.writerow(r[:5] + [f"{r[5]:.5f}", f"{r[6]:.5f}", f"{r[7]:.2f}", f"{r[8]:.2f}"])
