"""Week 8-4: 정답 없는 머리 모델 적응 (실제 기록만 보고 두개골 전도도와 전극 위치를 그 사람에게 맞춤)

왜: 합친 방법은 가짜 신호에서 3.8 mm인데 실제 기록에서 14.2 mm다. 모델이 아니라 머리 모델이 실제와 어긋나서 생기는 차이로 보인다.
기준 (정답 위치 안 씀): 그 피험자의 모든 run에서 '쌍극자 하나로 설명되지 않고 남는 비율'(1 − GOF 최대값)의 평균.
  머리 모델이 맞을수록 자극 펄스(진짜 쌍극자에 가까움)를 더 잘 설명할 것이라는 가정.
단계
  1) 두개골 전도도: 0.003, 0.0045, 0.006(기본, Mikulan 외), 0.009, 0.012, 0.015 S/m 중 기준이 가장 좋은 것
  2) 전극 전체 이동: x, y, z 각 −6~6 mm (3 mm 간격, 125가지). ±3 mm로 만든 리드필드의 차분으로 선형 근사해 빠르게 고른 뒤, 고른 값으로 정확히 다시 계산
비교 (평가에만 정답 사용): 기본 머리, 전도도만 적응, 전도도 + 전극 적응, 그리고 '정답으로 고른 전도도'(상한)
저장: cache_lmi/adapt_sub-XX.npz (적응한 리드필드. GitHub에 올리지 않음), lmi_headadapt_sub-XX.csv (기준 값과 오차)

실행: python week08/lmi_headadapt.py "<Localize MI 폴더>" sub-01
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
CONDS = [0.003, 0.0045, 0.006, 0.009, 0.012, 0.015]
D = 3.0                                                          # 차분 간격 mm
GRID = [-6.0, -3.0, 0.0, 3.0, 6.0]
t0 = time.time()
S = L.setup(root, sub)
names, E = C.scalp_xyz(root, sub)
pos = np.vstack([s["rr"][s["vertno"]] for s in S["fwd"]["src"]]) * 1000
bads = set(S["fwd"]["info"]["bads"])
runs = [r for r in C.runs(root) if r["sub"] == sub]
cx = C.contacts_xyz(root, sub)
U, GOODS, MIDS = [], [], []
for r in runs:
    good = np.array([n not in (bads | C.bad_channels(root, sub, r["run"])) for n in names])
    U.append(signal(C.evoked(root, sub, r["run"]), good))
    GOODS.append(good)
    MIDS.append((cx[r["contacts"][0]] + cx[r["contacts"][1]]) / 2 * 1000)


def fit_all(G):
    """run마다 (1 − GOF 최대값), 쌍극자 스캔 위치 오차 mm"""
    res, err = [], []
    for u, good, mid in zip(U, GOODS, MIDS):
        A = np.transpose(G[good], (1, 0, 2))
        Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
        g = (np.einsum("vkj,k->vj", Q, u) ** 2).sum(1)
        v = int(g.argmax())
        res.append(1 - g[v])
        err.append(np.linalg.norm(pos[v] - mid))
    return np.array(res), np.array(err)


rows = []
Gc = {}
for c in CONDS:
    Gc[c] = L.forward(S, E, skull=c).astype(np.float32)
    res, err = fit_all(Gc[c])
    rows.append(["전도도", c, 0, 0, 0, res.mean(), np.median(err), err.mean()])
    print(f"{sub} 전도도 {c}: 남는 비율 {res.mean():.4f}, 오차 중앙값 {np.median(err):.1f} mm ({time.time() - t0:.0f} s)", flush=True)
c_star = min(CONDS, key=lambda c: [x for x in rows if x[1] == c][0][5])
c_oracle = min(CONDS, key=lambda c: [x for x in rows if x[1] == c][0][6])
J = []
for i in range(3):
    dE = np.zeros(3)
    dE[i] = D / 1000
    J.append(((L.forward(S, E + dE, skull=0.006) - L.forward(S, E - dE, skull=0.006)) / (2 * D)).astype(np.float32))   # mm당 변화
print(f"{sub} 차분 리드필드 완료 ({time.time() - t0:.0f} s)", flush=True)
best = None
for t in (itertools.product(GRID, GRID, GRID) if os.environ.get("SKIP_GRID") != "1" else [(0.0, 0.0, 0.0)]):
    G = Gc[c_star] + sum(t[i] * J[i] for i in range(3))
    res, err = fit_all(G)
    rows.append(["전극 이동(근사)", c_star, *t, res.mean(), np.median(err), err.mean()])
    if best is None or res.mean() < best[0]:
        best = (res.mean(), t)
t_star = np.array(best[1])
G_final = L.forward(S, E + t_star / 1000, skull=c_star) if np.any(t_star) else Gc[c_star].astype(np.float64)
res, err = fit_all(G_final)
rows.append(["최종(정확히 다시 계산)", c_star, *t_star, res.mean(), np.median(err), err.mean()])
res0, err0 = fit_all(Gc[0.006])
print(f"{sub}: 고른 전도도 {c_star} (정답으로 고르면 {c_oracle}), 전극 이동 {t_star} mm | 쌍극자 스캔 오차 중앙값 기본 {np.median(err0):.1f} → 적응 {np.median(err):.1f} mm | "
      f"남는 비율 {res0.mean():.4f} → {res.mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
np.savez_compressed(os.path.join(HERE, "cache_lmi", f"adapt_full_{sub}.npz"), conds=np.array(CONDS),
                    G_all=np.stack([Gc[c] for c in CONDS]).astype(np.float16) if False else np.stack([Gc[c] for c in CONDS]),
                    J=np.stack(J), D=D)
np.savez_compressed(os.path.join(HERE, "cache_lmi", f"adapt_{sub}.npz"), G_final=G_final.astype(np.float32),
                    G_cond=Gc[c_star].astype(np.float32), c_star=c_star, c_oracle=c_oracle, t_star=t_star,
                    G_oracle=Gc[c_oracle].astype(np.float32))
with open(os.path.join(HERE, f"lmi_headadapt_{sub}.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["단계", "전도도", "x이동mm", "y이동mm", "z이동mm", "남는비율평균", "스캔오차중앙값mm", "스캔오차평균mm"])
    for r in rows:
        w.writerow(r[:5] + [f"{r[5]:.5f}", f"{r[6]:.2f}", f"{r[7]:.2f}"])
