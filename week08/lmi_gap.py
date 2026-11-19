"""Week 8-6: 시뮬레이션(약 4 mm)과 실제 기록(약 14 mm) 사이 10 mm는 어디서 오나

run마다 실제 자극 위치에서 가짜 신호를 만들고, 원인을 하나씩 더해 가며 쌍극자 스캔 오차가 실제 오차에 얼마나 가까워지는지 본다.
  A. 피질 위 가장 가까운 source, 방향은 피질 법선, 흰 잡음 (지금까지의 시뮬레이션과 같은 종류)
  B. A + 방향을 '전극 축 방향'으로 (두 접점을 잇는 방향. 양극성 자극의 전류 방향)
  C. B + 위치를 피질 위가 아닌 '실제 접점 중점'으로 (BEM으로 그 점의 쌍극자를 직접 계산)
  D. C + 흰 잡음 대신 '실제 잡음' (그 run 평균 신호의 자극 전 구간에서 뽑은 시점. 공간 상관이 있음)
  E. D + 머리 모델 불일치 (신호는 두개골 0.009·0.0045로 만들고, 찾기는 기본 0.006으로)
  F. D + 전극마다 다른 위치 오차 (신호는 전극을 무작위로 5·10 mm 옮긴 머리로 만들고, 찾기는 원래 전극으로)
  각 조건에서 '진짜 쌍극자가 설명하는 비율'도 같이 잰다. 실제 기록의 비율(낮음)을 재현하는 조건이 원인 후보
  신호 크기: 실제 GFP 최대 시점의 크기에 맞춤. 흰 잡음의 크기는 실제 잡음과 같은 크기
같이 재는 것: 실제 신호를 '진짜 위치·진짜 방향 쌍극자 하나'가 얼마나 설명하는지 (설명 비율). 낮으면 신호가 쌍극자 하나가 아님
각 run에서 잡음 30번 뽑음. 오차 = 쌍극자 스캔 위치 → 접점 중점

실행: python week08/lmi_gap.py "<Localize MI 폴더>"
"""
import csv
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmi_common as C  # noqa: E402
import lmi_fwd_lib as L  # noqa: E402

root = sys.argv[1]
N_DRAW = 30
rng = np.random.default_rng(15000)
rows = []
t0 = time.time()


def refc(X):
    return X - X.mean(-1, keepdims=True)


for s in range(1, 8):
    sub = f"sub-0{s}"
    S = L.setup(root, sub)
    names, E = C.scalp_xyz(root, sub)
    G = S["fwd"]["sol"]["data"].reshape(256, -1, 3)
    pos = np.vstack([x["rr"][x["vertno"]] for x in S["fwd"]["src"]])
    nn = np.vstack([x["nn"][x["vertno"]] for x in S["fwd"]["src"]])
    bads = set(S["fwd"]["info"]["bads"])
    cx = C.contacts_xyz(root, sub)
    runs = [r for r in C.runs(root) if r["sub"] == sub]
    mids = np.array([(cx[r["contacts"][0]] + cx[r["contacts"][1]]) / 2 for r in runs])
    axes = np.array([cx[r["contacts"][1]] - cx[r["contacts"][0]] for r in runs])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    fields = {}
    for sk in (0.006, 0.009, 0.0045):
        bem = L.bem_solution(S, skull=sk)
        fields[sk] = L.dipole_fields(S, E, bem, mids, axes)          # (run, 256)
        if sk == 0.006:
            for j in (5, 10):
                Ej = E + np.random.default_rng(16000 + s * 10 + j).normal(scale=j / 1000, size=E.shape)
                fields[f"jit{j}"] = L.dipole_fields(S, Ej, bem, mids, axes)
        del bem
    for i, r in enumerate(runs):
        good = np.array([n not in (bads | C.bad_channels(root, sub, r["run"])) for n in names])
        Y = C.evoked(root, sub, r["run"])[good]
        Yr = refc(Y.T).T
        w = C.tidx(*C.WIN)
        seg = Yr[:, w]
        y_pk = seg[:, np.argmax((seg**2).sum(0))]
        base = Yr[:, C.tidx(*C.BASE)]                                  # (k, T)
        noise_rms = np.sqrt((base**2).sum(0).mean())
        A = np.transpose(G[good], (1, 0, 2))
        Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
        Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, good.sum())

        def scan_err(X):
            g = ((X @ Q2.T) ** 2).reshape(len(X), -1, 3).sum(2)
            return np.linalg.norm(pos[g.argmax(1)] - mids[i], axis=1) * 1000

        v_near = np.linalg.norm(pos - mids[i], axis=1).argmin()
        f_norm = refc(G[good][:, v_near, :] @ nn[v_near])
        f_axis = refc(G[good][:, v_near, :] @ axes[i])
        f_true = {sk: refc(fields[sk][i][good]) for sk in fields}
        amp = np.linalg.norm(y_pk)
        ft = f_true[0.006] / np.linalg.norm(f_true[0.006])

        def make(f, noise):
            f = f / np.linalg.norm(f) * amp
            sgn = rng.choice([-1, 1], N_DRAW)[:, None]
            if noise == "white":
                n_ = rng.normal(size=(N_DRAW, good.sum()))
                n_ = refc(n_) / np.sqrt((refc(n_) ** 2).sum(1, keepdims=True)) * noise_rms
            else:
                n_ = base[:, rng.integers(0, base.shape[1], N_DRAW)].T
            return sgn * f + n_

        conds = {"A 피질 법선 + 흰 잡음": make(f_norm, "white"), "B 전극 축 방향 + 흰 잡음": make(f_axis, "white"),
                 "C 접점 중점 + 흰 잡음": make(f_true[0.006], "white"), "D 접점 중점 + 실제 잡음": make(f_true[0.006], "real"),
                 "E 머리 모델 불일치(두개골 0.009) + 실제 잡음": make(f_true[0.009], "real"),
                 "E 머리 모델 불일치(두개골 0.0045) + 실제 잡음": make(f_true[0.0045], "real"),
                 "F 전극 위치 오차 5 mm + 실제 잡음": make(f_true["jit5"], "real"),
                 "F 전극 위치 오차 10 mm + 실제 잡음": make(f_true["jit10"], "real")}
        real = scan_err(y_pk[None])[0]
        u = y_pk / amp
        expl_true = float((u @ ft) ** 2)                               # 진짜 위치·방향 쌍극자 하나가 설명하는 비율
        g_all = ((u[None] @ Q2.T) ** 2).reshape(-1, 3).sum(1)
        expl_best = float(g_all.max())                                 # 피질 위 아무 곳·아무 방향 쌍극자가 설명하는 최대 비율
        snr = 20 * np.log10(amp / noise_rms)
        for k, X in conds.items():
            e = scan_err(X)
            Xu = refc(X) / np.linalg.norm(refc(X), axis=1, keepdims=True)
            ex = float(np.median((Xu @ ft) ** 2))
            rows.append([sub, r["run"], r["mA"], k, f"{np.median(e):.2f}", f"{e.mean():.2f}", f"{snr:.1f}", f"{ex:.4f}"])
        rows.append([sub, r["run"], r["mA"], "실제 기록", f"{real:.2f}", f"{real:.2f}", f"{snr:.1f}", f"{expl_true:.4f}"])
        rows.append([sub, r["run"], r["mA"], "설명 비율: 가장 잘 맞는 피질 쌍극자 (실제 기록)", f"{expl_best:.4f}", f"{expl_best:.4f}", f"{snr:.1f}", ""])
    rr = [x for x in rows if x[0] == sub]
    print(f"{sub}: " + "  ".join(f"{k[:2]}{np.median([float(x[4]) for x in rr if x[3] == k]):.3g}" for k in dict.fromkeys(x[3] for x in rr)) + f" ({time.time() - t0:.0f} s)", flush=True)

with open(os.path.join(HERE, "lmi_gap.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["피험자", "run", "mA", "조건", "값(오차중앙값mm 또는 비율)", "평균", "SNR_dB", "진짜쌍극자_설명비율"])
    w.writerows(rows)
print("저장")
