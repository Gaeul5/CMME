"""Week 9-2: Parmigiani 외 2022 실제 기록에서 자극 위치 찾기 (피험자 36명, 세션 323개)

데이터는 0.5~45 Hz로 거른 1 kHz라 0.5 ms 자극 펄스가 시간축으로 번져 있다. 시간을 앞뒤로 보는 필터라 펄스가 자극 '앞'으로도 번진다.
뇌 반응은 자극보다 먼저 나올 수 없으므로, 자극 전 구간의 신호는 순수한 자극 잡음(위치를 아는 전류원)으로 볼 수 있다.
구간 (자극 시점 0 기준, ms)
  자극 전 −5~−1   : 번진 자극 잡음만 (Localize-MI와 같은 성격의 정답)
  자극 직후 0~5    : 자극 잡음 + 아주 이른 반응
  자극 잡음 최대 5~12 : 번진 자극 잡음 최대 + 이른 반응
  초기 반응 15~40  : 뇌 반응(N1 근처). 정답은 '자극 위치 근처' 정도
각 구간에서 GFP가 가장 큰 시점 하나를 씀. 정답 = 자극한 두 접점의 중점 (SEEG 좌표 파일의 쌍극 접점 위치)
방법: 쌍극자 스캔, sLORETA(피질 법선) — 머리 모델은 템플릿 변형 머리(par_warp_heads.py)
기록: 자극 세기, 펄스 길이, 각도(평행/수직), 회백질/백질, Localize-MI와 겹치는 환자 여부

실행: python week09/par_eval.py <reduced 폴더>
"""
import csv
import glob
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
red = sys.argv[1]
LMI_SAME = {"sub-21": "sub-01", "sub-10": "sub-02", "sub-17": "sub-03", "sub-09": "sub-04", "sub-27": "sub-06", "sub-28": "sub-07"}
WINDOWS = {"자극 전 -5~-1": (-5, -1), "자극 직후 0~5": (0, 5), "자극 잡음 최대 5~12": (5, 12), "초기 반응 15~40": (15, 40)}


def tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ref(X):
    return X - X.mean(0, keepdims=True)


def slor(Gn, lam2=1.0 / 9.0):
    k = Gn.shape[0]
    R = np.eye(k) - 1.0 / k
    L = R @ Gn
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    return (Lw.T @ A @ W) / np.sqrt(np.einsum("ij,ji->i", Lw.T @ A, Lw))[:, None]


rows = []
for hf in sorted(glob.glob(os.path.join(HERE, "cache_par", "sub-*.npz"))):
    sub = os.path.basename(hf)[:-4]
    Z = np.load(hf)
    G, pos, hnames = Z["G"].astype(np.float64), Z["pos_t1w_mm"], list(Z["names"])
    A = np.transpose(G, (1, 0, 2))
    contacts = {r["name"]: np.array([float(r[k]) for k in "xyz"]) * 1000
                for r in tsv(glob.glob(os.path.join(red, sub, "*space-T1w_electrodes.tsv"))[0]) if r["x"] not in ("n/a", "")}
    for sf in sorted(glob.glob(os.path.join(red, sub, f"{sub}_run-*.npz"))):
        S = np.load(sf)
        desc = str(S["description"])
        m = re.match(r"Stimulation of (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)", desc)
        if not m or m.group(1) not in contacts:
            print(sub, "정답 접점 없음:", desc[:60]); continue
        pair, ma, dur, freq, angle, gw = m.groups()
        gt = contacts[pair]
        st = dict(zip(S["eeg_names"], S["eeg_status"]))
        idx = [i for i, n in enumerate(hnames) if st.get(n) == "good"]
        rowi = [list(S["eeg_names"]).index(hnames[i]) for i in idx]
        Y = ref(S["eeg_mean"][rowi].astype(np.float64))
        t = (np.arange(Y.shape[1]) / float(S["sfreq"]) - float(S["zero_time"])) * 1000
        base = Y[:, (t > -250) & (t < -50)]
        noise = np.sqrt((base**2).sum(0).mean())
        Gs = A[:, idx, :]
        Q = np.linalg.qr(Gs - Gs.mean(1, keepdims=True))[0]
        Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, len(idx))
        K = slor(G[idx][:, :, 2])
        floor = np.linalg.norm(pos - gt, axis=1).min()
        for wn, (a, b) in WINDOWS.items():
            w = (t >= a) & (t <= b)
            seg = Y[:, w]
            j = np.argmax((seg**2).sum(0))
            u = seg[:, j]
            snr = 20 * np.log10(np.linalg.norm(u) / noise)
            u = u / np.linalg.norm(u)
            g = ((Q2 @ u) ** 2).reshape(-1, 3).sum(1)
            v_scan = int(g.argmax())
            v_sl = int(np.abs(K @ u).argmax())
            for meth, v, ex in (("쌍극자 스캔", v_scan, g.max()), ("sLORETA", v_sl, np.nan)):
                rows.append([sub, os.path.basename(sf)[len(sub) + 5:-4], pair, ma, dur, angle, gw, LMI_SAME.get(sub, ""), wn, meth,
                             f"{np.linalg.norm(pos[v] - gt):.2f}", f"{floor:.2f}", f"{snr:.1f}", f"{t[w][j]:.0f}", f"{ex:.4f}", len(idx)])
    print(sub, "세션", len({r[1] for r in rows if r[0] == sub}), "| 자극 전 구간 스캔 오차 중앙값",
          np.median([float(r[10]) for r in rows if r[0] == sub and r[8] == "자극 전 -5~-1" and r[9] == "쌍극자 스캔"]).round(1), flush=True)
with open(os.path.join(HERE, "par_eval.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["피험자", "run", "접점", "세기", "펄스", "각도", "회백질백질", "LocalizeMI같은환자", "구간", "방법", "오차mm", "바닥mm", "SNR_dB", "시점ms", "설명비율", "채널수"])
    w.writerows(rows)
print("저장", len(rows))
