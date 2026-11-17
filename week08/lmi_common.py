"""Localize-MI 공통 도구: 데이터 읽기, 정답 위치, 신호 창, 고전 방법

데이터: Mikulan 외 (2020) Localize-MI, CC BY-NC-SA 4.0. 원본과 가공 데이터는 GitHub에 올리지 않는다 (결과 숫자만 저장).
좌표: 순방향 모델의 머리 좌표계 = T1w 좌표계 (sub-01에서 변환 이동량으로 확인). 단위 m.
입력 신호: run마다 시행 평균(evoked)에서 '자극 펄스 자체'(±2 ms)를 쓴다.
  Mikulan 외 2020 (Scientific Data)도 자극 잡음을 위치를 정확히 아는 전류원으로 보고 −2~2 ms에서 역문제를 풀었다.
  자극 후 뇌 반응(2~10 ms)은 신호가 약해 쓰지 않는다 (8-1 첫 실행에서 오차 중앙값 70 mm로 확인).
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import mne

mne.set_log_level("ERROR")
FS = 8000
T0 = 0.25                         # 에포크 시작에서 자극까지 (s)
WIN = (-0.002, 0.002)
BASE = (-0.25, -0.01)


def runs(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "derivatives/epochs/sub-*/eeg/*_epochs.json"))):
        sub = re.search(r"(sub-\d+)", f).group(1)
        run = re.search(r"run-(\d+)", f).group(1)
        desc = json.load(open(f))["Description"]
        m = re.match(r"Stimulation of channel (.+?)(\d+)-(\d+)\s+([\d.]+)\s*mA", desc)
        name, a, b, ma = m.group(1), m.group(2), m.group(3), float(m.group(4))
        out.append(dict(sub=sub, run=run, contacts=(f"{name}{a}", f"{name}{b}"), mA=ma, desc=desc))
    return out


def contacts_xyz(root, sub):
    t = pd.read_csv(os.path.join(root, f"derivatives/epochs/{sub}/ieeg/{sub}_task-seegstim_space-T1w_electrodes.tsv"), sep="\t")
    return {r["name"]: np.array([r["x"], r["y"], r["z"]], float) for _, r in t.iterrows()}


def scalp_xyz(root, sub):
    t = pd.read_csv(os.path.join(root, f"derivatives/epochs/{sub}/eeg/{sub}_task-seegstim_electrodes.tsv"), sep="\t")
    return list(t["name"]), t[["x", "y", "z"]].values.astype(float)


def load_fwd(root, sub):
    fwd = mne.read_forward_solution(os.path.join(root, f"derivatives/sourcemodelling/{sub}/fwd/{sub}_fwd.fif"))
    G = fwd["sol"]["data"].reshape(len(fwd["info"]["ch_names"]), -1, 3)
    pos = np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]])
    return G, pos, fwd["info"]["ch_names"], set(fwd["info"]["bads"])


def bad_channels(root, sub, run):
    t = pd.read_csv(os.path.join(root, f"derivatives/epochs/{sub}/eeg/{sub}_task-seegstim_run-01_channels.tsv"), sep="\t")
    return set(t.loc[t["status"] != "good", "name"])


def evoked(root, sub, run):
    a = np.load(os.path.join(root, f"_reduced/{sub}_run-{run}_evoked.npy"))
    return a[0].astype(np.float64)                     # (256, 2081) 평균


def tidx(lo, hi, n=2081):
    t = np.arange(n) / FS - T0
    return (t >= lo) & (t <= hi)


def prepare(Y, G, good):
    """평균 기준 + 기준선으로 잡음 백색화. 반환: 창 데이터 (k, T), 백색화한 리드필드 (k, V, 3)"""
    R = np.eye(good.sum()) - 1.0 / good.sum()
    Yg = R @ Y[good]
    Gg = np.einsum("ck,kvj->cvj", R, G[good])
    B = Yg[:, tidx(*BASE)]
    C = np.cov(B) + 1e-3 * np.trace(np.cov(B)) / len(B) * np.eye(len(B))   # 평균 기준이라 순위 하나 부족 → 작게 정규화
    e, V = np.linalg.eigh(C)
    keep = e > 1e-8 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Yw = W @ Yg[:, tidx(*WIN)]
    Gw = np.einsum("ck,kvj->cvj", W, Gg)
    return Yw, Gw


def top_pattern(Yw):
    U, S, _ = np.linalg.svd(Yw, full_matrices=False)
    return U[:, 0], S[0] ** 2 / (S**2).sum()


def scan(u, Gw):
    Q = np.linalg.qr(np.transpose(Gw, (1, 0, 2)))[0]          # (V, k, 3)
    return (np.einsum("vkj,k->vj", Q, u) ** 2).sum(1)


def min_norm(u, Gw, snr=3.0):
    k = Gw.shape[0]
    L = Gw.reshape(k, -1)
    lam2 = np.trace(L @ L.T) / k / snr**2
    A = np.linalg.inv(L @ L.T + lam2 * np.eye(k))
    K = (L.T @ A).reshape(-1, 3, k)                             # (V, 3, k)
    J = K @ u                                                   # (V, 3)
    Rb = np.einsum("vjk,kvl->vjl", K, Gw)                       # sLORETA 해상도 블록 (V, 3, 3)
    sl = np.einsum("vj,vj->v", J, np.linalg.solve(Rb + 1e-12 * np.eye(3), J[:, :, None])[:, :, 0])
    return (J**2).sum(1), sl
