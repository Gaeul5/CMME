"""Week 8-5: 적응 업그레이드 — 머리 모델 불확실성을 '고르기' 대신 '평균'과 '학습 때 섞기'로 다룸

8-4 결과: 정답 없이 두개골 전도도를 하나 고르면 맞을 때도 있고 틀릴 때도 있다 (전극 이동은 오히려 나빠짐).
그래서 하나를 고르지 않는 두 방법을 더한다. 모델은 합친 방법(8-3)과 같다.
  ① 시험 때 평균: 전도도 후보 6개(0.003~0.015) 각각으로 쌍극자 적합도 지도를 만들고 평균해서 넣음 (미리 정한 주 방법, 가중치 없음)
     비교: 교차 검증 기준으로 고른 전도도 하나(8-4b), 교차 검증 값으로 가중한 평균
  ② 학습 때 섞기: 다른 피험자 6명의 리드필드를 전도도 6개로 모두 써서 학습 (도메인 무작위화). 시험은 기본(0.006)과 ①
쌍극자 스캔에도 같은 ①을 적용 (평균한 지도의 최댓값)
학습/평가: 피험자 하나씩 빼고 학습(OASIS 10명 + 다른 Localize-MI 6명), 남은 1명의 실제 기록으로 평가. 정답은 평가에만 씀
필요: cache_lmi/adapt_full_sub-XX.npz (전도도 6개 리드필드), cache_lmi/sub-XX.npz (기준 머리 대응), lmi_select_sub-XX.csv (교차 검증 값)

실행: SEED=0 python week08/lmi_adapt2.py "<Localize MI 폴더>"
"""
import csv
import glob
import os
import sys
import time
import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmi_common as C  # noqa: E402
from lmi_routes_lib import ref_, unit, slor_kernel, dirs_rng, signal  # noqa: E402

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(14000 + SEED)
root = sys.argv[1]
N_TR = 12000 if QUICK else 60000
N_PCA = 256
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=30 if QUICK else 80, early_stopping=True, random_state=SEED)
SUBS = [f"sub-0{s}" for s in range(1, 8)]
mne_fwd = __import__("mne")


def affine_fit(P, Q):
    X = np.c_[P, np.ones(len(P))]
    A = np.linalg.lstsq(X, Q, rcond=None)[0]                      # (4, 3)
    return A


def apply(A, P):
    return np.c_[P, np.ones(len(P))] @ A


def apply_inv(A, Q):
    M, t = A[:3], A[3]
    return (Q - t) @ np.linalg.inv(M)


def simulate(G3, n):
    idx = rng.integers(0, G3.shape[1], n)
    w = dirs_rng(rng, n)
    clean = (rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n))[:, None] * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    r = ref_(clean)
    snr = rng.uniform(10, 40, (n, 1))
    return clean + rng.normal(size=clean.shape) * np.sqrt(np.mean(r**2, 1, keepdims=True) / 10 ** (snr / 10)), idx


def scan_idx(u, G3):
    A = np.transpose(G3, (1, 0, 2))
    Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]
    return int(((np.einsum("vkj,k->vj", Q, u)) ** 2).sum(1).argmax())


FEAT = os.environ.get("FEAT", "gof")


def gof_maps(G3sel, Yr):
    """위치마다 3방향 리드필드로 신호를 설명하는 비율 (방향과 무관한 지도). Yr (n, k) 평균 기준"""
    A = np.transpose(G3sel, (1, 0, 2))
    Q = np.linalg.qr(A - A.mean(1, keepdims=True))[0]                       # (V, k, 3)
    Q2 = np.transpose(Q, (0, 2, 1)).reshape(-1, G3sel.shape[0])
    num = ((Yr @ Q2.T) ** 2).reshape(len(Yr), -1, 3).sum(2)
    return num / np.maximum((Yr**2).sum(1, keepdims=True), 1e-30)


z0 = np.load(os.path.join(HERE, "..", "week05", "cache_heads", "head000.npz"))
pos0 = z0["pos"]
tree0 = cKDTree(pos0)

# 머리 목록: (리드필드 (k,V,3), 법선 성분 (k,V), source 위치 mm, 기준 좌표로 가는 아핀, 채널 수, 최소 채널)
HEADS = {}
for f in sorted(glob.glob(os.path.join(HERE, "..", "week06", "cache_oasis", "OAS*.npz"))):
    z = np.load(f)
    G = z["G_model"].astype(np.float64)
    HEADS[os.path.basename(f)[:8]] = dict(G=G, Gn=G[:, :, 2], pos=z["pos"], A=affine_fit(z["pos"], pos0), nch=64, kmin=8)
for s in SUBS:
    fwd = mne_fwd.read_forward_solution(os.path.join(root, f"derivatives/sourcemodelling/{s}/fwd/{s}_fwd.fif"), verbose=False)
    G = fwd["sol"]["data"].reshape(len(fwd["info"]["ch_names"]), -1, 3)
    pos = np.vstack([x["rr"][x["vertno"]] for x in fwd["src"]]) * 1000
    nn = np.vstack([x["nn"][x["vertno"]] for x in fwd["src"]])
    w = np.load(os.path.join(HERE, "cache_lmi", f"{s}.npz"))
    A = affine_fit(w["pos_t1w_mm"], pos0)                          # 변형 머리 source(T1w mm) ↔ 기준 source 번호 대응으로 맞춤
    bads = set(fwd["info"]["bads"])
    HEADS[s] = dict(G=G, Gn=np.einsum("cvj,vj->cv", G, nn), pos=pos, A=A, nch=G.shape[0], kmin=32,
                    good=np.array([n not in bads for n in fwd["info"]["ch_names"]]), names=fwd["info"]["ch_names"])
import pandas as pd  # noqa: E402
for s_ in SUBS:
    Zf = np.load(os.path.join(HERE, "cache_lmi", f"adapt_full_{s_}.npz"))
    HEADS[s_]["G_all"] = Zf["G_all"]                                       # (6, k, V, 3) float32
    HEADS[s_]["conds"] = list(Zf["conds"])
    sel_csv = pd.read_csv(os.path.join(HERE, f"lmi_select_{s_}.csv"))
    cvv = sel_csv[sel_csv["단계"] == "전도도"].set_index("전도도")["교차검증_예측오차"]
    HEADS[s_]["cv"] = np.array([cvv.loc[c] for c in HEADS[s_]["conds"]])
    assert np.allclose(HEADS[s_]["G_all"][HEADS[s_]["conds"].index(0.006)], HEADS[s_]["G"], rtol=1e-3, atol=1e-6 * np.abs(HEADS[s_]["G"]).max())
for k, h in HEADS.items():
    near = cKDTree(apply(h["A"], h["pos"])).query(pos0)[1]          # 기준 7,498곳마다 가장 가까운 그 머리 source
    h["S"] = near
    h["tree_own"] = cKDTree(h["pos"])
    if k.startswith("sub"):
        h["affine_err_mm"] = float(np.median(np.linalg.norm(apply(h["A"], h["pos"])[cKDTree(apply(h["A"], h["pos"])).query(pos0)[1]] - pos0, axis=1)))
print("기준 7,498곳 ↔ 실제 source 대응 거리 중앙값 mm:", {k: round(h["affine_err_mm"], 1) for k, h in HEADS.items() if k.startswith("sub")}, flush=True)


def B_for(h, B):
    """기준 공간 기저 B (256, 7498) → 그 머리 source 공간 기저 (256, V_h): 같은 source로 모이는 기준 점의 기저를 더함"""
    Bh = np.zeros((B.shape[0], h["G"].shape[1]))
    np.add.at(Bh.T, h["S"], B.T)
    return Bh


t0 = time.time()
# 공유 PCA 기저: 기준 머리(sample)의 sLORETA 지도
G0 = z0["G_model"].astype(np.float64)
maps = []
for _ in range(40 if QUICK else 150):
    sel = np.sort(rng.choice(64, rng.integers(8, 65), replace=False))
    Y, _ = simulate(G0[sel], 40)
    if os.environ.get("FEAT", "gof") == "slor":
        maps.append(unit(ref_(Y) @ slor_kernel(G0[sel][:, :, 2]).T))
    else:
        maps.append(unit(gof_maps(G0[sel], ref_(Y)) ** 4))
B = np.linalg.svd(np.vstack(maps), full_matrices=False)[2][:N_PCA]
del maps
for h in HEADS.values():
    h["Bh"] = B_for(h, B)


def gof_ref(h, G, Y, sel):
    return gof_maps(G[sel].astype(np.float64), ref_(Y[:, sel]))[:, h["S"]]


def feats_maps(M):
    return unit(M ** 4) @ B.T


def feats(h, Y, sel):
    if FEAT == "slor":
        return unit(ref_(Y[:, sel]) @ (h["Bh"] @ slor_kernel(h["Gn"][sel])).T)
    M = gof_maps(h["G"][sel], ref_(Y[:, sel]))[:, h["S"]]                    # 기준 7,498곳으로 옮김
    return unit(M ** 4) @ B.T                                                # 봉우리를 강조 (GOF는 넓게 퍼져 있음)


RUNS = C.runs(root)
CX = {s: C.contacts_xyz(root, s) for s in SUBS}
rows = []
ONE = os.environ.get("ONE")
MIX = os.environ.get("MIX") == "1"
ONLY_LMI = os.environ.get("ONLY_LMI") == "1"
for test in ([ONE] if ONE else SUBS):
    X, T = [], []
    train = [k for k in HEADS if k != test and (k.startswith("sub") or not ONLY_LMI)]
    if MIX:                                                                  # 피험자 머리는 전도도마다 따로 한 번씩 넣어 6배로
        train = [k for k in train if not k.startswith("sub")] + [k for k in train if k.startswith("sub") for _ in range(6)]
    per = N_TR // len(train)
    seen = {}
    for k in train:
        h = HEADS[k]
        if MIX and k.startswith("sub"):
            seen[k] = seen.get(k, -1) + 1
            Gtr = h["G_all"][seen[k]].astype(np.float64)                     # 같은 피험자를 전도도 6개로 한 번씩
        else:
            Gtr = h["G"]
        Y, idx = simulate(Gtr, per)
        chans = np.where(h["good"])[0] if "good" in h else np.arange(h["nch"])
        for j in range(10):
            ii = np.arange(j, per, 10)
            sel = np.sort(rng.choice(chans, rng.integers(h["kmin"], len(chans) + 1), replace=False))
            X.append(feats(dict(h, G=Gtr), Y[ii], sel))
            T.append(apply(h["A"], h["pos"][idx[ii]]))
    m = MLPRegressor(**MLP_KW).fit(np.vstack(X), np.vstack(T) / 100.0)
    del X, T
    h = HEADS[test]
    for r in [x for x in RUNS if x["sub"] == test]:
        good = h["good"] & np.array([n not in C.bad_channels(root, test, r["run"]) for n in h["names"]])
        sel = np.where(good)[0]
        Yr = C.evoked(root, test, r["run"])
        u = signal(Yr, good)
        Yfull = np.zeros((1, h["nch"]))
        Yfull[0, sel] = u
        mid = (CX[test][r["contacts"][0]] + CX[test][r["contacts"][1]]) / 2 * 1000
        tag = "학습 때 섞기" if MIX else "기본 학습"
        Mc = np.stack([gof_ref(h, h["G_all"][i], Yfull, sel)[0] for i in range(len(h["conds"]))])   # (6, 7498)
        w_cv = np.exp(-(h["cv"] - h["cv"].min()) / 0.002); w_cv /= w_cv.sum()
        i_cv = int(np.argmin(h["cv"])); i0 = h["conds"].index(0.006)
        own = lambda M: h["tree_own"].query(apply_inv(h["A"], m.predict(feats_maps(M[None])) * 100.0))[1][0]
        to_own = lambda M: h["S"][int(np.argmax(M))]                         # 기준 점 → 그 사람 source (쌍극자 스캔용)
        ests = {f"합친 방법 ({tag}) · 기본 전도도": own(Mc[i0]),
                f"합친 방법 ({tag}) · 교차 검증으로 고른 전도도": own(Mc[i_cv]),
                f"합친 방법 ({tag}) · 전도도 평균": own(Mc.mean(0)),
                f"합친 방법 ({tag}) · 교차 검증 가중 평균": own((w_cv[:, None] * Mc).sum(0))}
        if not MIX:
            Mo = np.stack([gof_maps(h["G_all"][i][sel].astype(np.float64), ref_(Yfull[:, sel]))[0] for i in range(len(h["conds"]))])  # 자기 source 공간
            ests.update({"쌍극자 스캔 · 기본 전도도": int(Mo[i0].argmax()), "쌍극자 스캔 · 교차 검증으로 고른 전도도": int(Mo[i_cv].argmax()),
                         "쌍극자 스캔 · 전도도 평균": int(Mo.mean(0).argmax()),
                         "쌍극자 스캔 · 교차 검증 가중 평균": int((w_cv[:, None] * Mo).sum(0).argmax())})
        for k, v in ests.items():
            rows.append([SEED, test, r["run"], "-".join(r["contacts"]), r["mA"], k, f"{np.linalg.norm(h['pos'][v] - mid):.2f}"])
    rr = [x for x in rows if x[1] == test]
    print(f"{test}: " + "  ".join(f"{k} {np.median([float(x[6]) for x in rr if x[5] == k]):.1f}" for k in dict.fromkeys(x[5] for x in rr)) + f"  ({time.time() - t0:.0f} s)", flush=True)

out = os.path.join(HERE, f"lmi_adapt2_{'mix' if MIX else 'base'}_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "피험자", "run", "접점", "mA", "방법", "오차mm"])
    w.writerows(rows)
print("저장:", out)
