"""Week 8-3: 추천 조합 — 범용 어댑터 + 실제 머리 모델 (경로 1과 경로 2를 합친 것)

경로 1의 문제: 범용 모델은 좋았지만 '템플릿을 늘린 변형 머리'의 리드필드를 써서 머리 모델 오차가 들어갔다.
경로 2의 문제: 실제 머리 모델을 썼지만 피험자마다 모델을 따로 학습했다 (범용이 아님).
합친 방법
  1) 입력: 그 피험자의 '실제' 리드필드(데이터의 순방향 모델, 8,196곳)로 만든 sLORETA 지도
  2) 그 지도를 공통 좌표로 옮김: 피험자 머리 → 기준 머리 아핀 변환 A (12개 값. 8-2의 정합 결과에서 최소제곱으로 맞춤, 잔차 약 2 mm)
     기준 머리의 7,498곳마다 가장 가까운 피험자 source 값을 가져온 뒤 공유 PCA 256으로 줄임 → 모든 머리가 같은 입력 공간
     (계산은 '피험자별 PCA 기저 B_h = 기준 기저를 가장 가까운 source로 모은 것'으로 합쳐서 한 번에 함)
  3) 출력: 기준 머리 좌표 → A의 역변환 → 피험자 머리에서 가장 가까운 source
  4) 학습: OASIS 변형 머리 10명(64채널) + '다른' Localize-MI 피험자 6명의 실제 리드필드(256채널)로 만든 가짜 신호. 실제 기록은 안 씀
     평가: 남은 1명의 실제 기록 (피험자 하나씩 빼며 7번)
  sLORETA 지도의 방향: 피질 법선 성분 (8-2 경로 2는 z 성분을 썼다. 여기서 바로잡음)
비교: 같은 실제 머리 모델의 쌍극자 스캔, sLORETA (8-1, 8-2와 같은 신호와 오차 정의)

실행: SEED=0 python week08/lmi_hybrid.py "<Localize MI 폴더>"
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
rng = np.random.default_rng(13000 + SEED)
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


def feats(h, Y, sel):
    if FEAT == "slor":
        return unit(ref_(Y[:, sel]) @ (h["Bh"] @ slor_kernel(h["Gn"][sel])).T)
    M = gof_maps(h["G"][sel], ref_(Y[:, sel]))[:, h["S"]]                    # 기준 7,498곳으로 옮김
    return unit(M ** 4) @ B.T                                                # 봉우리를 강조 (GOF는 넓게 퍼져 있음)


RUNS = C.runs(root)
CX = {s: C.contacts_xyz(root, s) for s in SUBS}
rows = []
ONE = os.environ.get("ONE")
ONLY_LMI = os.environ.get("ONLY_LMI") == "1"
for test in ([ONE] if ONE else SUBS):
    X, T = [], []
    train = [k for k in HEADS if k != test and (k.startswith("sub") or not ONLY_LMI)]
    per = N_TR // len(train)
    for k in train:
        h = HEADS[k]
        Y, idx = simulate(h["G"], per)
        chans = np.where(h["good"])[0] if "good" in h else np.arange(h["nch"])
        for j in range(10):
            ii = np.arange(j, per, 10)
            sel = np.sort(rng.choice(chans, rng.integers(h["kmin"], len(chans) + 1), replace=False))
            X.append(feats(h, Y[ii], sel))
            T.append(apply(h["A"], h["pos"][idx[ii]]))
    m = MLPRegressor(**MLP_KW).fit(np.vstack(X), np.vstack(T) / 100.0)
    del X, T
    h = HEADS[test]
    if os.environ.get("DIAG") == "1":                 # 진단: 그 피험자의 실제 리드필드로 만든 가짜 신호에서의 오차
        Ys, ids = simulate(h["G"], 500)
        sel = np.where(h["good"])[0]
        Ys = Ys.copy(); Ys[:, ~h["good"]] = 0
        pr = m.predict(feats(h, Ys, sel)) * 100.0
        e_ref = np.linalg.norm(pr - apply(h["A"], h["pos"][ids]), axis=1)
        v = h["tree_own"].query(apply_inv(h["A"], pr))[1]
        e_own = np.linalg.norm(h["pos"][v] - h["pos"][ids], axis=1)
        tr_pred = m.predict(np.vstack([feats(HEADS[train[0]], simulate(HEADS[train[0]]["G"], 300)[0], np.arange(HEADS[train[0]]["nch"]))]))
        print(f"  [진단 {test}] 가짜 신호 오차: 기준 좌표 {np.median(e_ref):.1f} mm, 자기 머리 {np.median(e_own):.1f} mm", flush=True)
    for r in [x for x in RUNS if x["sub"] == test]:
        good = h["good"] & np.array([n not in C.bad_channels(root, test, r["run"]) for n in h["names"]])
        sel = np.where(good)[0]
        Yr = C.evoked(root, test, r["run"])
        u = signal(Yr, good)
        Yfull = np.zeros((1, h["nch"]))
        Yfull[0, sel] = u
        pr = m.predict(feats(h, Yfull, sel)) * 100.0
        v_hy = h["tree_own"].query(apply_inv(h["A"], pr))[1][0]
        Ks = slor_kernel(h["Gn"][sel])
        mid = (CX[test][r["contacts"][0]] + CX[test][r["contacts"][1]]) / 2 * 1000
        ests = {"합친 방법: 범용 어댑터 + 실제 머리 모델": v_hy, "쌍극자 스캔 (데이터 머리 모델)": scan_idx(u, h["G"][sel]),
                "sLORETA 법선 (데이터 머리 모델)": int(np.abs(Ks @ u).argmax())}
        for k, v in ests.items():
            rows.append([SEED, test, r["run"], "-".join(r["contacts"]), r["mA"], k, f"{np.linalg.norm(h['pos'][v] - mid):.2f}"])
    rr = [x for x in rows if x[1] == test]
    print(f"{test}: " + "  ".join(f"{k} {np.median([float(x[6]) for x in rr if x[5] == k]):.1f}" for k in dict.fromkeys(x[5] for x in rr)) + f"  ({time.time() - t0:.0f} s)", flush=True)

out = os.path.join(HERE, f"lmi_hybrid_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["시드", "피험자", "run", "접점", "mA", "방법", "오차mm"])
    w.writerows(rows)
print("저장:", out)
