"""Week 5-1: 몽타주 어댑터 (11/4 미팅 메모 반영).

메모: "어떤 몽타주의 입력이 들어오든 대응할 수 있게끔 하는 것이 중요. pre-process 같은 어댑터 하나 만드는 것도 나쁘지 않음.
       채널 수에 따른 오차를 최대한 적게 → 핵심 과제."

질문: 몽타주가 바뀌어도 모델 1개로 쓰려면 입력을 어떤 모양으로 바꿔 주는 게 좋은가?
비교하는 입력 표현 (모델 구조는 모두 같은 MLP (256, 128)):
  1) 마스크 (Week 4 방식): 64칸에 값(없는 채널은 0) + 64칸 마스크
  2) MNE 어댑터: 몽타주마다 자기 leadfield로 MNE 역연산자를 만들어 7,498곳의 source 지도로 바꾼 뒤,
     모든 몽타주가 공유하는 PCA 기저(256차원)에 투영. 입력 크기가 채널 수와 무관해짐
  3) sLORETA 어댑터: 2)와 같되 sLORETA 정규화를 적용
  4) 참고: 몽타주마다 따로 학습한 MLP (상한선에 가까운 기준), 고전 방법 sLORETA
입력은 모두 샘플마다 크기를 1로 맞춤 (세기 정보 제거, 표현만 비교).

학습 방식 두 가지:
  C1 (엄격): 학습에는 무작위 채널 부분집합(3~64개)만 쓰고, 이름 있는 몽타주(64/32/16/이마4/이마2)는 한 번도 못 본 채로 테스트
  C2 (Week 4와 비슷): 학습 샘플의 절반은 이름 있는 몽타주, 절반은 무작위 부분집합
테스트는 정답 머리(전도도·전극 위치 오차 다름), 방향 0~30도 무작위, SNR 10 dB(64채널 기준).
무작위 부분집합 테스트는 학습에 안 쓴 새 부분집합을 채널 수 구간별로 평가.

한계: 머리(MRI)는 sample 1명. 어댑터가 '새 머리'에도 통하는지는 이 실험으로 알 수 없음 (다중 MRI는 다음 단계).

실행:
  python week05/adapter.py <MNE-sample-data 경로>
  QUICK=1 SEED=0 python week05/adapter.py <경로>
"""
import csv
import os
import sys
import time
import warnings
import numpy as np
import mne
from sklearn.neural_network import MLPRegressor

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)

SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(SEED)
N_TR = 6000 if QUICK else 30000       # 몽타주별 모델의 학습 개수. 범용 모델은 2배
N_TE = 500 if QUICK else 2000
N_RANDOM_POOL = 60 if QUICK else 200  # 학습에 쓰는 무작위 부분집합 개수
N_PCA = 256
SNR_DB = 10.0
SKULL_TRUE, SKULL_MODEL, JITTER_MM = 0.0100, 0.0060, 5.0
AMP_RANGE = (0.5, 1.5)
TRAIN_TILT, TEST_TILT = 60.0, 30.0
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

data_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else str(mne.datasets.sample.data_path())
HERE = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(HERE, "..", "week04", "cache")      # Week 4와 같은 BEM 계산 결과 재사용
os.makedirs(cache_dir, exist_ok=True)


def first_existing(*cands):
    for c in cands:
        p = os.path.join(data_path, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cands)


bem_surf_file = first_existing("subjects/sample/bem/sample-5120-5120-5120-bem.fif",
                               "subjects/sample/bem/sample-1280-1280-1280-bem.fif")
src_file = first_existing("subjects/sample/bem/sample-oct-6-src.fif")
raw_file = first_existing("MEG/sample/sample_audvis_raw.fif", "MEG/sample/sample_audvis_trunc_raw.fif")
trans_file = first_existing("MEG/sample/sample_audvis_raw-trans.fif", "MEG/sample/sample_audvis_trunc-trans.fif")

# ---------------------------------------------------------------------------
# 1. 머리와 forward (Week 4와 같은 설정)
# ---------------------------------------------------------------------------
std = mne.channels.make_standard_montage("biosemi64")
NAMES = std.ch_names
std_pos = std.get_positions()["ch_pos"]
P = np.array([std_pos[n] for n in NAMES])
info_raw = mne.io.read_info(raw_file)
eeg_idx = mne.pick_types(info_raw, meg=False, eeg=True, exclude="bads")
samp = np.array([info_raw["chs"][i]["loc"][:3] for i in eeg_idx])
c_s, c_p = samp.mean(0), P.mean(0)
scale = np.median(np.linalg.norm(samp - c_s, axis=1)) / np.median(np.linalg.norm(P - c_p, axis=1))
P_model = (P - c_p) * scale + c_s
P_true = P_model + rng.normal(scale=JITTER_MM / 1000, size=P.shape)


def make_info(positions):
    info = mne.create_info(NAMES, sfreq=1000.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(NAMES, positions)), coord_frame="head"))
    return info


def bem_solution(skull):
    fname = os.path.join(cache_dir, f"bem-sol-skull{skull:.4f}.fif")
    if os.path.exists(fname):
        return mne.read_bem_solution(fname)
    surfs = mne.read_bem_surfaces(bem_surf_file)
    for s in surfs:
        s["sigma"] = {mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_BRAIN: 0.3,
                      mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_SKULL: skull,
                      mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD: 0.3}[s["id"]]
    sol = mne.make_bem_solution(surfs)
    mne.write_bem_solution(fname, sol, overwrite=True)
    return sol


def leadfield(positions, skull):
    src = mne.read_source_spaces(src_file)
    fwd = mne.make_forward_solution(make_info(positions), trans=trans_file, src=src,
                                    bem=bem_solution(skull), meg=False, eeg=True, mindist=5.0)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)
    n_ch = fwd["sol"]["data"].shape[0]
    return fwd["sol"]["data"].reshape(n_ch, -1, 3), fwd["source_rr"] * 1000


t0 = time.time()
G_true, pos = leadfield(P_true, SKULL_TRUE)
G_model, pos_b = leadfield(P_model, SKULL_MODEL)
assert np.allclose(pos, pos_b)
N_SRC = pos.shape[0]
print(f"forward 2개 {time.time() - t0:.0f} s, source {N_SRC}개, seed {SEED}{' (QUICK)' if QUICK else ''}")


def simulate(G3, n, tilt_max, snr_db=None):
    """샘플마다 방향이 피질 수직에서 0~tilt_max도 무작위로 틀어진 쌍극자 1개. 64채널 전위와 정답 인덱스."""
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    amp = rng.uniform(*AMP_RANGE, n)[:, None]
    clean = amp * np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    ref = clean - clean.mean(1, keepdims=True)
    snr = SNR_DB if snr_db is None else snr_db
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (snr / 10))
    return clean + rng.normal(size=clean.shape) * sigma, idx


# ---------------------------------------------------------------------------
# 2. 몽타주와 입력 표현
# ---------------------------------------------------------------------------
NAMED = {
    "64": NAMES,
    "32": mne.channels.make_standard_montage("biosemi32").ch_names,
    "16": mne.channels.make_standard_montage("biosemi16").ch_names,
    "이마4(+Fpz)": ["Fp1", "Fp2", "AF7", "AF8", "Fpz"],
    "이마2(+Fpz)": ["Fp1", "Fp2", "Fpz"],
}
NAMED = {k: np.array([NAMES.index(c) for c in v]) for k, v in NAMED.items()}


def pick_ref(Y, sel):
    X = Y[:, sel]
    return X - X.mean(1, keepdims=True)


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def inverse_kernels(G3_model, sel, lam2=1.0 / 9.0):
    """MNE, sLORETA 역연산자 (Week 4와 같음). 평균 기준으로 바꾼 (k채널) 입력에 곱해 source 지도 (위치)를 얻음."""
    k = len(sel)
    R = np.eye(k) - 1.0 / k
    L = R @ G3_model[sel, :, 2]
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K_mne = Lw.T @ A @ W
    res = np.einsum("ij,ji->i", Lw.T @ A, Lw)
    return K_mne, K_mne / np.sqrt(res)[:, None]


def random_montage():
    return np.sort(rng.choice(64, rng.integers(3, 65), replace=False))


# 공유 PCA 기저: 여러 몽타주의 source 지도를 모아 한 번 계산 (학습 머리 기준)
def build_bases(n_mont=60 if QUICK else 150, per=40):
    maps_m, maps_s = [], []
    for j in range(n_mont):
        sel = random_montage() if j % 5 else list(NAMED.values())[(j // 5) % 5]
        Y, _ = simulate(G_model, per, TRAIN_TILT)
        K_m, K_s = inverse_kernels(G_model, sel)
        X = pick_ref(Y, sel)
        maps_m.append(unit(X @ K_m.T))
        maps_s.append(unit(X @ K_s.T))
    bases = []
    for maps in (maps_m, maps_s):
        M = np.vstack(maps)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        bases.append(Vt[:N_PCA])
    return bases


t0 = time.time()
B_mne, B_slor = build_bases()
print(f"PCA 기저 계산 {time.time() - t0:.0f} s")

_proj_cache = {}


def projectors(sel):
    """몽타주 하나의 어댑터: (256 x k) 행렬 두 개. 입력(k채널, 평균 기준) @ 이 행렬의 전치 = 256차원 표현."""
    key = tuple(sel)
    if key not in _proj_cache:
        K_m, K_s = inverse_kernels(G_model, sel)
        _proj_cache[key] = (B_mne @ K_m, B_slor @ K_s)
    return _proj_cache[key]


def feat_masked(Y, sel):
    n = Y.shape[0]
    X = np.zeros((n, 64))
    X[:, sel] = unit(pick_ref(Y, sel))
    M = np.zeros((n, 64))
    M[:, sel] = 1.0
    return np.hstack([X, M])


def feat_adapter(Y, sel, which):
    return unit(pick_ref(Y, sel) @ projectors(sel)[which].T)


def feat_own(Y, sel):
    return unit(pick_ref(Y, sel))


def build_inputs(Y, assign, pool):
    """샘플마다 배정된 몽타주(pool[assign[i]])로 세 가지 표현을 만듦."""
    n = Y.shape[0]
    Xm, Xa, Xs = np.zeros((n, 128)), np.zeros((n, N_PCA)), np.zeros((n, N_PCA))
    for j, sel in enumerate(pool):
        ii = np.where(assign == j)[0]
        if len(ii) == 0:
            continue
        Xm[ii] = feat_masked(Y[ii], sel)
        Xa[ii] = feat_adapter(Y[ii], sel, 0)
        Xs[ii] = feat_adapter(Y[ii], sel, 1)
    return Xm, Xa, Xs


def fit(X, idx):
    return MLPRegressor(**MLP_KW).fit(X, pos[idx] / 100.0)


def predict(m, X):
    return m.predict(X) * 100.0


rows = []


def report(exp, cond, method, p, idx):
    e = np.linalg.norm(p - pos[idx], axis=1)
    rows.append((exp, cond, method, e.mean(), np.median(e), 100 * np.mean(e <= 10)))
    print(f"  {method:<22}: 평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}  10mm 이내 {100 * np.mean(e <= 10):5.1f}%")


# ---------------------------------------------------------------------------
# 3. 몽타주별 모델 (기준선): 이름 있는 몽타주마다 따로 학습
# ---------------------------------------------------------------------------
print("\n=== 기준선: 몽타주별 MLP ===")
own = {}
for name, sel in NAMED.items():
    Y, idx = simulate(G_model, N_TR, TRAIN_TILT)
    own[name] = fit(feat_own(Y, sel), idx)

# ---------------------------------------------------------------------------
# 4. 범용 모델 세 가지 × 학습 방식 두 가지
# ---------------------------------------------------------------------------
random_pool = [random_montage() for _ in range(N_RANDOM_POOL)]
named_list = list(NAMED.values())
BUCKETS = [(3, 8), (9, 16), (17, 32), (33, 64)]
test_pools = {}
for lo, hi in BUCKETS:                               # 학습에 안 쓴 새 부분집합 (채널 수 구간별)
    ms = []
    while len(ms) < 20:
        s = np.sort(rng.choice(64, rng.integers(lo, hi + 1), replace=False))
        ms.append(s)
    test_pools[(lo, hi)] = ms

for regime, desc in [("C1", "학습: 무작위 부분집합만 (이름 있는 몽타주는 처음 봄)"),
                     ("C2", "학습: 이름 있는 몽타주 50% + 무작위 부분집합 50%")]:
    print(f"\n=== {regime}: {desc} ===")
    t0 = time.time()
    n = 2 * N_TR
    Y, idx = simulate(G_model, n, TRAIN_TILT)
    if regime == "C1":
        pool = random_pool
        assign = rng.integers(0, len(pool), n)
    else:
        pool = random_pool + named_list
        use_named = rng.random(n) < 0.5
        assign = np.where(use_named, len(random_pool) + rng.integers(0, len(named_list), n),
                          rng.integers(0, len(random_pool), n))
    Xm, Xa, Xs = build_inputs(Y, assign, pool)
    models = {"범용: 마스크": (fit(Xm, idx), lambda Yt, s: feat_masked(Yt, s)),
              "범용: MNE 어댑터": (fit(Xa, idx), lambda Yt, s: feat_adapter(Yt, s, 0)),
              "범용: sLORETA 어댑터": (fit(Xs, idx), lambda Yt, s: feat_adapter(Yt, s, 1))}
    print(f"  학습 {time.time() - t0:.0f} s")

    for name, sel in NAMED.items():
        print(f" [{regime} / {name}, {len(sel)}개 전극]")
        Yt, it = simulate(G_true, N_TE, TEST_TILT)
        K_m, K_s = inverse_kernels(G_model, sel)
        report(regime, name, "sLORETA", pos[np.abs(pick_ref(Yt, sel) @ K_s.T).argmax(1)], it)
        report(regime, name, "몽타주별 MLP", predict(own[name], feat_own(Yt, sel)), it)
        for mname, (m, f) in models.items():
            report(regime, name, mname, predict(m, f(Yt, sel)), it)

    for (lo, hi), ms in test_pools.items():
        cond = f"무작위 {lo}~{hi}채널"
        print(f" [{regime} / {cond}, 새 부분집합 20개]")
        per = max(N_TE // 20, 25)
        preds = {k: [] for k in ["sLORETA", *models]}
        idxs = []
        for sel in ms:
            Yt, it = simulate(G_true, per, TEST_TILT)
            idxs.append(it)
            K_m, K_s = inverse_kernels(G_model, sel)
            preds["sLORETA"].append(pos[np.abs(pick_ref(Yt, sel) @ K_s.T).argmax(1)])
            for mname, (m, f) in models.items():
                preds[mname].append(predict(m, f(Yt, sel)))
        it = np.concatenate(idxs)
        for k, v in preds.items():
            report(regime, cond, k, np.vstack(v), it)

center = pos.mean(0)
print(f"\n참고: 항상 중심을 답하면 평균 오차 약 {np.linalg.norm(pos - center, axis=1).mean():.1f} mm")

out = os.path.join(HERE, f"adapter_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["실험", "조건", "방법", "평균mm", "중앙값mm", "10mm이내%"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.1f}"])
print("결과 저장:", out)
