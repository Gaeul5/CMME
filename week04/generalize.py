"""Week 4: 몽타주·방향·머리모델 불일치에 대한 일반화 실험.

10/27 미팅 메모를 반영한 Week 3 업그레이드.
- inverse crime 제거: 데이터를 만드는 forward(정답 머리)와 추정·학습에 쓰는 forward(모델 머리)를 분리
  * 정답 머리: 두개골 전도도 SKULL_TRUE, 전극 위치에 JITTER_MM 오차
  * 모델 머리: 두개골 전도도 SKULL_MODEL(MNE 기본값), 표준 전극 위치
  * 대조군으로 "같은 forward" 조건을 한 줄 남겨 inverse crime이 숫자를 얼마나 부풀리는지 보여 줌
- 고전 방법은 MNE, sLORETA만 비교 (dipole scan은 메모대로 제외)
- 실험 A (방향): 실제 방향이 피질 수직에서 0~90도 틀어질 때. 수직 데이터로만 학습한 MLP vs 각도 증강(0~60도)으로 학습한 MLP
- 실험 B (몽타주): Biosemi 64/32/16, 이마 4채널, 이마 2채널. 방법 비교
  * 몽타주별 MLP (몽타주마다 따로 학습)
  * 보간 + 64채널 MLP (spherical spline으로 64채널을 채운 뒤 64채널 모델에 넣음)
  * 범용 MLP 1개 (무작위 채널 부분집합으로 학습, 입력 = 0으로 채운 64채널 값 + 채널 마스크)
- SNR은 64채널 기준으로 정의하고, 채널마다 같은 잡음을 넣은 뒤 몽타주를 고름 (채널 수 효과와 SNR 효과 분리)

실행:
  python week04/generalize.py                 # MNE sample 데이터 기본 위치 사용
  python week04/generalize.py <MNE-sample-data 경로>
  QUICK=1 python week04/generalize.py          # 데이터 수를 줄인 빠른 확인용
  SEED=1 python week04/generalize.py           # 시드 변경
"""
import csv
import os
import sys
import warnings
import time
import numpy as np
from numpy.polynomial import legendre
import mne
from sklearn.neural_network import MLPRegressor

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=UserWarning)          # 짧은 학습에서 나오는 수렴 경고
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng = np.random.default_rng(SEED)

N_TR = 6000 if QUICK else 30000      # 학습 개수 (몽타주별 모델 기준)
N_TE = 500 if QUICK else 2000        # 테스트 개수
SNR_DB = 10.0                        # 64채널 기준 SNR
SKULL_TRUE = 0.0100                  # 정답 머리의 두개골 전도도 (S/m). 문헌 범위 약 0.004~0.02
SKULL_MODEL = 0.0060                 # 모델 머리 (MNE 기본값)
JITTER_MM = 5.0                      # 전극 위치 오차 (표준편차, mm)
AMP_RANGE = (0.5, 1.5)               # source 세기 범위 (세미나 슬라이드와 같은 범위)
MLP_KW = dict(hidden_layer_sizes=(256, 128), max_iter=80, early_stopping=True, random_state=SEED)

# ---------------------------------------------------------------------------
# 1. 파일 위치
# ---------------------------------------------------------------------------
data_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else str(mne.datasets.sample.data_path())


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
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(cache_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Biosemi 64 위치를 sample 머리 크기에 맞춤 (Week 3 실험 3과 같은 방식: 크기만 맞추고 MNE가 두피에 투영)
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
P_model = (P - c_p) * scale + c_s                                    # 모델이 알고 있는 위치
P_true = P_model + rng.normal(scale=JITTER_MM / 1000, size=P.shape)  # 실제로 붙은 위치 (오차 포함)
print(f"Biosemi 64 크기 보정 {scale:.2f}배, 전극 위치 오차 {JITTER_MM:.0f} mm")


def make_info(positions):
    info = mne.create_info(NAMES, sfreq=1000.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(NAMES, positions)), coord_frame="head"))
    return info


# ---------------------------------------------------------------------------
# 3. forward 두 개: 정답 머리(A)와 모델 머리(B). source space는 공유 → 정답 인덱스가 같음
# ---------------------------------------------------------------------------
def bem_solution(skull):
    fname = os.path.join(cache_dir, f"bem-sol-skull{skull:.4f}.fif")
    if os.path.exists(fname):
        return mne.read_bem_solution(fname)
    surfs = mne.read_bem_surfaces(bem_surf_file)
    for s in surfs:  # 뇌 0.3 / 두개골 skull / 두피 0.3 (안쪽에서 바깥쪽 순서와 무관하게 id로 지정)
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
    return fwd["sol"]["data"].reshape(n_ch, -1, 3), fwd["source_rr"] * 1000  # (64, 위치, [접선1, 접선2, 수직]), mm


t0 = time.time()
G_true, pos = leadfield(P_true, SKULL_TRUE)
G_model, pos_b = leadfield(P_model, SKULL_MODEL)
assert np.allclose(pos, pos_b), "두 forward의 source 위치가 달라짐"
N_SRC = pos.shape[0]
print(f"forward 2개 계산 {time.time() - t0:.0f} s, 피질 source {N_SRC}개")

# ---------------------------------------------------------------------------
# 4. 시뮬레이션: 64채널로 만들고, 몽타주는 나중에 고름
# ---------------------------------------------------------------------------
def simulate(G3, n, tilt_deg=0.0, tilt_max=None, snr_db=None):
    """tilt_deg: 고정 각도. tilt_max가 있으면 샘플마다 0~tilt_max 사이 무작위 각도 (증강용)."""
    idx = rng.integers(0, N_SRC, n)
    th = np.deg2rad(rng.uniform(0, tilt_max, n) if tilt_max is not None else np.full(n, tilt_deg))
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    amp = rng.uniform(*AMP_RANGE, n)[:, None]
    clean = amp * np.einsum("cnk,nk->nc", G3[:, idx, :], w)            # (n, 64), 기준 없는 전위
    ref = clean - clean.mean(1, keepdims=True)
    snr = SNR_DB if snr_db is None else snr_db
    sigma = np.sqrt(np.mean(ref**2, axis=1, keepdims=True) / 10 ** (snr / 10))  # 64채널 기준 SNR (inf = 잡음 없음)
    y = clean + rng.normal(size=clean.shape) * sigma                     # 채널마다 같은 크기의 잡음
    return y, idx


# ---------------------------------------------------------------------------
# 5. 몽타주. 모든 몽타주는 자기 채널들의 평균 기준으로 바꿔 씀.
#    이마 몽타주는 기준 전극 Fpz를 포함함 (Fpz 기준으로 잰 2채널 = {Fp1, Fp2, Fpz} 평균 기준과 정보량이 같음)
# ---------------------------------------------------------------------------
MONTAGES = {
    "64": NAMES,
    "32": mne.channels.make_standard_montage("biosemi32").ch_names,
    "16": mne.channels.make_standard_montage("biosemi16").ch_names,
    "이마4(+Fpz)": ["Fp1", "Fp2", "AF7", "AF8", "Fpz"],
    "이마2(+Fpz)": ["Fp1", "Fp2", "Fpz"],
}
MIDX = {k: np.array([NAMES.index(c) for c in v]) for k, v in MONTAGES.items()}


def pick_ref(Y, sel):
    X = Y[:, sel]
    return X - X.mean(1, keepdims=True)


# 학습 입력 크기 맞춤용 상수 (모델 머리 기준, 한 번만 계산)
_Ycal, _ = simulate(G_model, 2000)
SCALE = np.median(np.linalg.norm(pick_ref(_Ycal, MIDX["64"]), axis=1))


def feat(Y, sel):
    return pick_ref(Y, sel) / SCALE


def feat_masked(Y, sel):
    """범용 모델 입력: 64칸에 값(없는 채널은 0) + 64칸 마스크."""
    n = Y.shape[0]
    X = np.zeros((n, 64))
    X[:, sel] = pick_ref(Y, sel) / SCALE
    M = np.zeros((n, 64))
    M[:, sel] = 1.0
    return np.hstack([X, M])


# ---------------------------------------------------------------------------
# 6. 고전 방법: MNE / sLORETA (numpy 직접 구현, 피질 수직 방향 가정, 모델 머리 leadfield 사용)
#    잡음 공분산 = 단위행렬(기준 변경 후 R R^T), lambda^2 = 1/9, depth weighting 없음 (Week 3와 같음)
# ---------------------------------------------------------------------------
def inverse_kernels(G3_model, sel, lam2=1.0 / 9.0):
    k = len(sel)
    R = np.eye(k) - 1.0 / k                                  # 평균 기준 행렬
    L = R @ G3_model[sel, :, 2]                              # (k, 위치), 수직 방향
    e, V = np.linalg.eigh(R @ R.T)
    keep = e > 1e-10 * e.max()
    W = (V[:, keep] / np.sqrt(e[keep])).T                    # 백색화 (rank k-1)
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / keep.sum())          # MNE와 같은 크기 정규화
    A = np.linalg.solve(Lw @ Lw.T + lam2 * np.eye(keep.sum()), np.eye(keep.sum()))
    K_mne = Lw.T @ A @ W                                     # (위치, k)
    res = np.einsum("ij,ji->i", Lw.T @ A, Lw)                # 해상도 행렬 대각 (sLORETA 정규화)
    K_slor = K_mne / np.sqrt(res)[:, None]
    return K_mne, K_slor


def classical(Y, sel, kernels):
    X = pick_ref(Y, sel)
    return [pos[np.abs(X @ K.T).argmax(1)] for K in kernels]


# ---------------------------------------------------------------------------
# 7. 보간: spherical spline (Perrin 1989, MNE와 같은 식)
# ---------------------------------------------------------------------------
_, sph_o, _ = mne.bem.fit_sphere_to_headshape(make_info(P_model), dig_kinds=("eeg",), units="m")


def _g(cosang, m=4, n_terms=7):
    c = [0.0] + [(2 * n + 1) / (n**m * (n + 1) ** m * 4 * np.pi) for n in range(1, n_terms + 1)]
    return legendre.legval(cosang, c)


def spline_matrix(sel, alpha=1e-5):
    u = P_model - sph_o
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    uf, k = u[sel], len(sel)
    C = np.zeros((k + 1, k + 1))
    C[:k, :k] = _g(np.clip(uf @ uf.T, -1, 1)) + alpha * np.eye(k)
    C[:k, k] = C[k, :k] = 1.0
    Ci = np.linalg.pinv(C)
    G_tf = np.hstack([_g(np.clip(u @ uf.T, -1, 1)), np.ones((64, 1))])
    return G_tf @ Ci[:, :k]                                  # (64, k)


def interp_to64(Y, sel):
    X64 = pick_ref(Y, sel) @ spline_matrix(sel).T
    return (X64 - X64.mean(1, keepdims=True)) / SCALE


# ---------------------------------------------------------------------------
# 8. 평가 도구
# ---------------------------------------------------------------------------
rows = []


def report(exp, cond, method, pred, idx):
    e = np.linalg.norm(pred - pos[idx], axis=1)
    rows.append((exp, cond, method, e.mean(), np.median(e), 100 * np.mean(e <= 10)))
    print(f"  {method:<26}: 평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}  10mm 이내 {100 * np.mean(e <= 10):5.1f}%")


def fit_mlp(X, idx):
    return MLPRegressor(**MLP_KW).fit(X, pos[idx] / 100.0)


def pred(m, X):
    return m.predict(X) * 100.0


# ---------------------------------------------------------------------------
# 실험 0. inverse crime 대조: 같은 forward vs 다른 forward (64채널, 피질 수직)
# ---------------------------------------------------------------------------
print("\n=== 실험 0: inverse crime 대조 (64채널, 피질 수직, SNR 10 dB) ===")
sel64 = MIDX["64"]
K64 = inverse_kernels(G_model, sel64)
for cond, Gsim in [("같은 forward (inverse crime)", G_model), ("다른 forward (정답 머리)", G_true)]:
    print(f" [{cond}]")
    Yte, ite = simulate(Gsim, N_TE)
    m_, s_ = classical(Yte, sel64, K64)
    report("0", cond, "MNE", m_, ite)
    report("0", cond, "sLORETA", s_, ite)
print(" [검산: 같은 forward, 잡음 없음 → sLORETA는 이론상 0 mm]")
Yte, ite = simulate(G_model, N_TE, snr_db=np.inf)
m_, s_ = classical(Yte, sel64, K64)
report("0", "검산: 같은 forward, 잡음 없음", "MNE", m_, ite)
report("0", "검산: 같은 forward, 잡음 없음", "sLORETA", s_, ite)

# ---------------------------------------------------------------------------
# 실험 A. 방향 틀어짐 각도 0~90도 (64채널, 다른 forward)
# ---------------------------------------------------------------------------
print("\n=== 실험 A: 방향 틀어짐 각도 (64채널, SNR 10 dB, 테스트는 정답 머리) ===")
Ytr, itr = simulate(G_model, N_TR, 0.0)
mlp_norm = fit_mlp(feat(Ytr, sel64), itr)
Ytr, itr = simulate(G_model, N_TR, tilt_max=60.0)
mlp_aug = fit_mlp(feat(Ytr, sel64), itr)
for tilt in [0, 15, 30, 45, 60, 90]:
    print(f" [{tilt}도]")
    Yte, ite = simulate(G_true, N_TE, float(tilt))
    m_, s_ = classical(Yte, sel64, K64)
    report("A", f"{tilt}도", "MNE", m_, ite)
    report("A", f"{tilt}도", "sLORETA", s_, ite)
    report("A", f"{tilt}도", "MLP (수직만 학습)", pred(mlp_norm, feat(Yte, sel64)), ite)
    report("A", f"{tilt}도", "MLP (0~60도 증강)", pred(mlp_aug, feat(Yte, sel64)), ite)

# ---------------------------------------------------------------------------
# 실험 B. 몽타주 일반화 (피질 수직 + 0~30도 무작위 틀어짐을 테스트에 섞음, 다른 forward)
# ---------------------------------------------------------------------------
print("\n=== 실험 B: 몽타주 (SNR 10 dB 64채널 기준, 테스트 방향 0~30도 무작위, 정답 머리) ===")
TEST_TILT = 30.0
TRAIN_TILT = 60.0

# 범용 모델: 샘플마다 무작위 채널 부분집합 (절반은 이름 있는 몽타주, 절반은 3~64개 무작위)
Ytr, itr = simulate(G_model, 2 * N_TR, tilt_max=TRAIN_TILT)
Xu = np.zeros((len(itr), 128))
names_list = list(MIDX.values())
for i in range(len(itr)):
    if rng.random() < 0.5:
        sel = names_list[rng.integers(len(names_list))]
    else:
        sel = np.sort(rng.choice(64, rng.integers(3, 65), replace=False))
    Xu[i] = feat_masked(Ytr[i:i + 1], sel)[0]
mlp_uni = fit_mlp(Xu, itr)
mlp64 = mlp_aug  # 보간용 64채널 모델 (실험 A의 증강 모델 재사용)

for name, sel in MIDX.items():
    print(f" [{name}, {len(sel)}개 전극]")
    t = time.time()
    Ytr, itr = simulate(G_model, N_TR, tilt_max=TRAIN_TILT)
    mlp_m = mlp64 if name == "64" else fit_mlp(feat(Ytr, sel), itr)
    Yte, ite = simulate(G_true, N_TE, tilt_max=TEST_TILT)
    m_, s_ = classical(Yte, sel, inverse_kernels(G_model, sel))
    report("B", name, "MNE", m_, ite)
    report("B", name, "sLORETA", s_, ite)
    report("B", name, "MLP (몽타주별 학습)", pred(mlp_m, feat(Yte, sel)), ite)
    if name != "64":
        report("B", name, "보간→64채널 MLP", pred(mlp64, interp_to64(Yte, sel)), ite)
    report("B", name, "범용 MLP (마스크)", pred(mlp_uni, feat_masked(Yte, sel)), ite)
    print(f"  ({time.time() - t:.0f} s)")

center = pos.mean(0)
print(f"\n참고: 항상 중심을 답하면 평균 오차 약 {np.linalg.norm(pos - center, axis=1).mean():.1f} mm")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["실험", "조건", "방법", "평균mm", "중앙값mm", "10mm이내%"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.2f}", f"{r[5]:.1f}"])
print("결과 저장:", out)
