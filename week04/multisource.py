"""Week 4-2: 멀티소스 + 시계열 실험.

generalize.py(단일 쌍극자, 시간 없음)를 메모와 세미나 내용대로 한 단계 현실적으로 만든 버전.

Week 4 generalize.py 와 같은 점
- 정답 머리(전도도 0.010, 전극 위치 오차 5 mm)와 모델 머리(0.006, 표준 위치)를 분리 (inverse crime 제거)
- 학습 데이터와 고전 방법은 모델 머리만 사용, 테스트는 정답 머리로 만듦

이번에 새로 넣은 것
1. 시계열: source 하나가 시간에 따라 변함. s(t) = 사건 반응(감쇠 진동 덩어리) + Jansen-Rit 배경 리듬 (세미나 16번 슬라이드와 같은 구성).
   Jansen-Rit는 입력 p가 150 이상이면 10~11 Hz 알파 리듬이 나오고(검증함), 시간을 0.6~1.4배로 늘여 6~15 Hz로 흩어 놓음
2. 공간 퍼짐: 점이 아니라 피질 표면을 따라 퍼지는 가우시안 패치 (측지 거리, 세미나 15번 슬라이드). 패치 크기 sigma 3~8 mm
3. 멀티소스: source 1~3개가 동시에 켜짐. 서로 20 mm 이상 떨어짐
4. 확률 분포 출력: 좌표 1개 대신 8,196 후보 위치 위의 확률 분포를 출력 (세미나 방식). 정답 개수 K는 알려 주고, 분포에서 봉우리 K개를 뽑음
5. 평가: 정답 K개와 찾은 봉우리 K개를 짝지어(헝가리안) 오차를 잼. 정답 근처(10 mm) 확률 질량도 같이 봄
6. 시계열이 도움이 되는지: 시계열 특징(공간 SVD) 모델 vs 한 순간만 보는 모델 비교

실험
  E0 검산  : 점 source 1개, 잡음 없음, 같은 forward → sLORETA는 0 mm여야 함 (파이프라인 검산)
  E1       : source 개수 1 / 2 / 3
  E2       : source 2개의 간격 (가까울수록 어려움)
  E3       : 두 source의 파형이 얼마나 같이 움직이는가 (상관 0 / 0.5 / 0.9)
  E4       : 학습은 SNR 10 dB로만 했는데 테스트 SNR을 0 / 5 / 10 / 20 dB로 바꿈

실행:
  python week04/multisource.py <MNE-sample-data 경로>
  QUICK=1 python week04/multisource.py <경로>    # 빠른 확인
  SEED=1 python week04/multisource.py <경로>

신경망은 PyTorch 없이 numpy로 직접 구현함 (은닉층 2개, ReLU, softmax + 교차 엔트로피, Adam).
"""
import csv
import os
import sys
import time
import warnings
import numpy as np
import mne
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linear_sum_assignment

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", 0))
QUICK = os.environ.get("QUICK") == "1"
rng_geom = np.random.default_rng(SEED)        # 전극 위치 오차용 (generalize.py와 같은 방식)
rng = np.random.default_rng(SEED + 100)        # 데이터 생성용

# ---------------- 설정값 ----------------
FS = 250.0                      # 표본 주파수 (Hz)
T = 100                         # 창 길이 (표본 수, 0.4 초)
SNR_DB = 10.0                   # 학습 SNR (64채널·전체 시간 기준)
SKULL_TRUE, SKULL_MODEL = 0.0100, 0.0060
JITTER_MM = 5.0
AMP_RANGE = (0.5, 1.5)
SIGMA_RANGE = (3.0, 8.0)        # 패치 크기 (mm, 측지 거리 가우시안)
PATCH_LIMIT = 16.0              # 패치가 퍼질 수 있는 최대 측지 거리 (mm)
SEP_MIN = 20.0                  # source 사이 최소 간격 (mm)
NMS_R = 20.0                    # 봉우리를 뽑은 뒤 지우는 반경 (mm)
N_BANK = 400 if QUICK else 1500  # Jansen-Rit 파형 은행 크기
N_TR = 6000 if QUICK else 40000
N_TE = 150 if QUICK else 600
EPOCHS = 6 if QUICK else 25
HID = 384
RANK = 4                        # 시계열 특징: 공간 SVD 상위 성분 개수

# ---------------- 파일 위치 ----------------
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

# ---------------- 전극 위치 (generalize.py 와 같은 방식) ----------------
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
P_true = P_model + rng_geom.normal(scale=JITTER_MM / 1000, size=P.shape)


def make_info(positions):
    info = mne.create_info(NAMES, sfreq=1000.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(NAMES, positions)), coord_frame="head"))
    return info


def bem_solution(skull):
    fname = os.path.join(cache_dir, f"bem-sol-skull{skull:.4f}.fif")
    if os.path.exists(fname):
        return mne.read_bem_solution(fname)
    surfs = mne.read_bem_surfaces(bem_surf_file)
    ids = mne.io.constants.FIFF
    for s in surfs:
        s["sigma"] = {ids.FIFFV_BEM_SURF_ID_BRAIN: 0.3, ids.FIFFV_BEM_SURF_ID_SKULL: skull,
                      ids.FIFFV_BEM_SURF_ID_HEAD: 0.3}[s["id"]]
    sol = mne.make_bem_solution(surfs)
    mne.write_bem_solution(fname, sol, overwrite=True)
    return sol


SRC_FULL = mne.read_source_spaces(src_file)


def leadfield(positions, skull):
    fwd = mne.make_forward_solution(make_info(positions), trans=trans_file, src=SRC_FULL,
                                    bem=bem_solution(skull), meg=False, eeg=True, mindist=5.0)
    fwd_s = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)
    n_ch = fwd_s["sol"]["data"].shape[0]
    return fwd_s["sol"]["data"].reshape(n_ch, -1, 3), fwd_s["source_rr"] * 1000, fwd["src"]


t0 = time.time()
G_true, pos, fsrc = leadfield(P_true, SKULL_TRUE)
G_model, pos_b, _ = leadfield(P_model, SKULL_MODEL)
assert np.allclose(pos, pos_b)
N_SRC = pos.shape[0]
Gt_true = np.ascontiguousarray(G_true.transpose(1, 0, 2))     # (위치, 64, 3)
Gt_model = np.ascontiguousarray(G_model.transpose(1, 0, 2))
print(f"forward 2개 {time.time() - t0:.0f} s, 후보 위치 {N_SRC}개 (피질 8,196개 중 두개골에 너무 가까운 것 제외)")

# ---------------- 피질 표면 측지 거리 (패치용) ----------------
# 전체 8,196개 위치의 삼각망에서 최단 경로를 구하고, forward에 남은 위치만 골라 씀
offs, full_rr, keep = 0, [], []
tri_all = []
for h, hs in enumerate(SRC_FULL):
    vn = hs["vertno"]
    to_local = -np.ones(hs["rr"].shape[0], int)
    to_local[vn] = np.arange(len(vn)) + offs
    tri_all.append(to_local[hs["use_tris"]])
    full_rr.append(hs["rr"][vn] * 1000)
    keep.append(np.searchsorted(vn, fsrc[h]["vertno"]) + offs)
    offs += len(vn)
full_rr = np.vstack(full_rr)
keep = np.concatenate(keep)
tri_all = np.vstack(tri_all)
assert np.allclose(np.linalg.norm(full_rr[keep] - full_rr[keep].mean(0), axis=1).mean(),
                   np.linalg.norm(pos - pos.mean(0), axis=1).mean(), atol=1e-3)
ei = np.concatenate([tri_all[:, [0, 1]], tri_all[:, [1, 2]], tri_all[:, [2, 0]]])
ew = np.linalg.norm(full_rr[ei[:, 0]] - full_rr[ei[:, 1]], axis=1)
graph = sp.coo_matrix((ew, (ei[:, 0], ei[:, 1])), shape=(len(full_rr),) * 2).tocsr()
NB_IDX, NB_D = [None] * N_SRC, [None] * N_SRC
for a in range(0, N_SRC, 400):
    ids = np.arange(a, min(a + 400, N_SRC))
    D = dijkstra(graph, directed=False, indices=keep[ids], limit=PATCH_LIMIT)[:, keep]
    for r, i in enumerate(ids):
        ok = np.where(np.isfinite(D[r]))[0]
        NB_IDX[i], NB_D[i] = ok, D[r, ok]
print(f"측지 이웃 계산 완료 (패치 반경 {PATCH_LIMIT:.0f} mm 안 평균 {np.mean([len(x) for x in NB_IDX]):.1f}개 위치)")

# ---------------- Jansen-Rit 파형 은행 ----------------
def jansen_rit_bank(n, seconds=1.6, transient=1.0, dt=1e-4):
    A, B, a, b, C = 3.25, 22.0, 100.0, 50.0, 135.0
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    e0, v0, r = 2.5, 6.0, 0.56
    S = lambda v: 2 * e0 / (1 + np.exp(r * (v0 - v)))
    # 입력 세기 p: p >= 150 이면 10~11 Hz 알파 리듬이 안정적으로 생김 (p=120 근처는 진동이 시작되는 경계라 느리고 뾰족함).
    # 65%는 알파 영역(150~260), 35%는 진동 없이 잡음이 만드는 불규칙한 활동(70~110)
    pm = np.where(rng.random(n) < 0.65, rng.uniform(150, 260, n), rng.uniform(70, 110, n))
    y = rng.normal(scale=0.1, size=(6, n))
    steps, every = int((seconds + transient) / dt), int(round(1 / FS / dt))
    rec = []
    for k in range(steps):
        p = pm + 22.0 * rng.normal(size=n)
        y0, y1, y2, y3, y4, y5 = y
        d = np.stack([y3, y4, y5,
                      A * a * S(y1 - y2) - 2 * a * y3 - a * a * y0,
                      A * a * (p + C2 * S(C1 * y0)) - 2 * a * y4 - a * a * y1,
                      B * b * C4 * S(C3 * y0) - 2 * b * y5 - b * b * y2])
        y = y + dt * d
        if k >= int(transient / dt) and (k % every == 0):
            rec.append(y[1] - y[2])
    out = np.array(rec[:int(seconds * FS)]).T          # (n, 400)
    out = out - out.mean(1, keepdims=True)
    return (out / (out.std(1, keepdims=True) + 1e-9)).astype(np.float32)


t0 = time.time()
BANK = jansen_rit_bank(N_BANK)
spec = np.abs(np.fft.rfft(BANK, axis=1)) ** 2
fpk = np.fft.rfftfreq(BANK.shape[1], 1 / FS)[spec[:, 1:].argmax(1) + 1]
print(f"Jansen-Rit 파형 {N_BANK}개 ({time.time() - t0:.0f} s). 가장 센 주파수 분포(Hz): "
      f"10%={np.percentile(fpk, 10):.1f}  중앙={np.median(fpk):.1f}  90%={np.percentile(fpk, 90):.1f}")


def waveforms(m, rho):
    """(m, 3, T) 파형. rho: (m,) 두 번째·세 번째 source가 첫 source와 같이 움직이는 정도"""
    bi = rng.integers(0, BANK.shape[0], (m, 3))
    warp = rng.uniform(0.6, 1.4, (m, 3, 1))              # 시간을 늘이거나 줄여 주파수를 6~15 Hz로 흩어 놓음
    off = rng.uniform(0, BANK.shape[1] - 1 - warp * (T - 1), (m, 3, 1))
    pos_t = off + warp * np.arange(T)
    i0 = np.floor(pos_t).astype(int)
    fr_ = (pos_t - i0).astype(np.float32)
    W = BANK[bi[..., None], i0] * (1 - fr_) + BANK[bi[..., None], i0 + 1] * fr_
    t = np.arange(T) / FS
    t0_ = rng.uniform(0.08, 0.28, (m, 3, 1))
    f = rng.uniform(5, 15, (m, 3, 1))
    st = rng.uniform(0.03, 0.06, (m, 3, 1))
    ae = rng.uniform(1.5, 3.0, (m, 3, 1))
    ev = ae * np.exp(-(t - t0_) ** 2 / (2 * st**2)) * np.sin(2 * np.pi * f * (t - t0_))
    W = (W + ev).astype(np.float32)
    r = np.asarray(rho, dtype=np.float32).reshape(-1, 1)
    for k in (1, 2):
        W[:, k] = r * W[:, 0] + np.sqrt(1 - r**2) * W[:, k]
    return W


# ---------------- 시뮬레이션 ----------------
def pick_sources(K, sep_range=None):
    """K개 위치 고르기. sep_range=(lo,hi)이면 첫 두 source 사이 간격을 그 범위로."""
    while True:
        idx = [int(rng.integers(N_SRC))]
        ok = True
        for k in range(1, K):
            d_all = [np.linalg.norm(pos - pos[i], axis=1) for i in idx]
            mask = np.all([d >= SEP_MIN for d in d_all], axis=0)
            if k == 1 and sep_range is not None:
                mask &= (d_all[0] >= sep_range[0]) & (d_all[0] < sep_range[1])
            cand = np.where(mask)[0]
            if len(cand) == 0:
                ok = False
                break
            idx.append(int(rng.choice(cand)))
        if ok:
            return idx


def patch(i, sigma):
    nb, d = NB_IDX[i], NB_D[i]
    keepm = d <= 2.0 * sigma
    return nb[keepm], np.exp(-d[keepm] ** 2 / (2 * sigma**2))


def simulate(Gt, n, Ks, tilt_max=30.0, rho=0.0, sep_range=None, snr_db=SNR_DB, single_vertex=False):
    """Ks: 정수 하나 또는 (n,) 배열. 돌려주는 것: 노이즈 낀 EEG (n,64,T), 정답 위치 (n,3개, 패딩 -1), 학습용 목표 분포(희소)."""
    Ks = np.full(n, Ks) if np.isscalar(Ks) else np.asarray(Ks)
    rho_arr = np.full(n, rho, dtype=float) if np.isscalar(rho) else np.asarray(rho)
    pat = np.zeros((n, 3, 64), np.float32)
    centers = -np.ones((n, 3), int)
    tv, tw = [], []
    for s in range(n):
        idx = pick_sources(int(Ks[s]), sep_range)
        vs_all, ws_all = [], []
        for k, i in enumerate(idx):
            sigma = 1e-3 if single_vertex else rng.uniform(*SIGMA_RANGE)
            vs, w = patch(i, sigma)
            th, ph = np.deg2rad(rng.uniform(0, tilt_max)), rng.uniform(0, 2 * np.pi)
            orient = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            pat[s, k] = np.einsum("v,vck,k->c", w, Gt[vs], orient) * rng.uniform(*AMP_RANGE)
            centers[s, k] = i
            vs_all.append(vs)
            ws_all.append(w / w.sum() / len(idx))          # source마다 같은 질량 1/K
        tv.append(np.concatenate(vs_all))
        tw.append(np.concatenate(ws_all))
    W = waveforms(n, rho_arr)
    clean = np.einsum("nkc,nkt->nct", pat, W)
    ref = clean - clean.mean(1, keepdims=True)
    if np.isfinite(snr_db):
        sig = np.sqrt(np.mean(ref**2, axis=(1, 2), keepdims=True) / 10 ** (snr_db / 10))
        clean = clean + rng.normal(size=clean.shape).astype(np.float32) * sig
    return clean.astype(np.float32), centers, Ks, tv, tw


# ---------------- 특징 ----------------
def feat_time(X):
    """시계열 특징: 평균 기준 → 공간 SVD 상위 RANK개 (U·s), 전체 크기로 나눔, 부호 통일."""
    Xr = X - X.mean(1, keepdims=True)
    U, S, _ = np.linalg.svd(Xr, full_matrices=False)
    nrm = np.sqrt((S**2).sum(1, keepdims=True)) + 1e-12
    U, S = U[:, :, :RANK], S[:, :RANK]
    sgn = np.sign(np.take_along_axis(U, np.abs(U).argmax(1)[:, None, :], 1))
    return ((U * sgn) * (S / nrm)[:, None, :]).reshape(len(X), -1).astype(np.float32)


def feat_snap(X):
    """한 순간 특징: 전체 전력이 가장 센 시점의 전위 패턴 (크기 1, 부호 통일)."""
    Xr = X - X.mean(1, keepdims=True)
    tpk = (Xr**2).sum(1).argmax(1)
    x = Xr[np.arange(len(X)), :, tpk]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    return (x * np.sign(x[np.arange(len(X)), np.abs(x).argmax(1)])[:, None]).astype(np.float32)


# ---------------- numpy 신경망 (softmax + 교차 엔트로피, Adam) ----------------
class MLP:
    def __init__(self, din, hid, dout, seed):
        r = np.random.default_rng(seed)
        dims = [din, hid, hid, dout]
        self.W = [(r.normal(size=(a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b, np.float32) for b in dims[1:]]
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def forward(self, x):
        h1 = np.maximum(x @ self.W[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.W[1] + self.b[1], 0)
        return x, h1, h2, h2 @ self.W[2] + self.b[2]

    @staticmethod
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def step(self, x, tgt, lr):
        x, h1, h2, z = self.forward(x)
        p = self.softmax(z)
        dz = (p - tgt) / len(x)
        g3 = [h2.T @ dz, dz.sum(0)]
        d2 = (dz @ self.W[2].T) * (h2 > 0)
        g2 = [h1.T @ d2, d2.sum(0)]
        d1 = (d2 @ self.W[1].T) * (h1 > 0)
        g1 = [x.T @ d1, d1.sum(0)]
        grads = [g1[0], g2[0], g3[0], g1[1], g2[1], g3[1]]
        params = self.W + self.b
        self.t += 1
        for i, (p_, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh, vh = self.m[i] / (1 - 0.9**self.t), self.v[i] / (1 - 0.999**self.t)
            p_ -= lr * mh / (np.sqrt(vh) + 1e-8)

    def loss(self, x, tgt):
        z = self.forward(x)[3]
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(-(tgt * lp).sum(1).mean())

    def predict(self, x, bs=512):
        return np.vstack([self.softmax(self.forward(x[i:i + bs])[3]) for i in range(0, len(x), bs)])


def dense_targets(tv, tw, ids):
    out = np.zeros((len(ids), N_SRC), np.float32)
    for r, i in enumerate(ids):
        np.add.at(out[r], tv[i], tw[i])
    return out


def train_mlp(F, tv, tw, name, seed):
    n = len(F)
    nv = n // 10
    perm = np.random.default_rng(seed).permutation(n)
    va, tr = perm[:nv], perm[nv:]
    net = MLP(F.shape[1], HID, N_SRC, seed)
    Fv, Tv = F[va], dense_targets(tv, tw, va)
    best, best_w = 1e9, None
    bs = 128
    total = EPOCHS * (len(tr) // bs)
    it = 0
    for ep in range(EPOCHS):
        np.random.default_rng(seed + ep).shuffle(tr)
        for a in range(0, len(tr) - bs + 1, bs):
            ids = tr[a:a + bs]
            lr = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / total)))
            net.step(F[ids], dense_targets(tv, tw, ids), lr)
            it += 1
        vl = net.loss(Fv, Tv)
        if vl < best:
            best, best_w = vl, ([w.copy() for w in net.W], [b.copy() for b in net.b])
        print(f"  [{name}] epoch {ep + 1:2d}/{EPOCHS}  검증 손실 {vl:.3f}")
    net.W, net.b = best_w
    return net


# ---------------- 고전 방법 (MNE / sLORETA, 시계열 전력 지도) ----------------
def inverse_kernels(Gm3, lam2=1.0 / 9.0):
    k = 64
    R = np.eye(k) - 1.0 / k
    L = R @ Gm3[:, :, 2]
    e, V = np.linalg.eigh(R @ R.T)
    kp = e > 1e-10 * e.max()
    W = (V[:, kp] / np.sqrt(e[kp])).T
    Lw = W @ L
    Lw /= np.sqrt(np.trace(Lw @ Lw.T) / kp.sum())
    A = np.linalg.inv(Lw @ Lw.T + lam2 * np.eye(kp.sum()))
    K_mne = Lw.T @ A @ W
    res = np.einsum("ij,ji->i", Lw.T @ A, Lw)
    return K_mne.astype(np.float32), (K_mne / np.sqrt(res)[:, None]).astype(np.float32)


K_MNE, K_SLOR = inverse_kernels(G_model)


def classical_maps(X, kern, chunk=40):
    out = []
    for a in range(0, len(X), chunk):
        Xr = X[a:a + chunk] - X[a:a + chunk].mean(1, keepdims=True)
        S = np.matmul(kern, Xr)                           # (m, 위치, T)
        out.append((S**2).mean(2))
    return np.vstack(out)


# ---------------- 평가 ----------------
def pick_peaks(maps, Ks):
    res = []
    for m, K in zip(maps, Ks):
        m = m.astype(np.float64).copy()
        pk = []
        for _ in range(int(K)):
            i = int(np.argmax(m))
            pk.append(i)
            m[np.linalg.norm(pos - pos[i], axis=1) < NMS_R] = -np.inf
        res.append(pk)
    return res


def score(maps, centers, Ks):
    peaks = pick_peaks(maps, Ks)
    errs, near_mass = [], []
    for s, pk in enumerate(peaks):
        tc = centers[s][: int(Ks[s])]
        D = np.linalg.norm(pos[pk][:, None] - pos[tc][None], axis=2)
        r, c = linear_sum_assignment(D)
        errs.extend(D[r, c])
        p = np.maximum(maps[s], 0)
        p = p / (p.sum() + 1e-30)
        inside = np.zeros(N_SRC, bool)
        for i in tc:
            inside |= np.linalg.norm(pos - pos[i], axis=1) <= 10.0
        near_mass.append(p[inside].sum())
    e = np.array(errs)
    return e.mean(), np.mean(e <= 10) * 100, np.mean(e <= 20) * 100, np.mean(near_mass) * 100


rows = []


def run_condition(exp, cond, models, Gt=Gt_true, **kw):
    X, centers, Ks, _, _ = simulate(Gt, N_TE, **kw)
    res = {"MNE": classical_maps(X, K_MNE), "sLORETA": classical_maps(X, K_SLOR)}
    res["MLP 분포 (시계열)"] = models["time"].predict(feat_time(X))
    res["MLP 분포 (한 순간)"] = models["snap"].predict(feat_snap(X))
    print(f" [{cond}]")
    for name, maps in res.items():
        m, w10, w20, nm = score(maps, centers, Ks)
        rows.append((exp, cond, name, m, w10, w20, nm))
        print(f"  {name:<20}: 평균 {m:5.1f} mm  10mm 이내 {w10:5.1f}%  20mm 이내 {w20:5.1f}%  정답근처 질량 {nm:5.1f}%")


# ---------------- E0 검산 ----------------
print("\n=== E0 검산: 점 source 1개, 잡음 없음, 같은 forward (sLORETA는 0 mm여야 함) ===")
X, centers, Ks, _, _ = simulate(Gt_model, 200, 1, tilt_max=0.0, snr_db=np.inf, single_vertex=True)
for name, kern in [("MNE", K_MNE), ("sLORETA", K_SLOR)]:
    m, w10, _, _ = score(classical_maps(X, kern), centers, Ks)
    print(f"  {name:<10}: 평균 {m:.2f} mm, 10mm 이내 {w10:.0f}%")
    rows.append(("E0", "검산(점 source, 잡음 없음, 같은 forward)", name, m, w10, 0, 0))

# ---------------- 학습 ----------------
print(f"\n학습 데이터 생성 {N_TR}개 (source 1~3개, 상관 0~1, 기울기 0~60도, 모델 머리)")
t0 = time.time()
Ft, Fs, TV, TW = [], [], [], []
for a in range(0, N_TR, 2000):
    m = min(2000, N_TR - a)
    rho_tr = np.where(rng.random(m) < 0.5, 0.0, rng.random(m))
    X, _, _, tv, tw = simulate(Gt_model, m, rng.integers(1, 4, m), tilt_max=60.0, rho=rho_tr)
    Ft.append(feat_time(X))
    Fs.append(feat_snap(X))
    TV += tv
    TW += tw
Ft, Fs = np.vstack(Ft), np.vstack(Fs)
print(f"  데이터 {time.time() - t0:.0f} s, 특징 크기 시계열 {Ft.shape[1]} / 한 순간 {Fs.shape[1]}")
t0 = time.time()
models = {"time": train_mlp(Ft, TV, TW, "시계열", SEED), "snap": train_mlp(Fs, TV, TW, "한 순간", SEED + 1)}
print(f"  학습 {time.time() - t0:.0f} s")

# ---------------- 실험 ----------------
print("\n=== E1: source 개수 (정답 머리에서 테스트, 방향 0~30도 무작위, SNR 10 dB) ===")
for K in (1, 2, 3):
    run_condition("E1", f"source {K}개", models, Ks=K)

print("\n=== E2: source 2개의 간격 ===")
for lo, hi in [(SEP_MIN, 40), (40, 70), (70, 120), (120, 1000)]:
    run_condition("E2", f"간격 {lo:.0f}~{min(hi, 999):.0f} mm", models, Ks=2, sep_range=(lo, hi))

print("\n=== E3: 두 source 파형의 상관 (학습은 상관 0~1 섞음) ===")
for rho in (0.0, 0.5, 0.9):
    run_condition("E3", f"상관 {rho}", models, Ks=2, rho=rho)

print("\n=== E4: 테스트 SNR (학습은 10 dB만) ===")
for snr in (0, 5, 10, 20):
    run_condition("E4", f"SNR {snr} dB", models, Ks=2, snr_db=float(snr))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"multisource_seed{SEED}{'_quick' if QUICK else ''}.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["실험", "조건", "방법", "평균mm", "10mm이내%", "20mm이내%", "정답근처질량%"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", f"{r[4]:.1f}", f"{r[5]:.1f}", f"{r[6]:.1f}"])
print("결과 저장:", out)
