"""Week 3: mini_e2e 확장 비교 실험.
(1) SNR별로 dipole scan 오차가 어떻게 변하나 (13 mm가 높은지 확인)
(2) 선형 모델 vs MLP (비선형성이 필요한가)
(3) 고전 역해법 MNE / sLORETA 와 비교
"""
import time
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

mne.set_log_level("ERROR")
rng = np.random.default_rng(0)

names = ("Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 CP5 CP1 CP2 CP6 "
         "P7 P3 Pz P4 P8 O1 Oz O2").split()
try:
    montage = mne.channels.make_standard_montage("colin27_1020")
except ValueError:
    montage = mne.channels.make_standard_montage("standard_1020")
pos_all = montage.get_positions()["ch_pos"]
ch_pos = {n: pos_all[n] / np.linalg.norm(pos_all[n]) * 0.096 for n in names}
info = mne.create_info(names, sfreq=250.0, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.002), head_radius=0.095, info=info)
src = mne.setup_volume_source_space(sphere=(0.0, 0.0, 0.0, 80.0), pos=10.0, sphere_units="mm")
fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0)
G = fwd["sol"]["data"]
pos = fwd["source_rr"] * 1000
n_ch, n_src = G.shape[0], pos.shape[0]
G3 = G.reshape(n_ch, n_src, 3)


_nrm = np.linalg.norm(pos, axis=1, keepdims=True)
RADIAL = np.where(_nrm > 1e-6, pos / np.maximum(_nrm, 1e-6), np.array([0.0, 0.0, 1.0]))  # 바깥 방향 (중심점은 위쪽)


def simulate(n, snr_db, fixed=False):
    idx = rng.integers(0, n_src, n)
    if fixed:  # 위치마다 방향을 하나로 고정 (구 바깥 방향), 세기는 양수
        u = RADIAL[idx]
    else:      # 방향 무작위 (부호도 무작위)
        u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    clean = np.einsum("cnk,nk->nc", G3[:, idx, :], u)
    y = clean.copy()
    if np.isfinite(snr_db):
        p = np.mean(clean**2, axis=1, keepdims=True)
        y = y + rng.normal(size=clean.shape) * np.sqrt(p / 10 ** (snr_db / 10))
    y = y - y.mean(axis=1, keepdims=True)
    return y / np.linalg.norm(y, axis=1, keepdims=True), idx


Gc = G3 - G3.mean(axis=0, keepdims=True)
P = np.stack([Gc[:, i, :] @ np.linalg.pinv(Gc[:, i, :]) for i in range(n_src)])


def dipole_scan(X):
    return pos[np.einsum("nc,icd,nd->ni", X, P, X).argmax(axis=1)]


# 고전 역해법: MNE-Python의 minimum_norm (평균 참조, 임의 잡음 공분산)
info_ref = info.copy()
raw_tmp = mne.io.RawArray(np.zeros((n_ch, 10)), info_ref)
raw_tmp.set_eeg_reference("average", projection=True)
info_ref = raw_tmp.info
cov = mne.make_ad_hoc_cov(info_ref)
inv = make_inverse_operator(info_ref, fwd, cov, loose=1.0, depth=None)


def classical(X, method):
    ev = mne.EvokedArray(X.T * 1e-6, info_ref, tmin=0.0)  # 시점 = 샘플
    ev.apply_proj()
    stc = apply_inverse(ev, inv, lambda2=1.0 / 9.0, method=method, pick_ori=None)
    return pos[np.abs(stc.data).argmax(axis=0)]


def err(pred, true):
    e = np.linalg.norm(pred - true, axis=1)
    return f"평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}"


for fixed, snr in [(False, 10.0), (False, 30.0), (False, np.inf), (True, 10.0)]:
    Xtr, itr = simulate(20000, snr, fixed)
    Xte, ite = simulate(2000, snr, fixed)
    Ytr, Yte = pos[itr], pos[ite]
    label = "잡음 없음" if not np.isfinite(snr) else f"SNR {snr:.0f} dB"
    label += ", 방향 고정" if fixed else ", 방향 무작위"
    print(f"\n=== {label} (테스트 2000개, 위치 오차 mm) ===")
    print(f"  항상 중심        : {err(np.tile(pos.mean(0), (len(Yte), 1)), Yte)}")
    t = time.time(); ds = dipole_scan(Xte); t_ds = time.time() - t
    print(f"  dipole scan      : {err(ds, Yte)}   ({t_ds:.2f} s)")
    for m in ["MNE", "sLORETA"]:
        t = time.time(); pr = classical(Xte, m); tt = time.time() - t
        print(f"  {m:<16} : {err(pr, Yte)}   ({tt:.2f} s)")
    lin = Ridge(alpha=1e-3).fit(Xtr, Ytr / 100.0)
    print(f"  선형 (Ridge)     : {err(lin.predict(Xte) * 100, Yte)}")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=60, random_state=0).fit(Xtr, Ytr / 100.0)
    print(f"  MLP (ReLU)       : {err(mlp.predict(Xte) * 100, Yte)}")
    mlpl = MLPRegressor(hidden_layer_sizes=(128, 64), activation="identity", max_iter=60, random_state=0).fit(Xtr, Ytr / 100.0)
    print(f"  MLP (활성함수 없음): {err(mlpl.predict(Xte) * 100, Yte)}")
