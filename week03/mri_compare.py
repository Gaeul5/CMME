"""Week 3: 구형 모델 대신 실제 MRI 기반 머리 모델로 비교 실험.

MNE sample 데이터(한 사람의 실제 MRI)로 만든 3층 BEM과 피질 source space를 사용한다.
- source: 피질 위 위치, 방향은 피질 표면에 수직으로 고정 (연구실 논문과 같은 방식)
- 전극: sample 데이터의 실제 EEG 전극 위치 (59채널)
- 조건: SNR 10 / 30 dB / 잡음 없음, 그리고 실제 방향이 피질 수직에서 30도 틀어진 경우
  (30도 조건: 학습은 수직 방향 데이터로, 테스트만 틀어진 데이터로 함)

실행: python week03/mri_compare.py [데이터경로 또는 -] [orig59|biosemi16|biosemi32|biosemi64]
(MNE sample 데이터가 없으면 처음 실행 때 약 1.6GB를 내려받는다)
"""
import os
import sys
import time
import numpy as np
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

mne.set_log_level("ERROR")
rng = np.random.default_rng(0)

# 1. 데이터 위치 (기본: MNE sample 데이터. 다른 폴더를 쓰려면 인자로 경로 전달)
data_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else str(mne.datasets.sample.data_path())


def first_existing(*cands):
    for c in cands:
        p = os.path.join(data_path, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cands)


bem_file = first_existing("subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif",
                          "subjects/sample/bem/sample-1280-1280-1280-bem-sol.fif")
src_file = first_existing("subjects/sample/bem/sample-oct-6-src.fif")
raw_file = first_existing("MEG/sample/sample_audvis_raw.fif",
                          "MEG/sample/sample_audvis_trunc_raw.fif")
trans_file = first_existing("MEG/sample/sample_audvis_raw-trans.fif",
                            "MEG/sample/sample_audvis_trunc-trans.fif")
print("BEM :", os.path.basename(bem_file))

# 2. 전극 위치
#   orig59: sample 데이터의 실제 EEG 전극 (불량 채널 제외 59개)
#   biosemi16/32/64: Biosemi 표준 캡. 표준 위치를 sample 머리 크기에 맞게 균일하게 키운 뒤,
#   forward 계산 때 MNE가 BEM 두피 표면으로 전극을 투영한다
MONTAGE = sys.argv[2] if len(sys.argv) > 2 else "orig59"
info_raw = mne.io.read_info(raw_file)
info = mne.pick_info(info_raw, mne.pick_types(info_raw, meg=False, eeg=True, exclude="bads"))
if MONTAGE != "orig59":
    real = np.array([ch["loc"][:3] for ch in info["chs"]])  # sample 실제 전극 (head 좌표, m)
    montage = mne.channels.make_standard_montage(MONTAGE)
    std = montage.get_positions()["ch_pos"]
    std_arr = np.array([std[n] for n in montage.ch_names])
    # 두 전극 세트의 윗부분(z>0) 평균 반지름 비율로 균일 스케일
    scale = np.linalg.norm(real[real[:, 2] > 0], axis=1).mean() / np.linalg.norm(std_arr[std_arr[:, 2] > 0], axis=1).mean()
    ch_pos = {n: std[n] * scale for n in montage.ch_names}
    info = mne.create_info(montage.ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
    print(f"표준 몽타주 스케일 {scale:.3f}")
print(f"몽타주 {MONTAGE}: {len(info['ch_names'])}채널")

# 3. forward: 실제 MRI 기반 3층 BEM + 피질 source space
src = mne.read_source_spaces(src_file)
fwd = mne.make_forward_solution(info, trans=trans_file, src=src, bem=bem_file,
                                meg=False, eeg=True, mindist=5.0)
# 피질 기준 좌표계로 바꿈: 위치마다 [접선1, 접선2, 수직] 3열
fwd_surf = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False, use_cps=True)
fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)

n_ch = fwd_surf["sol"]["data"].shape[0]
G3 = fwd_surf["sol"]["data"].reshape(n_ch, -1, 3)  # (채널, 위치, 3)
pos = fwd_surf["source_rr"] * 1000  # mm
n_src = pos.shape[0]
G_norm = G3[:, :, 2]  # 피질 수직 방향 leadfield
print(f"채널 {n_ch}개, 피질 source {n_src}개")


def simulate(n, snr_db, tilt_deg=0.0):
    """위치 하나에 피질 수직 방향(또는 tilt_deg만큼 기울어진 방향) 쌍극자, 세기는 양수."""
    idx = rng.integers(0, n_src, n)
    th = np.deg2rad(tilt_deg)
    ph = rng.uniform(0, 2 * np.pi, n)
    w = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                  np.full(n, np.cos(th))], axis=1)  # [접선1, 접선2, 수직] 비율
    clean = np.einsum("cnk,nk->nc", G3[:, idx, :], w)
    y = clean.copy()
    if np.isfinite(snr_db):
        p = np.mean(clean**2, axis=1, keepdims=True)
        y = y + rng.normal(size=clean.shape) * np.sqrt(p / 10 ** (snr_db / 10))
    y = y - y.mean(axis=1, keepdims=True)  # average reference
    return y / np.linalg.norm(y, axis=1, keepdims=True), idx


# 4. dipole scan: 방향을 피질 수직으로 알고 있다고 보고 위치만 탐색
Gc = G_norm - G_norm.mean(axis=0, keepdims=True)
Gc_unit = Gc / np.linalg.norm(Gc, axis=0, keepdims=True)


def dipole_scan(X):
    return pos[np.abs(X @ Gc_unit).argmax(axis=1)]


# 5. MNE / sLORETA (방향 고정 inverse)
raw_tmp = mne.io.RawArray(np.zeros((n_ch, 10)), info)
raw_tmp.set_eeg_reference("average", projection=True)
info_ref = raw_tmp.info
cov = mne.make_ad_hoc_cov(info_ref)
inv = make_inverse_operator(info_ref, fwd_fixed, cov, fixed=True, depth=None)


def classical(X, method):
    ev = mne.EvokedArray(X.T * 1e-6, info_ref, tmin=0.0)
    ev.apply_proj()
    stc = apply_inverse(ev, inv, lambda2=1.0 / 9.0, method=method)
    return pos[np.abs(stc.data).argmax(axis=0)]


def err(pred, true):
    e = np.linalg.norm(pred - true, axis=1)
    return f"평균 {e.mean():5.1f}  중앙값 {np.median(e):5.1f}"


for snr, tilt in [(10.0, 0.0), (30.0, 0.0), (np.inf, 0.0), (10.0, 30.0)]:
    # 학습은 항상 피질 수직 방향 데이터로 함 (논문처럼). 30도 조건은 테스트 데이터만 틀어짐
    Xtr, itr = simulate(20000, snr, 0.0)
    Xte, ite = simulate(2000, snr, tilt)
    Ytr, Yte = pos[itr], pos[ite]
    label = "잡음 없음" if not np.isfinite(snr) else f"SNR {snr:.0f} dB"
    label += f", 방향 {tilt:.0f}도 틀어짐" if tilt else ", 피질 수직 방향"
    print(f"\n=== {label} (테스트 2000개, 위치 오차 mm) ===")
    print(f"  항상 중심        : {err(np.tile(pos.mean(0), (len(Yte), 1)), Yte)}")
    t = time.time(); ds = dipole_scan(Xte); tt = time.time() - t
    print(f"  dipole scan      : {err(ds, Yte)}   ({tt:.2f} s)")
    for m in ["MNE", "sLORETA"]:
        t = time.time(); pr = classical(Xte, m); tt = time.time() - t
        print(f"  {m:<16} : {err(pr, Yte)}   ({tt:.2f} s)")
    lin = Ridge(alpha=1e-3).fit(Xtr, Ytr / 100.0)
    print(f"  선형 (Ridge)     : {err(lin.predict(Xte) * 100, Yte)}")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=60, random_state=0).fit(Xtr, Ytr / 100.0)
    print(f"  MLP (ReLU)       : {err(mlp.predict(Xte) * 100, Yte)}")
