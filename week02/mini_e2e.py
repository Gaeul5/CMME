"""가장 작은 end-to-end: 시뮬레이션 EEG -> 작은 신경망 -> source 위치 -> 오차(mm).

샘플 데이터(1.6GB) 없이 돌아가도록 구형 머리 모델 + 표준 10-20 전극을 쓴다.
"""
import numpy as np
import mne
from sklearn.neural_network import MLPRegressor

mne.set_log_level("WARNING")
rng = np.random.default_rng(0)

# 1. 전극과 머리 모델 (구형 3층: brain / skull / scalp)
try:
    montage = mne.channels.make_standard_montage("standard_1020")
except ValueError:  # 최신 MNE에서는 이름이 바뀜
    montage = mne.channels.make_standard_montage("colin27_1020")
names = ("Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 CP5 CP1 CP2 CP6 "
         "P7 P3 Pz P4 P8 O1 Oz O2").split()  # 28채널
pos_all = montage.get_positions()["ch_pos"]
R = 0.096  # 전극을 반지름 96 mm 구 표면에 맞춤 (머리 구는 95 mm)
ch_pos = {n: pos_all[n] / np.linalg.norm(pos_all[n]) * R for n in names}
info = mne.create_info(names, sfreq=250.0, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"))
# 구 중심을 2 mm 옮김: 격자점이 정확히 구 중심에 놓이면 계산이 NaN이 됨
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.002), head_radius=0.095, info=info)

# 2. source space: 구 안쪽에 10 mm 간격 격자 (후보 위치)
src = mne.setup_volume_source_space(
    sphere=(0.0, 0.0, 0.0, 80.0), pos=10.0, sphere_units="mm"
)

# 3. forward: leadfield G, shape = (채널 수, 소스 수 x 3방향)
fwd = mne.make_forward_solution(
    info, trans=None, src=src, bem=sphere, meg=False, eeg=True, mindist=5.0
)
G = fwd["sol"]["data"]
pos = fwd["source_rr"] * 1000  # mm
n_ch, n_src = G.shape[0], pos.shape[0]
G3 = G.reshape(n_ch, n_src, 3)
print(f"채널 {n_ch}개, 소스 위치 {n_src}개, G.shape={G.shape}")


# 4. 시뮬레이션: 소스 위치 i, 방향 u, 세기 a -> 센서 신호 y = a * G_i u + 잡음
def simulate(n, snr_db):
    idx = rng.integers(0, n_src, n)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    clean = np.einsum("cnk,nk->nc", G3[:, idx, :], u)
    sig_pow = np.mean(clean**2, axis=1, keepdims=True)
    noise = rng.normal(size=clean.shape) * np.sqrt(sig_pow / 10 ** (snr_db / 10))
    y = clean + noise
    y = y - y.mean(axis=1, keepdims=True)  # average reference
    y = y / np.linalg.norm(y, axis=1, keepdims=True)  # 세기는 버리고 모양만
    return y, idx


SNR = 10.0
Xtr, itr = simulate(20000, SNR)
Xte, ite = simulate(2000, SNR)
Ytr, Yte = pos[itr], pos[ite]

# 5. 작은 신경망: EEG 벡터 -> (x, y, z)
net = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=60, random_state=0)
net.fit(Xtr, Ytr / 100.0)
pred = net.predict(Xte) * 100.0
err_nn = np.linalg.norm(pred - Yte, axis=1)

# 6. 비교 기준 1: 항상 중심을 답하기
err_center = np.linalg.norm(Yte - pos.mean(axis=0), axis=1)

# 7. 비교 기준 2: 고전적 dipole scan (모든 후보 위치에 방향 3개를 최소제곱으로 맞춤)
Gc = G3 - G3.mean(axis=0, keepdims=True)  # average reference
P = np.empty((n_src, n_ch, n_ch))
for i in range(n_src):
    Gi = Gc[:, i, :]
    P[i] = Gi @ np.linalg.pinv(Gi)
score = np.einsum("nc,icd,nd->ni", Xte, P, Xte)  # 클수록 잘 설명
best = score.argmax(axis=1)
err_scan = np.linalg.norm(pos[best] - Yte, axis=1)

print(f"\nSNR {SNR:.0f} dB, 테스트 {len(Xte)}개, 위치 오차(mm)")
print(f"  항상 중심이라고 답하기 : 평균 {err_center.mean():5.1f}  중앙값 {np.median(err_center):5.1f}")
print(f"  작은 신경망 (MLP)      : 평균 {err_nn.mean():5.1f}  중앙값 {np.median(err_nn):5.1f}")
print(f"  dipole scan (고전 방식): 평균 {err_scan.mean():5.1f}  중앙값 {np.median(err_scan):5.1f}")
