"""Week 6-3: 머리끼리 리드필드가 얼마나 다른가 (가짜 머리 vs OASIS 진짜 머리)

Week 5의 숫자: 가짜 머리끼리 리드필드 상관 0.995 (너무 비슷해서 어댑터 효과가 작게 나왔다고 해석함).
OASIS 머리끼리 이 값이 훨씬 낮게 나와야 그 해석이 맞다.

재는 것 (머리 두 개씩 짝지어)
  - 리드필드 상관: 모델 리드필드의 피질 수직 성분(64채널 x 공통 source), 평균 기준으로 바꾼 뒤 전체를 펼쳐 피어슨 상관
  - source 위치 차이: 같은 번호 source끼리 거리의 평균 (mm). OASIS는 fsaverage 번호로 맞춰져 있음

출력은 숫자 요약만 화면에 찍는다. npz(리드필드)나 MRI 파생 파일은 만들지 않는다.

실행
  python week06/head_similarity.py week05/heads_oasis/*.npz         (OASIS, 연구실 컴퓨터에서)
  python week06/head_similarity.py week05/cache_heads/head0*.npz     (가짜 머리, 비교용)
"""
import itertools
import os
import sys
import numpy as np

files = sorted(sys.argv[1:])
assert len(files) >= 2, "npz 파일을 2개 이상 주세요"
G, P, V, names = [], [], [], []
for f in files:
    z = np.load(f)
    g = z["G_model"][:, :, 2].astype(np.float64)
    G.append(g - g.mean(0, keepdims=True))          # 평균 기준
    P.append(z["pos"])
    V.append(z["valid"] if "valid" in z.files else np.ones(g.shape[1], bool))
    names.append(os.path.basename(f).replace(".npz", ""))
common = np.logical_and.reduce(V)
print(f"머리 {len(files)}개, 공통 유효 source {common.sum()}/{len(common)}곳")

C, D = [], []
for i, j in itertools.combinations(range(len(files)), 2):
    a, b = G[i][:, common].ravel(), G[j][:, common].ravel()
    C.append(np.corrcoef(a, b)[0, 1])
    D.append(np.linalg.norm(P[i][common] - P[j][common], axis=1).mean())
C, D = np.array(C), np.array(D)
print(f"짝 {len(C)}개")
print(f"리드필드 상관   : 평균 {C.mean():.3f}, 최소 {C.min():.3f}, 최대 {C.max():.3f}")
print(f"source 위치 차이: 평균 {D.mean():.1f} mm, 최소 {D.min():.1f}, 최대 {D.max():.1f}")
print("머리별 (다른 머리들과의 평균 상관):")
M = np.full((len(files), len(files)), np.nan)
for (i, j), c in zip(itertools.combinations(range(len(files)), 2), C):
    M[i, j] = M[j, i] = c
for n, row in zip(names, M):
    print(f"  {n:<28} {np.nanmean(row):.3f}")
