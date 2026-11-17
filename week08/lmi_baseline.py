"""Week 8-1: Localize-MI 실제 기록에서 고전 방법의 위치 오차 (피험자 7명, run 61개)

정답: 자극한 접점 쌍의 중점 (T1w 좌표)
오차 두 가지
  A. 추정 위치 → 접점 중점 거리
  B. 추정 위치 → '접점 중점에서 가장 가까운 source 위치' 거리 (source 공간이 피질에만 있어 생기는 바닥값을 뺀 것)
방법: 쌍극자 스캔, sLORETA, MNE (자유 방향). 리드필드는 데이터에 들어 있는 피험자별 순방향 모델
신호: 자극 펄스 −2~2 ms 창의 첫 SVD 성분, 그리고 그 창에서 GFP가 가장 큰 한 시점 (Mikulan 외와 같은 방식)
잡음: 자극 전 기준선으로 백색화. 백색화 없는 결과도 같이 저장

실행: python week08/lmi_baseline.py "<Localize MI 폴더>"
"""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lmi_common as C  # noqa: E402

root = sys.argv[1]
R = C.runs(root)
rows = []
cache = {}
for r in R:
    sub = r["sub"]
    if sub not in cache:
        cache[sub] = (C.load_fwd(root, sub), C.contacts_xyz(root, sub))
    (G, pos, names, bads), cx = cache[sub]
    bads = bads | C.bad_channels(root, sub, r["run"])
    good = np.array([n not in bads for n in names])
    mid = (cx[r["contacts"][0]] + cx[r["contacts"][1]]) / 2
    near = np.linalg.norm(pos - mid, axis=1).argmin()
    floor = np.linalg.norm(pos[near] - mid) * 1000
    Y = C.evoked(root, sub, r["run"])
    for white in (True, False):
        if white:
            Yw, Gw = C.prepare(Y, G, good)
        else:
            Rm = np.eye(good.sum()) - 1.0 / good.sum()
            Yw, Gw = (Rm @ Y[good])[:, C.tidx(*C.WIN)], np.einsum("ck,kvj->cvj", Rm, G[good])
        u, frac = C.top_pattern(Yw)
        ui = Yw[:, np.argmax((Yw**2).sum(0))]
        for sig, vec in (("창 첫 성분", u), ("한 시점", ui / np.linalg.norm(ui))):
            mn, sl = C.min_norm(vec, Gw)
            for meth, sc in (("쌍극자 스캔", C.scan(vec, Gw)), ("sLORETA", sl), ("MNE", mn)):
                v = int(np.argmax(sc))
                rows.append([sub, r["run"], "-".join(r["contacts"]), r["mA"], "백색화" if white else "없음", sig, meth,
                             f"{np.linalg.norm(pos[v] - mid) * 1000:.2f}", f"{np.linalg.norm(pos[v] - pos[near]) * 1000:.2f}",
                             f"{floor:.2f}", f"{frac:.3f}", int(good.sum())])
    print(sub, r["run"], r["contacts"], r["mA"], "바닥", round(floor, 1), "→",
          {x[6]: x[7] for x in rows[-18:] if x[4] == "백색화" and x[5] == "창 첫 성분"}, flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lmi_baseline.csv")
with open(out, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["피험자", "run", "접점", "mA", "잡음", "신호", "방법", "오차A_중점mm", "오차B_가까운source_mm", "바닥mm", "첫성분비율", "채널수"])
    w.writerows(rows)
print("저장:", out)
