"""Week 8 결과 정리: Localize-MI 실제 기록 (피험자 7명, run 61개)

출력: lmi_summary.md, fig_lmi.svg
불확실성: 피험자 단위 부트스트랩 (피험자를 다시 뽑아 run 오차 중앙값의 차이를 계산, 5,000번). 학습 모델은 시드 평균
실행: python week08/summarize_lmi.py
"""
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
B = pd.read_csv(os.path.join(HERE, "lmi_baseline.csv"))
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "lmi_routes_seed[0-9].csv")))])
seeds = sorted(R["시드"].unique())
per = R.groupby(["피험자", "run", "mA", "방법"])["오차mm"].mean().unstack()           # run × 방법 (시드 평균)
b = B[(B["잡음"] == "없음") & (B["신호"] == "한 시점")].pivot_table(index=["피험자", "run"], columns="방법", values="오차A_중점mm")
b.columns = [f"{c} (데이터 머리 모델, 8-1)" for c in b.columns]
per = per.reset_index().merge(b.reset_index().assign(run=lambda d: d["run"].astype(int)), on=["피험자", "run"]).set_index(["피험자", "run", "mA"])
rng = np.random.default_rng(0)
subs = sorted(per.index.get_level_values(0).unique())


def boot(a, b_, n=5000):
    d = per[a] - per[b_]
    by = {s: d.xs(s, level=0).values for s in subs}
    est = np.median(d.values)
    bs = [np.median(np.concatenate([by[s] for s in rng.choice(subs, len(subs))])) for _ in range(n)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(np.median(by[s]) < 0 for s in subs)


ORDER = ["쌍극자 스캔 (데이터 머리 모델)", "sLORETA (데이터 머리 모델, 8-1)", "MNE (데이터 머리 모델, 8-1)",
         "경로 2: 어댑터 (전용 모델)", "경로 2: 센서만 (전용 모델)",
         "쌍극자 스캔 (변형 머리)", "sLORETA (변형 머리)", "경로 1: 어댑터 (변형 머리)"]
out = [f"# Week 8: Localize-MI 실제 기록 (피험자 7명, run 61개, 학습 모델은 시드 {len(seeds)}개 평균)", "",
       "신호: 자극 펄스 ±2 ms에서 GFP 최대 시점, 백색화 없음. 오차: 추정 위치 → 자극 접점 쌍의 중점", "",
       "## 1. 전체 (run 61개)", "", "| 방법 | 중앙값 mm | 평균 mm | 10 mm 이내 | 20 mm 이내 | 피험자별 중앙값 범위 |", "| --- | --- | --- | --- | --- | --- |"]
for m in ORDER:
    v = per[m]
    ps = [np.median(v.xs(s, level=0)) for s in subs]
    out.append(f"| {m} | {v.median():.1f} | {v.mean():.1f} | {100 * (v < 10).mean():.0f}% | {100 * (v < 20).mean():.0f}% | {min(ps):.1f}~{max(ps):.1f} |")
out += ["", "## 2. 같은 머리 모델 위에서 학습 모델 − 쌍극자 스캔 (run별 차이의 중앙값, 피험자 부트스트랩 95% 구간, 음수면 학습 모델이 나음)", "",
        "| 비교 | 차이 중앙값 mm | 95% 구간 | 학습 모델이 나은 피험자 수 |", "| --- | --- | --- | --- |"]
for a, b_ in (("경로 2: 어댑터 (전용 모델)", "쌍극자 스캔 (데이터 머리 모델)"), ("경로 2: 센서만 (전용 모델)", "쌍극자 스캔 (데이터 머리 모델)"),
              ("경로 1: 어댑터 (변형 머리)", "쌍극자 스캔 (변형 머리)"), ("경로 1: 어댑터 (변형 머리)", "쌍극자 스캔 (데이터 머리 모델)"),
              ("쌍극자 스캔 (변형 머리)", "쌍극자 스캔 (데이터 머리 모델)")):
    e, lo, hi, k = boot(a, b_)
    out.append(f"| {a} − {b_} | {e:+.1f} | {lo:+.1f} ~ {hi:+.1f} | {k} / 7 |")
out += ["", "## 3. 피험자별 중앙값 (mm)", "", "| 피험자 | run 수 | " + " | ".join(ORDER) + " |", "| --- | --- | " + " | ".join(["---"] * len(ORDER)) + " |"]
for s in subs:
    x = per.xs(s, level=0)
    out.append(f"| {s} | {len(x)} | " + " | ".join(f"{np.median(x[m]):.1f}" for m in ORDER) + " |")
out += ["", "## 4. 자극 세기별 중앙값 (mm)", "", "| mA | run 수 | " + " | ".join(ORDER) + " |", "| --- | --- | " + " | ".join(["---"] * len(ORDER)) + " |"]
for ma in sorted(per.index.get_level_values(2).unique()):
    x = per.xs(ma, level=2)
    out.append(f"| {ma} | {len(x)} | " + " | ".join(f"{np.median(x[m]):.1f}" for m in ORDER) + " |")
sd = R.groupby(["피험자", "run", "방법"])["오차mm"].std().groupby("방법").median()
out += ["", "## 5. 시드에 따른 흔들림 (run별 시드 간 표준편차의 중앙값, mm)", ""] + [f"- {k}: {v:.1f}" for k, v in sd.items() if k.startswith("경로")]
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "lmi_summary.md"), "w", encoding="utf-8").write(txt + "\n")

# 그림: 누적 분포
cols = {"쌍극자 스캔 (데이터 머리 모델)": "#b33", "sLORETA (데이터 머리 모델, 8-1)": "#8a9099", "경로 2: 어댑터 (전용 모델)": "#2b6cb0",
        "경로 2: 센서만 (전용 모델)": "#7fb2e5", "쌍극자 스캔 (변형 머리)": "#e8a0a0", "경로 1: 어댑터 (변형 머리)": "#d9822b"}
W, H, x0, x1, y0, y1 = 640, 400, 56, 610, 300, 50
xs = lambda v: x0 + (x1 - x0) * min(v, 60) / 60
ys = lambda p: y0 - (y0 - y1) * p
g = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"><rect width="{W}" height="{H}" fill="#fff"/>',
     '<text x="16" y="22" font-size="13" font-weight="600" fill="#222">Localize-MI 실제 기록: 위치 오차가 이 값 이하인 run의 비율 (피험자 7명, run 61개)</text>',
     '<text x="16" y="38" font-size="10" fill="#555">오차 = 추정 위치 → 자극 접점 쌍의 중점. 학습 모델은 시드 평균</text>']
for p in (0, .25, .5, .75, 1):
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(p):.0f}" y2="{ys(p):.0f}" stroke="#e6e6e6"/><text x="{x0 - 6}" y="{ys(p) + 4:.0f}" font-size="10" text-anchor="end" fill="#666">{int(p * 100)}%</text>')
for v in (0, 10, 20, 30, 40, 50, 60):
    g.append(f'<text x="{xs(v):.0f}" y="{y0 + 15}" font-size="10" text-anchor="middle" fill="#666">{v}</text>')
g.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 30}" font-size="10" text-anchor="middle" fill="#555">위치 오차 mm</text>')
for i, (m, c) in enumerate(cols.items()):
    v = np.sort(per[m].values)
    pts = [f"{xs(0):.1f},{ys(0):.1f}"] + [f"{xs(e):.1f},{ys((j + 1) / len(v)):.1f}" for j, e in enumerate(v) if e <= 60]
    g.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
    g.append(f'<rect x="{56 + (i % 2) * 290}" y="{342 + (i // 2) * 16}" width="10" height="10" fill="{c}"/><text x="{71 + (i % 2) * 290}" y="{351 + (i // 2) * 16}" font-size="10" fill="#333">{m}</text>')
g.append("</svg>")
open(os.path.join(HERE, "fig_lmi.svg"), "w", encoding="utf-8").write("".join(g))
