"""Week 7-1 결과 정리: 신호원 1~3개를 '가장 센 것부터 하나씩' 찾기

출력: ranked_summary.md, fig_ranked_dist.svg (64채널 오차 분포)
실행: python week07/summarize_ranked.py
"""
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
ORDER = ["한 시점, sLORETA 봉우리", "sLORETA 순차 제거", "어댑터 순차 제거 (일반 학습)", "어댑터 순차 제거 (제거 학습)", "마스크 순차 제거 (제거 학습)"]
COL = ["#9aa0a6", "#444b54", "#7fb2e5", "#2b6cb0", "#d9822b"]


def ci(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f} ± {T975[len(v)] * v.std(ddof=1) / np.sqrt(len(v)):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


R = []
for f in sorted(glob.glob(os.path.join(HERE, "ranked_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        R += list(csv.DictReader(fh))
seeds = sorted({r["시드"] for r in R})


def ps(mont, meth, k, col):
    out = []
    for s in seeds:
        v = [float(r[col]) for r in R if r["시드"] == s and r["배치"] == mont and r["방법"] == meth and r["신호원수"] == str(k) and r[col] != "nan"]
        out.append(np.mean(v))
    return out


out = [f"# Week 7-1: 가장 센 신호원부터 하나씩 찾기 (OASIS 처음 보는 4명, 시드 {len(seeds)}개, mm ± 시드 간 95% 구간)", ""]
for mont in ["64채널", "32채널", "16채널"]:
    out += [f"## {mont}: 평균 위치 오차 (찾은 것과 정답을 가장 가깝게 짝지은 뒤)", "", "| 방법 | 신호원 1개 | 2개 | 3개 | 놓침 % (1 / 2 / 3개) |", "| --- | --- | --- | --- | --- |"]
    for m in ORDER:
        out.append(f"| {m} | " + " | ".join(ci(ps(mont, m, k, "평균mm")) for k in (1, 2, 3)) + " | " + " / ".join(f"{np.mean(ps(mont, m, k, '놓침%')):.0f}" for k in (1, 2, 3)) + " |")
    out.append("")
out += ["## 64채널, 신호원 3개: 세기 순위별 오차와 '처음 찾은 것이 가장 센 것'인 비율", "", "| 방법 | 가장 센 것 | 두 번째 | 세 번째 | 처음 찾은 것이 가장 센 것 % |", "| --- | --- | --- | --- | --- |"]
for m in ORDER:
    out.append(f"| {m} | " + " | ".join(ci(ps("64채널", m, 3, f"순위{i}mm")) for i in (1, 2, 3)) + f" | {np.mean(ps('64채널', m, 3, '첫번째가가장센것%')):.0f} |")
out += ["", "## 64채널, 신호원 2개: 세기 순위별 오차", "", "| 방법 | 가장 센 것 | 두 번째 |", "| --- | --- | --- |"]
for m in ORDER:
    out.append(f"| {m} | " + " | ".join(ci(ps("64채널", m, 2, f"순위{i}mm")) for i in (1, 2)) + " |")

# 분포
H = {}
for f in sorted(glob.glob(os.path.join(HERE, "ranked_hist_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        for r in rd:
            key = (r[0], r[1], r[2])
            H[key] = H.get(key, 0) + np.array(r[3:], float)
edges = np.array(hdr[3:], float)
out += ["", "## 64채널: 오차 분포 (전체 신호원 중 비율 %)", "", "| 방법 | 신호원 수 | 중앙값 mm | 10 mm 이내 | 20 mm 이내 | 40 mm 넘음 |", "| --- | --- | --- | --- | --- | --- |"]
for m in ORDER:
    for k in ("1", "2", "3"):
        h = H[("64채널", m, k)]
        c = np.cumsum(h) / h.sum()
        med = edges[np.searchsorted(c, 0.5)] + 1
        out.append(f"| {m} | {k} | {med:.0f} | {100 * h[edges < 10].sum() / h.sum():.0f} | {100 * h[edges < 20].sum() / h.sum():.0f} | {100 * h[edges >= 40].sum() / h.sum():.0f} |")
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "ranked_summary.md"), "w", encoding="utf-8").write(txt + "\n")

# 그림: 누적 분포 3칸
W, Hh = 760, 330
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"><rect width="{W}" height="{Hh}" fill="#fff"/>',
     '<text x="16" y="22" font-size="13" font-weight="600" fill="#222">64채널: 위치 오차가 이 값 이하인 신호원의 비율 (OASIS 처음 보는 4명, 시드 5개 합침)</text>']
for pi, k in enumerate(("1", "2", "3")):
    x0, x1, y0, y1 = 50 + pi * 240, 50 + pi * 240 + 205, 250, 50
    s.append(f'<text x="{(x0 + x1) / 2}" y="42" font-size="11" text-anchor="middle" fill="#333">신호원 {k}개</text>')
    for p in (0, 25, 50, 75, 100):
        y = y0 - (y0 - y1) * p / 100
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" stroke="#e6e6e6"/>')
        if pi == 0:
            s.append(f'<text x="{x0 - 5}" y="{y + 4}" font-size="10" text-anchor="end" fill="#666">{p}%</text>')
    for mm in (0, 20, 40, 60):
        x = x0 + (x1 - x0) * mm / 60
        s.append(f'<text x="{x}" y="{y0 + 14}" font-size="10" text-anchor="middle" fill="#666">{mm}</text>')
    s.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 28}" font-size="10" text-anchor="middle" fill="#555">위치 오차 mm</text>')
    for m, c in zip(ORDER, COL):
        h = H[("64채널", m, k)]
        cdf = np.cumsum(h) / h.sum()
        pts = [f"{x0:.1f},{y0:.1f}"] + [f"{x0 + (x1 - x0) * (e + 2) / 60:.1f},{y0 - (y0 - y1) * v:.1f}" for e, v in zip(edges, cdf) if e + 2 <= 60]
        s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2"/>')
for i, (m, c) in enumerate(zip(ORDER, COL)):
    x, y = 50 + (i % 3) * 240, 296 + (i // 3) * 16
    s.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" fill="{c}"/><text x="{x + 15}" y="{y}" font-size="10" fill="#333">{m}</text>')
s.append("</svg>")
open(os.path.join(HERE, "fig_ranked_dist.svg"), "w", encoding="utf-8").write("".join(s))
