"""Week 6-2 결과 정리: channel_curve_seed*.csv → 표(평균 ± 95% 신뢰구간, 시드 5개 기준)와 그림(SVG).

신뢰구간은 '시드별 평균'(각 시드에서 머리 6개와 배치를 평균한 값) 5개의 t 분포 구간이다.
시드가 바뀌면 학습 데이터, 학습된 모델, 평가 배치, 평가 샘플이 모두 바뀌므로 이 구간은 그 전체 흩어짐을 담는다.
머리 간 흩어짐은 따로 범위(최소~최대)로 적는다.

실행: python week06/summarize_curve.py
"""
import collections
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "channel_curve_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        rows += list(csv.DictReader(fh))
seeds = sorted({r["시드"] for r in rows})
n = len(seeds)
print(f"시드 {n}개: {seeds}")


def stat(exp, cond, method):
    per_seed, per_head = [], collections.defaultdict(list)
    for s in seeds:
        v = [float(r["평균mm"]) for r in rows if r["시드"] == s and r["실험"] == exp and r["조건"] == cond and r["방법"] == method]
        per_seed.append(np.mean(v))
    for r in rows:
        if r["실험"] == exp and r["조건"] == cond and r["방법"] == method:
            per_head[r["머리"]].append(float(r["평균mm"]))
    hm = [np.mean(v) for v in per_head.values()]
    m = np.mean(per_seed)
    ci = T975[n] * np.std(per_seed, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return m, ci, min(hm), max(hm), per_seed


def paired(exp, cond, a, b):
    """방법 a - 방법 b 의 시드별 차이: 평균, 95% 구간"""
    d = np.array(stat(exp, cond, a)[4]) - np.array(stat(exp, cond, b)[4])
    return d.mean(), T975[n] * d.std(ddof=1) / np.sqrt(n)


M = ["sLORETA (자기 머리)", "범용 1개: 마스크", "범용 1개: 어댑터"]
COUNTS = [4, 6, 8, 12, 16, 24, 32, 48, 53, 64]
out = ["## E6a 채널 수 곡선 (처음 보는 머리 6개, 처음 보는 무작위 배치, 평균 위치 오차 mm, 평균 ± 95% 구간)", "",
       "| 채널 수 | sLORETA (자기 머리) | 범용: 마스크 | 범용: 어댑터 | 어댑터 − sLORETA | 어댑터의 머리별 범위 |", "| --- | --- | --- | --- | --- | --- |"]
curve = {k: [] for k in M}
for c in COUNTS:
    s = [stat("E6a", f"{c}채널", k) for k in M]
    for k, v in zip(M, s):
        curve[k].append((v[0], v[1]))
    d, dci = paired("E6a", f"{c}채널", M[2], M[0])
    out.append(f"| {c} | {s[0][0]:.1f} ± {s[0][1]:.1f} | {s[1][0]:.1f} ± {s[1][1]:.1f} | {s[2][0]:.1f} ± {s[2][1]:.1f} | {d:+.1f} ± {dci:.1f} | {s[2][2]:.1f}~{s[2][3]:.1f} |")
out += ["", "## E6b 표준 배치: 범용 모델 1개 vs 채널 수별 전용 모델 (처음 보는 머리 6개)", "",
        "| 배치 | sLORETA (자기 머리) | 범용: 마스크 | 범용: 어댑터 | 채널 수별 전용 모델 | 어댑터 − 전용 (범용의 대가) |", "| --- | --- | --- | --- | --- | --- |"]
M2 = M + ["채널 수별 전용 모델"]
for name in ["64채널", "32채널", "16채널"]:
    s = [stat("E6b", name, k) for k in M2]
    d, dci = paired("E6b", name, M[2], M2[3])
    out.append(f"| {name} | " + " | ".join(f"{v[0]:.1f} ± {v[1]:.1f}" for v in s) + f" | {d:+.1f} ± {dci:.1f} |")
txt = "\n".join(out)
print(txt)
with open(os.path.join(HERE, "channel_curve_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")

# 그림: 채널 수(로그 눈금) 대 오차, 띠 = 95% 구간
FONT = 'font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"'
W, H, x0, y0, w, h, ymax = 600, 400, 56, 40, 510, 260, 70
px = lambda c: x0 + w * (np.log(c) - np.log(4)) / (np.log(64) - np.log(4))
py = lambda v: y0 + h - h * min(v, ymax) / ymax
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" {FONT}><rect width="{W}" height="{H}" fill="#fff"/>',
     f'<text x="16" y="22" font-size="13" font-weight="600" fill="#222">채널 수에 따른 위치 오차 (처음 보는 머리 6개, 처음 보는 배치, 시드 {n}개)</text>']
for v in range(0, ymax + 1, 10):
    o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{py(v):.0f}" y2="{py(v):.0f}" stroke="#e6e6e6"/><text x="{x0 - 6}" y="{py(v) + 4:.0f}" font-size="10" text-anchor="end" fill="#666">{v}</text>')
o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{py(52.5):.0f}" y2="{py(52.5):.0f}" stroke="#b33" stroke-dasharray="4 3"/>'
         f'<text x="{x0 + w}" y="{py(52.5) - 4:.0f}" font-size="10" text-anchor="end" fill="#b33">항상 중심을 답할 때 52.5</text>')
for c in COUNTS:
    if c != 53:
        o.append(f'<text x="{px(c):.0f}" y="{y0 + h + 15}" font-size="10" text-anchor="middle" fill="#333">{c}</text>')
o.append(f'<text x="{x0 + w / 2:.0f}" y="{y0 + h + 32}" font-size="10" text-anchor="middle" fill="#555">채널 수 (로그 눈금)</text>'
         f'<text x="{x0 - 40}" y="{y0 - 8}" font-size="10" fill="#555">평균 위치 오차 mm</text>')
COL = {M[0]: "#8a9099", M[1]: "#d9822b", M[2]: "#2b6cb0"}
for k in M:
    up = " ".join(f"{px(c):.1f},{py(m + ci):.1f}" for c, (m, ci) in zip(COUNTS, curve[k]))
    dn = " ".join(f"{px(c):.1f},{py(m - ci):.1f}" for c, (m, ci) in reversed(list(zip(COUNTS, curve[k]))))
    o.append(f'<polygon points="{up} {dn}" fill="{COL[k]}" fill-opacity="0.18"/>')
    o.append(f'<polyline points="{" ".join(f"{px(c):.1f},{py(m):.1f}" for c, (m, _) in zip(COUNTS, curve[k]))}" fill="none" stroke="{COL[k]}" stroke-width="2.2"/>')
    o.append("".join(f'<circle cx="{px(c):.1f}" cy="{py(m):.1f}" r="3" fill="{COL[k]}"/>' for c, (m, _) in zip(COUNTS, curve[k])))
for i, k in enumerate(M):
    o.append(f'<rect x="{x0 + 8}" y="{y0 + h + 44 + i * 16}" width="10" height="10" fill="{COL[k]}"/><text x="{x0 + 23}" y="{y0 + h + 53 + i * 16}" font-size="10" fill="#333">{k}</text>')
o.append(f'<text x="{x0 + 250}" y="{y0 + h + 53}" font-size="10" fill="#555">띠는 시드 간 95% 구간</text></svg>')
with open(os.path.join(HERE, "fig_channel_curve.svg"), "w", encoding="utf-8") as f:
    f.write("".join(o))
print("그림 저장: fig_channel_curve.svg")
