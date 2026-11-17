"""Week 7-2 결과 정리: 정답 없는 신호로 추론 단계에서 적응하기

출력: tta_summary.md, fig_tta.svg
실행: python week07/summarize_tta.py
"""
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
INPUTS = ["센서만", "어댑터 템플릿 머리", "어댑터 자기 머리"]
METHODS = ["입력 통계 맞추기", "특징 정렬 (SSA)", "물리 자기학습", "물리 자기학습 + 물리 보정", "정답으로 미세조정 (상한)"]


def ci(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f} ± {T975[len(v)] * v.std(ddof=1) / np.sqrt(len(v)):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


R = []
for f in sorted(glob.glob(os.path.join(HERE, "tta_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        R += list(csv.DictReader(fh))
seeds = sorted({r["시드"] for r in R})
NS = sorted({int(r["정답 없는 기록 수"]) for r in R} - {0})


def ps(inp, meth, n):
    return np.array([np.mean([float(r["평균mm"]) for r in R if r["시드"] == s and r["입력"] == inp and r["적응 방법"] == meth and r["정답 없는 기록 수"] == str(n)]) for s in seeds])


out = [f"# Week 7-2: 정답 없는 신호로 적응하기 (변형 머리 12개로 학습 → OASIS 10명에서 평가, 64채널, 쌍극자 1개, 시드 {len(seeds)}개)", "",
       f"학습 없는 방법: sLORETA {ci(ps('-', 'sLORETA (자기 머리)', 0))} mm, 쌍극자 스캔 {ci(ps('-', '쌍극자 스캔 (자기 머리)', 0))} mm", "",
       "## 평균 위치 오차 mm ± 시드 간 95% 구간", "", "| 적응 방법 | 정답 없는 기록 수 | " + " | ".join(INPUTS) + " |", "| --- | --- | --- | --- | --- |"]
out.append("| 없음 | - | " + " | ".join(ci(ps(i, "없음", 0)) for i in INPUTS) + " |")
out.append("| 물리 보정 (신호마다) | 0 | " + " | ".join(ci(ps(i, "물리 보정", 0)) for i in INPUTS) + " |")
for m in METHODS:
    for n in NS:
        out.append(f"| {m} | {n} | " + " | ".join(ci(ps(i, m, n)) for i in INPUTS) + " |")
out += ["", "## 적응 전과의 차이 (적응 − 없음, 음수면 좋아진 것)", "", "| 적응 방법 | 정답 없는 기록 수 | " + " | ".join(INPUTS) + " |", "| --- | --- | --- | --- | --- |"]
for m in METHODS:
    for n in NS:
        out.append(f"| {m} | {n} | " + " | ".join(ci(ps(i, m, n) - ps(i, "없음", 0)) for i in INPUTS) + " |")
# 머리별: 적응으로 나빠진 사람 수
out += ["", f"## 사람별로 봤을 때 나빠진 경우 (기록 {NS[-1]}개, 사람 × 시드 {len(seeds) * 10}건 중 적응 후 오차가 더 커진 비율 %)", "", "| 적응 방법 | " + " | ".join(INPUTS) + " |", "| --- | --- | --- | --- |"]
for m in METHODS[:3]:
    cells = []
    for i in INPUTS:
        base = {(r["시드"], r["머리"]): float(r["평균mm"]) for r in R if r["입력"] == i and r["적응 방법"] == "없음"}
        ad = {(r["시드"], r["머리"]): float(r["평균mm"]) for r in R if r["입력"] == i and r["적응 방법"] == m and r["정답 없는 기록 수"] == str(NS[-1])}
        cells.append(f"{100 * np.mean([ad[k] > base[k] for k in base]):.0f}")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "tta_summary.md"), "w", encoding="utf-8").write(txt + "\n")

# 그림: 어댑터 자기 머리, 기록 수별 막대
inp = "어댑터 자기 머리"
bars = [("적응 없음", ps(inp, "없음", 0).mean(), "#9aa0a6")]
for m, c in zip(["입력 통계 맞추기", "특징 정렬 (SSA)", "물리 자기학습", "정답으로 미세조정 (상한)"], ["#c9b458", "#7fb2e5", "#2b6cb0", "#444b54"]):
    for n in NS:
        bars.append((f"{m.split(' (')[0]} {n}", ps(inp, m, n).mean(), c))
bars += [("물리 보정", ps(inp, "물리 보정", 0).mean(), "#d9822b"), ("쌍극자 스캔", ps("-", "쌍극자 스캔 (자기 머리)", 0).mean(), "#b33")]
W, H, x0, y0, top = 760, 360, 50, 250, 50
ymax = 14
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"><rect width="{W}" height="{H}" fill="#fff"/>',
     f'<text x="16" y="22" font-size="13" font-weight="600" fill="#222">정답 없는 기록으로 적응했을 때 위치 오차 ({inp}, 변형 머리로 학습 → OASIS 10명, 시드 {len(seeds)}개)</text>',
     '<text x="16" y="40" font-size="10" fill="#555">평균 위치 오차 mm. 막대 아래 숫자는 적응에 쓴 정답 없는 기록 수</text>']
for v in range(0, ymax + 1, 2):
    y = y0 - (y0 - top) * v / ymax
    s.append(f'<line x1="{x0}" x2="{W - 20}" y1="{y}" y2="{y}" stroke="#e6e6e6"/><text x="{x0 - 5}" y="{y + 4}" font-size="10" text-anchor="end" fill="#666">{v}</text>')
bw = (W - 20 - x0) / len(bars)
for i, (lab, v, c) in enumerate(bars):
    x = x0 + i * bw + 5
    h = (y0 - top) * v / ymax
    s.append(f'<rect x="{x:.1f}" y="{y0 - h:.1f}" width="{bw - 10:.1f}" height="{h:.1f}" fill="{c}"/><text x="{x + (bw - 10) / 2:.1f}" y="{y0 - h - 4:.1f}" font-size="9" text-anchor="middle" fill="#333">{v:.1f}</text>')
    parts = lab.rsplit(" ", 1)
    num = parts[1] if parts[-1].isdigit() else ""
    s.append(f'<text x="{x + (bw - 10) / 2:.1f}" y="{y0 + 13}" font-size="9" text-anchor="middle" fill="#333">{num}</text>')
leg = [("적응 없음", "#9aa0a6"), ("입력 통계 맞추기", "#c9b458"), ("특징 정렬 (SSA)", "#7fb2e5"), ("물리 자기학습", "#2b6cb0"), ("정답으로 미세조정 (상한)", "#444b54"), ("물리 보정 (신호마다)", "#d9822b"), ("쌍극자 스캔 (학습 없음)", "#b33")]
for i, (l, c) in enumerate(leg):
    x, y = 50 + (i % 4) * 175, 290 + (i // 4) * 18
    s.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" fill="{c}"/><text x="{x + 15}" y="{y}" font-size="10" fill="#333">{l}</text>')
s.append("</svg>")
open(os.path.join(HERE, "fig_tta.svg"), "w", encoding="utf-8").write("".join(s))
