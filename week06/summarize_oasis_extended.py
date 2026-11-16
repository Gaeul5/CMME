"""Week 6-4 결과 정리: extended_seed*.csv → 표 (평균 ± 95% 구간, 시드별 평균 기준)

실행: python week06/summarize_extended.py
"""
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "oasis_extended_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        rows += list(csv.DictReader(fh))
seeds = sorted({r["시드"] for r in rows})
n = len(seeds)
M = ["MNE (자기 머리)", "sLORETA (자기 머리)", "범용 1개: 마스크", "범용 1개: 어댑터"]
CONDS = ["표준 64채널", "표준 32채널", "표준 16채널", "처음 보는 배치 32채널", "처음 보는 배치 16채널", "처음 보는 배치 8채널"]


def per_seed(cond, method, group, col):
    return np.array([np.mean([float(r[col]) for r in rows if r["시드"] == s and r["조건"] == cond and r["방법"] == method
                              and r["신호원수"] == group]) for s in seeds])


def fmt(v, d=1):
    ci = T975[n] * v.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return f"{v.mean():.{d}f} ± {ci:.{d}f}"


out = [f"시드 {n}개, 처음 보는 OASIS 머리 4명. ± 는 시드 간 95% 구간", ""]
for group in ["신호원 1개", "신호원 2~3개", "전체"]:
    for col, title, d in (("위치오차mm", "위치 오차 mm (낮을수록 좋음)", 1), ("AUC", "AUC (높을수록 좋음)", 3), ("놓침%", "20 mm 안에 못 찾은 신호원 % (낮을수록 좋음)", 1)):
        out += [f"## {group} — {title}", "", "| 배치 | " + " | ".join(M) + " | 어댑터 − sLORETA |", "| --- " * (len(M) + 2) + "|"]
        for c in CONDS:
            v = [per_seed(c, m, group, col) for m in M]
            diff = v[3] - v[1]
            out.append(f"| {c} | " + " | ".join(fmt(x, d) for x in v) + f" | {fmt(diff, d)} |")
        out.append("")
txt = "\n".join(out)
print(txt)
with open(os.path.join(HERE, "oasis_extended_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
