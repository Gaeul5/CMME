"""Week 6-6 결과 정리: oasis_ablation_seed*.csv → 표 (평균 ± 95% 구간, 시드별 평균 기준)

처음 보는 배치는 같은 채널 수의 배치 2개를 묶어서 평균한다.
실행: python week06/summarize_ablation.py
"""
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "oasis_ablation_seed[0-9].csv"))):
    with open(f, encoding="utf-8-sig") as fh:
        rows += list(csv.DictReader(fh))
for r in rows:
    r["묶음"] = r["조건"].split(" #")[0]
seeds = sorted({r["시드"] for r in rows})
n = len(seeds)
GROUPS = ["표준 64채널", "표준 32채널", "표준 16채널", "처음 보는 배치 32채널", "처음 보는 배치 16채널"]


def ps(g, m):
    return np.array([np.mean([float(r["평균mm"]) for r in rows if r["시드"] == s and r["묶음"] == g and r["방법"] == m]) for s in seeds])


def fmt(v):
    return f"{v.mean():.1f} ± {T975[n] * v.std(ddof=1) / np.sqrt(n):.1f}" if n > 1 else f"{v.mean():.1f}"


def table(title, methods, base=None):
    out = [f"## {title}", "", "| 배치 | " + " | ".join(methods) + (f" | {methods[base[0]].split(' ')[0]} − {methods[base[1]].split(' ')[0]} |" if base else " |"),
           "| --- " * (len(methods) + 1 + (1 if base else 0)) + "|"]
    for g in GROUPS:
        v = [ps(g, m) for m in methods]
        line = f"| {g} | " + " | ".join(fmt(x) for x in v)
        if base:
            line += f" | {fmt(v[base[0]] - v[base[1]])}"
        out.append(line + " |")
    return out + [""]


txt = [f"시드 {n}개, 처음 보는 OASIS 머리 4명 (시드마다 다른 4명), 평균 위치 오차 mm (± 시드 간 95% 구간). 항상 그 머리의 중심을 답하면 OASIS 10명 평균 49.4 mm (46.9~55.3)", ""]
txt += table("1. 학습 조건: 무엇을 거쳐 넣는가", ["A 자기 머리 리드필드", "B 템플릿 머리 리드필드", "C 무작위 가짜 리드필드", "D 마스크 (변환 없음)"], base=(2, 0))
txt += table("2. 머리 물리의 기여: 자기 머리 vs 템플릿 머리", ["A 자기 머리 리드필드", "B 템플릿 머리 리드필드"], base=(1, 0))
txt += table("3. 시험 때 틀린 물리 (A 모델, 재학습 없음)", ["A 자기 머리 리드필드", "A1 시험 때 다른 사람 리드필드", "A2 시험 때 전극 좌우 뒤바뀜", "A3 시험 때 전극 순서 뒤섞임"])
txt += table("4. 같은 실수를 sLORETA에", ["sLORETA 정상", "sLORETA 다른 사람 리드필드", "sLORETA 전극 좌우 뒤바뀜", "sLORETA 전극 순서 뒤섞임"])
txt = "\n".join(txt)
print(txt)
with open(os.path.join(HERE, "oasis_ablation_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
