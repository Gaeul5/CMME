"""Week 6-7 결과 정리 (OASIS 머리): 학습 머리 종류·수(transfer), 전도도 적응·전극 오차(adapt), 머리별·스캐너별 분석

출력: oasis_more_summary.md
실행: python week06/summarize_oasis_more.py
"""
import collections
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def read(pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        with open(f, encoding="utf-8-sig") as fh:
            r = list(csv.DictReader(fh))
        for x in r:
            x["_file"] = os.path.basename(f)
        rows += r
    return rows


def ci(v):
    v = np.asarray(v, float)
    return f"{v.mean():.1f} ± {T975[len(v)] * v.std(ddof=1) / np.sqrt(len(v)):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


out = []

# ---- 1. 학습 머리 종류와 수
R = read("oasis_transfer_seed[0-9].csv")
if R:
    seeds = sorted({r["시드"] for r in R})
    def ps(method, train, test):
        return [np.mean([float(r["평균mm"]) for r in R if r["시드"] == s and r["방법"] == method and r["학습 머리"] == train and r["평가 머리 묶음"] == test]) for s in seeds]
    out += [f"## 1. 학습 머리 종류와 수 (64채널, 시드 {len(seeds)}개, 평균 위치 오차 mm ± 시드 간 95% 구간)", ""]
    for test in ["OASIS 평가 4명", "변형 머리 평가 6개"]:
        out += [f"### 평가: {test} (sLORETA 자기 머리 {ci(ps('sLORETA (자기 머리)', '-', test))})", "",
                "| 학습 머리 | 센서만 | 어댑터 (자기 머리) | 어댑터 − 센서만 |", "| --- | --- | --- | --- |"]
        for tr in ["변형 머리 12개", "OASIS 1명", "OASIS 3명", "OASIS 6명", "변형 12 + OASIS 6"]:
            a, b = ps("센서만", tr, test), ps("어댑터 (자기 머리)", tr, test)
            out.append(f"| {tr} | {ci(a)} | {ci(b)} | {ci(np.array(b) - np.array(a))} |")
        out.append("")

# ---- 2. 전도도 적응, 전극 위치 오차
R = read("oasis_adapt_seed[0-9].csv")
if R:
    files = sorted({r["_file"] for r in R})
    col = "평균mm(선택실험은 정확%)"
    col2 = "중앙값mm(선택실험은 한칸이내%)"
    def g(exp, cond, method, c=col):
        return [float(r[c]) for f in files for r in R if r["_file"] == f and r["실험"] == exp and r["조건"] == cond and r["방법"] == method]
    out += [f"## 2. 라벨 없는 전도도 적응 (평가 4명: OAS30004, 30006, 30008, 30010. 시드 {len(files)}개)", "",
            "| 사건 수 | 진짜에 가장 가까운 후보를 고른 비율 % | 한 칸 이내 % |", "| --- | --- | --- |"]
    for n in (10, 50, 200):
        c = f"선택 정확도 / 사건 {n}개"
        out.append(f"| {n} | {np.mean(g('E4', c, '정확히 맞힘 %')):.0f} | {np.mean(g('E4', c, '정확히 맞힘 %', col2)):.0f} |")
    cond = "평가 머리 4명, 사건 200개로 적응"
    out += ["", "| 방법 | 기본 0.006 | 적응 | 상한 | 적응 − 기본 |", "| --- | --- | --- | --- | --- |"]
    for m in ("sLORETA", "어댑터"):
        a, b, c = g("E4", cond, f"{m} 기본(0.006)"), g("E4", cond, f"{m} 적응"), g("E4", cond, f"{m} 상한(진짜에 가장 가까운 후보)")
        out.append(f"| {m} | {ci(a)} | {ci(b)} | {ci(c)} | {ci(np.array(b) - np.array(a))} |")
    out.append(f"| MLP 센서만 | {ci(g('E4', cond, 'MLP 센서만'))} | (적응 불가) | | |")
    out += ["", "## 3. 전극 위치 오차 (모델은 표준 위치, 평균 위치 오차 mm)", "",
            "| 전극 오차 | sLORETA (자기 머리) | MLP 센서만 | 어댑터 (자기 머리) |", "| --- | --- | --- | --- |"]
    for j in (0, 5, 10, 15):
        c = f"전극 오차 {j} mm"
        out.append(f"| {j} mm | {ci(g('E5', c, 'sLORETA (자기 머리)'))} | {ci(g('E5', c, 'MLP 센서만'))} | {ci(g('E5', c, '어댑터 (자기 머리)'))} |")
    out.append("")

# ---- 4. 머리별, 스캐너별 (6-6 결과에서, 평가에 들어간 시드만 평균)
R = read("oasis_ablation_seed[0-9].csv")
if R:
    size = {}
    for f in sorted(glob.glob(os.path.join(HERE, "cache_oasis", "OAS*.npz"))):
        size[os.path.basename(f)[:8]] = float(np.load(f)["scale"])
    OTHER = {"OAS30005", "OAS30006", "OAS30010"}       # 헤더가 다른 3명 (다른 스캐너로 보임)
    def per_head(method):
        d = collections.defaultdict(list)
        for r in R:
            if r["방법"] == method and r["조건"] == "표준 64채널":
                d[r["머리"]].append(float(r["평균mm"]))
        return {k: np.mean(v) for k, v in d.items()}
    A, B, S = per_head("A 자기 머리 리드필드"), per_head("B 템플릿 머리 리드필드"), per_head("sLORETA 정상")
    out += ["## 4. 머리별 결과 (6-6, 표준 64채널, 그 머리가 평가에 들어간 시드의 평균)", "",
            "| 머리 | 두개골 안쪽 크기 mm | 헤더 | 평가 횟수 | 어댑터 자기 머리 | 어댑터 템플릿 머리 | sLORETA |", "| --- | --- | --- | --- | --- | --- | --- |"]
    cnt = collections.Counter(r["머리"] for r in R if r["방법"] == "A 자기 머리 리드필드" and r["조건"] == "표준 64채널")
    for h in sorted(A):
        out.append(f"| {h} | {size[h]:.0f} | {'다름' if h in OTHER else '다수'} | {cnt[h]} | {A[h]:.1f} | {B[h]:.1f} | {S[h]:.1f} |")
    hs = sorted(A)
    x = np.array([size[h] for h in hs])
    out += ["", f"머리 크기와 오차의 상관 (머리 {len(hs)}명): 어댑터 자기 머리 r = {np.corrcoef(x, [A[h] for h in hs])[0, 1]:+.2f}, "
            f"템플릿 머리 r = {np.corrcoef(x, [B[h] for h in hs])[0, 1]:+.2f}, sLORETA r = {np.corrcoef(x, [S[h] for h in hs])[0, 1]:+.2f}",
            f"템플릿 − 자기 머리 (머리별): " + ", ".join(f"{h[5:]} {B[h] - A[h]:+.1f}" for h in hs),
            f"헤더가 다른 3명 평균: 어댑터 {np.mean([A[h] for h in hs if h in OTHER]):.1f}, sLORETA {np.mean([S[h] for h in hs if h in OTHER]):.1f} / "
            f"나머지: 어댑터 {np.mean([A[h] for h in hs if h not in OTHER]):.1f}, sLORETA {np.mean([S[h] for h in hs if h not in OTHER]):.1f}", ""]

txt = "\n".join(out)
print(txt)
with open(os.path.join(HERE, "oasis_more_summary.md"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
