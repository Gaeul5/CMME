"""Week 8-4, 8-5 결과 정리: 정답 없는 적응 (Localize-MI 실제 기록, 피험자 7명, run 61개)

출력: adapt_summary.md, fig_adapt.svg
불확실성: 피험자 단위 부트스트랩 5,000번 (run별 차이의 중앙값). 학습 모델은 시드 평균
"""
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "lmi_adapt2_*_seed[0-9].csv")))])
seeds = {t: sorted(R[R["방법"].str.contains(t)]["시드"].unique()) for t in ("기본 학습", "학습 때 섞기")}
per = R.groupby(["피험자", "run", "mA", "방법"])["오차mm"].mean().unstack()
subs = sorted(per.index.get_level_values(0).unique())
rng = np.random.default_rng(0)


def boot(a, b, n=5000):
    d = per[a] - per[b]
    by = {s: d.xs(s, level=0).values for s in subs}
    bs = [np.median(np.concatenate([by[s] for s in rng.choice(subs, len(subs))])) for _ in range(n)]
    return np.median(d.values), np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(np.median(by[s]) < 0 for s in subs)


SEL = {}
for s in subs:
    d = pd.read_csv(os.path.join(HERE, f"lmi_select_{s}.csv"))
    c = d[d["단계"] == "전도도"].set_index("전도도")
    f = d[d["단계"].str.startswith("최종")].iloc[0]
    SEL[s] = dict(cv=c["교차검증_예측오차"].idxmin(), ins=c["같은채널_남는비율"].idxmin(), oracle=c["스캔오차중앙값mm"].idxmin(),
                  base=c.loc[0.006, "스캔오차중앙값mm"], c_cv=c.loc[c["교차검증_예측오차"].idxmin(), "스캔오차중앙값mm"],
                  c_ins=c.loc[c["같은채널_남는비율"].idxmin(), "스캔오차중앙값mm"], best=c["스캔오차중앙값mm"].min(),
                  shift=(f["x이동mm"], f["y이동mm"], f["z이동mm"]), c_shift=f["스캔오차중앙값mm"])
out = ["# Week 8-4, 8-5: 정답 없는 적응 (Localize-MI 실제 기록, 피험자 7명, run 61개)", "",
       f"학습 모델 시드: 기본 학습 {len(seeds['기본 학습'])}개, 학습 때 섞기 {len(seeds['학습 때 섞기'])}개. 오차 = 추정 위치 → 자극 접점 쌍의 중점", "",
       "## 1. 두개골 전도도를 정답 없이 고르기 (쌍극자 스캔, 피험자별 오차 중앙값 mm)", "",
       "| 피험자 | 정답으로 고른 전도도 | 같은 채널 기준 | 교차 검증 기준 | 기본 0.006 | 같은 채널로 고름 | 교차 검증으로 고름 | 상한 | 교차 검증 + 전극 이동 |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for s in subs:
    x = SEL[s]
    out.append(f"| {s} | {x['oracle']} | {x['ins']} | {x['cv']} | {x['base']:.1f} | {x['c_ins']:.1f} | {x['c_cv']:.1f} | {x['best']:.1f} | {x['c_shift']:.1f} ({', '.join(f'{v:+.0f}' for v in x['shift'])}) |")
hit = lambda k: sum(SEL[s][k] == SEL[s]["oracle"] for s in subs)
out += ["", f"정답과 같은 전도도를 고른 피험자: 같은 채널 기준 {hit('ins')} / 7, 교차 검증 기준 {hit('cv')} / 7", ""]
M = [m for m in per.columns]
order = sorted(M, key=lambda m: (not m.startswith("쌍극자"), m))
out += ["## 2. run 61개 전체", "", "| 방법 | 중앙값 mm | 평균 mm | 10 mm 이내 | 20 mm 이내 |", "| --- | --- | --- | --- | --- |"]
for m in order:
    v = per[m]
    out.append(f"| {m} | {v.median():.1f} | {v.mean():.1f} | {100 * (v < 10).mean():.0f}% | {100 * (v < 20).mean():.0f}% |")
ref = "쌍극자 스캔 · 기본 전도도"
out += ["", f"## 3. 기준({ref})과의 차이 (run별 차이의 중앙값, 피험자 부트스트랩 95% 구간, 음수면 나음)", "",
        "| 방법 | 차이 mm | 95% 구간 | 나은 피험자 수 |", "| --- | --- | --- | --- |"]
for m in order:
    if m == ref:
        continue
    e, lo, hi, k = boot(m, ref)
    out.append(f"| {m} | {e:+.1f} | {lo:+.1f} ~ {hi:+.1f} | {k} / 7 |")
base = "합친 방법 (기본 학습) · 기본 전도도"
out += ["", f"## 4. 적응 효과 (각 방법 − {base})", "", "| 방법 | 차이 mm | 95% 구간 | 나은 피험자 수 |", "| --- | --- | --- | --- |"]
for m in order:
    if m.startswith("합친") and m != base:
        e, lo, hi, k = boot(m, base)
        out.append(f"| {m} | {e:+.1f} | {lo:+.1f} ~ {hi:+.1f} | {k} / 7 |")
out += ["", "## 5. 피험자별 중앙값 (mm)", "", "| 방법 | " + " | ".join(subs) + " |", "| --- | " + " | ".join(["---"] * len(subs)) + " |"]
for m in order:
    out.append(f"| {m} | " + " | ".join(f"{np.median(per[m].xs(s, level=0)):.1f}" for s in subs) + " |")
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "adapt_summary.md"), "w", encoding="utf-8").write(txt + "\n")
