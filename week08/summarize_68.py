"""Week 8-6 (오차의 원인), 8-2 경로 1 재평가, 8-7 (채널 줄이기) 정리 → week08/w86_summary.md"""
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
out = []


def boot(per, a, b, n=5000):
    d = (per[a] - per[b]).dropna()
    subs = sorted(d.index.get_level_values(0).unique())
    by = {s: d.xs(s, level=0).values for s in subs}
    bs = [np.mean(np.concatenate([by[s] for s in rng.choice(subs, len(subs))])) for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(np.mean(by[s]) < 0 for s in subs)


# ---- 8-6
d = pd.read_csv(os.path.join(HERE, "lmi_gap.csv"))
d["v"] = pd.to_numeric(d["값(오차중앙값mm 또는 비율)"])
d["ex"] = pd.to_numeric(d["진짜쌍극자_설명비율"], errors="coerce")
order = ["A 피질 법선 + 흰 잡음", "B 전극 축 방향 + 흰 잡음", "C 접점 중점 + 흰 잡음", "D 접점 중점 + 실제 잡음",
         "E 머리 모델 불일치(두개골 0.0045) + 실제 잡음", "E 머리 모델 불일치(두개골 0.009) + 실제 잡음",
         "F 전극 위치 오차 5 mm + 실제 잡음", "F 전극 위치 오차 10 mm + 실제 잡음", "실제 기록"]
out += ["# Week 8-6: 시뮬레이션과 실제 기록 사이 오차는 어디서 오나 (쌍극자 스캔, run 61개, 조건마다 잡음 30번)", "",
        "| 조건 | 오차 중앙값 mm | 오차 평균 mm | 진짜 쌍극자가 설명하는 비율 (중앙값) |", "| --- | --- | --- | --- |"]
for k in order:
    x = d[d["조건"] == k]
    out.append(f"| {k} | {x['v'].median():.1f} | {x['v'].mean():.1f} | {x['ex'].median():.3f} |")
r = d[d["조건"] == "실제 기록"]
best = d[d["조건"].str.startswith("설명 비율: 가장")]
out += ["", f"실제 기록: 피질 위에서 가장 잘 맞는 쌍극자가 설명하는 비율 중앙값 {best['v'].median():.3f}, 진짜 쌍극자 {r['ex'].median():.3f}. "
        f"진짜 쌍극자 설명 비율이 0.9 미만인 run {100 * (r['ex'] < 0.9).mean():.0f}%.",
        f"실제 오차와 진짜 쌍극자 설명 비율의 순위 상관 {r[['v', 'ex']].corr(method='spearman').iloc[0, 1]:+.2f}. SNR 중앙값 {r['SNR_dB'].median():.0f} dB (범위 {r['SNR_dB'].min():.0f}~{r['SNR_dB'].max():.0f})", ""]

# ---- 경로 1 재평가
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(HERE, "lmi_routes_seed[0-9]_r1.csv")))])
p1 = R.groupby(["피험자", "run", "방법"])["오차mm"].mean().unstack()
B = pd.read_csv(os.path.join(HERE, "lmi_baseline.csv"))
b = B[(B["잡음"] == "없음") & (B["신호"] == "한 시점") & (B["방법"] == "쌍극자 스캔")].set_index(["피험자", "run"])["오차A_중점mm"]
b.index = pd.MultiIndex.from_arrays([b.index.get_level_values(0), b.index.get_level_values(1).astype(int)])
p1["쌍극자 스캔 (데이터 머리 모델)"] = b
out += ["# 8-2 경로 1 재평가 (전극 위치를 바로잡은 변형 머리, 시드 3개)", "",
        "주의: lmi_routes_seed[0-2].csv와 lmi_summary.md의 경로 1 숫자(17.4 mm)는 틀린 전극 위치(모든 피험자에 sub-01 전극 파일)로 만든 변형 머리라 무효. 여기 숫자로 대체", "", "| 방법 | 중앙값 mm | 평균 mm | 10 mm 이내 | 20 mm 이내 |", "| --- | --- | --- | --- | --- |"]
for m in ["쌍극자 스캔 (데이터 머리 모델)", "쌍극자 스캔 (변형 머리)", "sLORETA (변형 머리)", "경로 1: 어댑터 (변형 머리)"]:
    v = p1[m]
    out.append(f"| {m} | {v.median():.1f} | {v.mean():.1f} | {100 * (v < 10).mean():.0f}% | {100 * (v < 20).mean():.0f}% |")
out += ["", "| 비교 (평균 차이, 피험자 부트스트랩 95% 구간) | 차이 mm | 95% 구간 | 앞쪽이 나은 피험자 |", "| --- | --- | --- | --- |"]
for a, c in (("쌍극자 스캔 (변형 머리)", "쌍극자 스캔 (데이터 머리 모델)"), ("경로 1: 어댑터 (변형 머리)", "쌍극자 스캔 (변형 머리)")):
    e, lo, hi, k = boot(p1, a, c)
    out.append(f"| {a} − {c} | {e:+.1f} | {lo:+.1f} ~ {hi:+.1f} | {k} / 7 |")
out.append("")

# ---- 8-7
F = sorted(glob.glob(os.path.join(HERE, "lmi_channels_seed[0-9].csv"))) + sorted(glob.glob(os.path.join(HERE, "lmi_channels_12_8_seed[0-9].csv")))
if F:
    Ch = pd.concat([pd.read_csv(f) for f in F])
    Ch["채널수"] = Ch["채널수"].astype(str)
    out += [f"# Week 8-7: 실제 기록에서 채널 줄이기 (시드 {Ch['시드'].nunique()}개, run 61개)", "",
            "오차 중앙값 mm (무작위 배치 3개와 고르게 고른 배치를 모두 합친 값. 학습 모델은 시드 평균)", "",
            "| 채널 수 | 배치 | sLORETA | 쌍극자 스캔 | 합친 방법 | 합친 − 스캔 (평균, 95% 구간) | 합친 방법이 나은 피험자 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in ("전체", "128", "64", "32", "16", "12", "8"):
        if not (Ch["채널수"] == n).any():
            continue
        for kind in (("전체",) if n == "전체" else ("고르게", "무작위")):
            x = Ch[(Ch["채널수"] == n) & (Ch["배치"].str.startswith(kind))]
            per = x.groupby(["피험자", "run", "배치", "방법"])["오차mm"].mean().unstack()
            per.index = per.index.droplevel(2) if False else per.index
            pr = per.groupby(level=[0, 1]).mean()
            e, lo, hi, k = boot(pr, "합친 방법", "쌍극자 스캔")
            out.append(f"| {n} | {kind} | {pr['sLORETA 법선'].median():.1f} | {pr['쌍극자 스캔'].median():.1f} | {pr['합친 방법'].median():.1f} | {e:+.1f} ({lo:+.1f} ~ {hi:+.1f}) | {k} / 7 |")
out += ["", "참고: 항상 머리 중심 근처 source를 답할 때 오차 중앙값 55.7 mm (평균 54.1)"]
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "w86_summary.md"), "w", encoding="utf-8").write(txt + "\n")
