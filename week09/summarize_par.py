"""Parmigiani 검증 요약: 구간별 오차, 합친 방법 - 스캔 (피험자 부트스트랩 CI), 자극 조건별, LMI 겹치는 환자."""
import glob, numpy as np, pandas as pd

rng = np.random.default_rng(0)
ev = pd.read_csv("week09/par_eval.csv")
hy = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("week09/par_hybrid_seed*.csv"))])
seeds = sorted(hy["시드"].unique())
key = ["피험자", "run", "구간"]
# 시드 평균으로 합친 방법 오차 하나로
hyb = hy[hy.방법 == "합친 방법"].groupby(key)["오차mm"].mean().rename("합친")
# 짝 비교는 같은 특징(방향 자유 GOF)을 쓴 par_hybrid 안의 스캔과 한다
scan = ev[ev.방법 == "쌍극자 스캔"].set_index(key).drop(columns="오차mm")
scan["오차mm"] = hy[(hy.방법 == "쌍극자 스캔") & (hy.시드 == seeds[0])].set_index(key)["오차mm"]
slor = ev[ev.방법 == "sLORETA"].set_index(key)["오차mm"].rename("sLORETA")
d = scan.join(hyb).join(slor).rename(columns={"오차mm": "스캔"}).reset_index()
d["차이"] = d["합친"] - d["스캔"]


def boot(df, col, B=2000):
    # 피험자별 평균 차이 -> 피험자 평균, 피험자 단위 재표집 (중앙값은 동점 0이 많아 평균 사용)
    subs = df["피험자"].unique()
    per = df.groupby("피험자")[col].mean()
    bs = [per.loc[rng.choice(subs, len(subs))].mean() for _ in range(B)]
    return per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


out = [f"# Parmigiani 검증 (시드 {seeds}, 피험자 {d.피험자.nunique()}명, 세션 {d.groupby(['피험자','run']).ngroups})\n"]
out.append("## 구간별 오차 (중앙값 mm, 20mm 이내 %)\n")
out.append("| 구간 | 스캔 | 합친 방법 | sLORETA | 스캔 20mm이내 | 합친 20mm이내 | 합친-스캔 평균 mm (피험자 부트스트랩 95% CI) | 합친이 나은 피험자 |")
out.append("|---|---|---|---|---|---|---|---|")
for g, x in d.groupby("구간"):
    m, lo, hi = boot(x, "차이")
    win = (x.groupby("피험자")["차이"].mean() < 0).sum()
    out.append(f"| {g} | {x.스캔.median():.1f} | {x.합친.median():.1f} | {x.sLORETA.median():.1f} | "
               f"{(x.스캔<20).mean()*100:.0f} | {(x.합친<20).mean()*100:.0f} | {m:+.1f} ({lo:+.1f}~{hi:+.1f}) | {win}/{x.피험자.nunique()} |")

r = d[d.구간 == "초기 반응 15~40"]
out.append("\n## 초기 반응 15~40 ms, 조건별 (중앙값 mm)\n")
for col in ["세기", "펄스", "각도", "회백질백질"]:
    t = r.groupby(col).agg(n=("스캔", "size"), 스캔=("스캔", "median"), 합친=("합친", "median"), 평균차이=("차이", "mean")).round(1)
    out.append(f"**{col}**\n\n" + t.to_markdown() + "\n")

out.append("## Localize-MI와 같은 환자 (15~40 ms)\n")
t = r[r.LocalizeMI같은환자.notna()].groupby(["LocalizeMI같은환자", "피험자"]).agg(
    세션=("스캔", "size"), 스캔=("스캔", "median"), 합친=("합친", "median")).round(1)
out.append(t.to_markdown() + "\n")
txt = "\n".join(out)
open("week09/par_summary.md", "w").write(txt)
print(txt)
