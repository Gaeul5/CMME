"""Week 7-4 결과 정리: 쌍극자 스캔(과 MUSIC)을 6-7a(채널 수), 6-7c(넓은 신호원), 7-1(여러 신호원)에 기준으로 넣었을 때

원래 실행과 평가 샘플이 똑같으므로(sLORETA 값이 일치하는지 여기서 확인) 방법 간 차이를 시드별로 짝지어 본다.
출력: scan_summary.md, fig_scan_curve.svg
실행: python week07/summarize_scan.py
"""
import csv
import glob
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W6 = os.path.join(HERE, "..", "week06")
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def read(d, pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, pat))):
        with open(f, encoding="utf-8-sig") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def ci(v, nd=1):
    v = np.asarray(v, float)
    return f"{v.mean():.{nd}f} ± {T975[len(v)] * v.std(ddof=1) / np.sqrt(len(v)):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"


def check(old, new, keys, val, name):
    a = {tuple(r[k] for k in keys): r[val] for r in old if r["방법"].startswith("sLORETA")}
    b = {tuple(r[k] for k in keys): r[val] for r in new if r["방법"].startswith("sLORETA")}
    same = sum(a[k] == b[k] for k in b if k in a)
    return f"{name}: sLORETA 행 {len(b)}개 중 {same}개가 원래 실행과 같은 값"


out = ["# Week 7-4: 쌍극자 스캔을 비교 기준에 넣기 (OASIS 처음 보는 4명, 시드 5개, ± 시드 간 95% 구간)", ""]
chk = []

# ---------- 1. 채널 수 곡선
A, S = read(W6, "oasis_curve_seed[0-9].csv"), read(HERE, "scan_curve_seed[0-9].csv")
if S:
    seeds = sorted({r["시드"] for r in S})
    chk.append(check(A, S, ["시드", "실험", "조건", "방법", "머리"], "평균mm", "채널 수"))   # 같은 조건에 배치가 여러 개면 마지막 값끼리 비교됨

    def ps(R, exp, cond, m):
        return np.array([np.mean([float(r["평균mm"]) for r in R if r["시드"] == s and r["실험"] == exp and r["조건"] == cond and r["방법"] == m]) for s in seeds])
    conds = [f"{c}채널" for c in (4, 6, 8, 12, 16, 24, 32, 48, 53, 64)]
    out += ["## 1. 채널 수 곡선 (쌍극자 1개, 처음 보는 무작위 배치, 평균 위치 오차 mm)", "",
            "| 채널 수 | sLORETA | 쌍극자 스캔 | 범용 마스크 | 범용 어댑터 | 어댑터 − 스캔 |", "| --- | --- | --- | --- | --- | --- |"]
    curve = {}
    for c in conds:
        sl, sc, mk, ad = ps(A, "E6a", c, "sLORETA (자기 머리)"), ps(S, "E6a", c, "쌍극자 스캔 (자기 머리)"), ps(A, "E6a", c, "범용 1개: 마스크"), ps(A, "E6a", c, "범용 1개: 어댑터")
        curve[c] = (sl, sc, mk, ad)
        out.append(f"| {c[:-2]} | {ci(sl)} | {ci(sc)} | {ci(mk)} | {ci(ad)} | {ci(ad - sc)} |")
    out += ["", "표준 배치 (E6b)", "", "| 배치 | sLORETA | 쌍극자 스캔 | 범용 어댑터 | 채널 수별 전용 모델 | 어댑터 − 스캔 |", "| --- | --- | --- | --- | --- | --- |"]
    for c in ("64채널", "32채널", "16채널"):
        sc, ad = ps(S, "E6b", c, "쌍극자 스캔 (자기 머리)"), ps(A, "E6b", c, "범용 1개: 어댑터")
        out.append(f"| {c} | {ci(ps(A, 'E6b', c, 'sLORETA (자기 머리)'))} | {ci(sc)} | {ci(ad)} | {ci(ps(A, 'E6b', c, '채널 수별 전용 모델'))} | {ci(ad - sc)} |")
    out.append("")
    # 그림
    Wd, Hh, x0, x1, y0, y1 = 600, 380, 56, 566, 300, 40
    xs = lambda c: x0 + (x1 - x0) * (np.log(c) - np.log(4)) / (np.log(64) - np.log(4))
    ys = lambda v: y0 - (y0 - y1) * v / 70
    g = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Hh}" width="{Wd}" height="{Hh}" font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"><rect width="{Wd}" height="{Hh}" fill="#fff"/>',
         '<text x="16" y="22" font-size="13" font-weight="600" fill="#222">채널 수에 따른 위치 오차: 쌍극자 스캔을 넣었을 때 (OASIS 처음 보는 4명, 시드 5개)</text>',
         '<text x="16" y="34" font-size="10" fill="#555">평균 위치 오차 mm, 쌍극자 1개</text>']
    for v in range(0, 71, 10):
        g.append(f'<line x1="{x0}" x2="{x1}" y1="{ys(v):.0f}" y2="{ys(v):.0f}" stroke="#e6e6e6"/><text x="{x0 - 6}" y="{ys(v) + 4:.0f}" font-size="10" text-anchor="end" fill="#666">{v}</text>')
    cs = [int(c[:-2]) for c in conds]
    for c in cs:
        if c != 53:
            g.append(f'<text x="{xs(c):.0f}" y="{y0 + 15}" font-size="10" text-anchor="middle" fill="#333">{c}</text>')
    g.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 32}" font-size="10" text-anchor="middle" fill="#555">채널 수 (로그 눈금)</text>')
    for i, (lab, col) in enumerate([("sLORETA (자기 머리)", "#8a9099"), ("쌍극자 스캔 (자기 머리, 학습 없음)", "#b33"), ("범용 1개: 마스크", "#d9822b"), ("범용 1개: 어댑터", "#2b6cb0")]):
        pts = " ".join(f"{xs(c):.1f},{ys(curve[f'{c}채널'][i].mean()):.1f}" for c in cs)
        g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>' + "".join(f'<circle cx="{xs(c):.1f}" cy="{ys(curve[f"{c}채널"][i].mean()):.1f}" r="3" fill="{col}"/>' for c in cs))
        g.append(f'<rect x="{64 + (i % 2) * 250}" y="{344 + (i // 2) * 16}" width="10" height="10" fill="{col}"/><text x="{79 + (i % 2) * 250}" y="{353 + (i // 2) * 16}" font-size="10" fill="#333">{lab}</text>')
    g.append("</svg>")
    open(os.path.join(HERE, "fig_scan_curve.svg"), "w", encoding="utf-8").write("".join(g))

# ---------- 2. 넓은 신호원
A, S = read(W6, "oasis_extended_seed[0-9].csv"), read(HERE, "scan_extended_seed[0-9].csv")
if S:
    seeds = sorted({r["시드"] for r in S})
    chk.append(check(A, S, ["시드", "조건", "방법", "머리", "신호원수"], "AUC", "넓은 신호원"))

    def pe(R, cond, m, g, col):
        return np.array([np.mean([float(r[col]) for r in R if r["시드"] == s and r["조건"] == cond and r["방법"] == m and r["신호원수"] == g]) for s in seeds])
    M = [("MNE (자기 머리)", A), ("sLORETA (자기 머리)", A), ("쌍극자 스캔 (자기 머리)", S), ("범용 1개: 마스크", A), ("범용 1개: 어댑터", A)]
    for cond in ("표준 64채널", "표준 16채널"):
        out += [f"## 2. 넓은 신호원, {cond}", "", "| 방법 | AUC 1개 | AUC 2~3개 | 위치 오차 1개 | 위치 오차 2~3개 | 놓침 % 2~3개 |", "| --- | --- | --- | --- | --- | --- |"]
        for m, R in M:
            out.append(f"| {m} | {ci(pe(R, cond, m, '신호원 1개', 'AUC'), 3)} | {ci(pe(R, cond, m, '신호원 2~3개', 'AUC'), 3)} | "
                       f"{ci(pe(R, cond, m, '신호원 1개', '위치오차mm'))} | {ci(pe(R, cond, m, '신호원 2~3개', '위치오차mm'))} | {pe(R, cond, m, '신호원 2~3개', '놓침%').mean():.0f} |")
        d = pe(A, cond, "범용 1개: 어댑터", "전체", "AUC") - pe(S, cond, "쌍극자 스캔 (자기 머리)", "전체", "AUC")
        d2 = pe(A, cond, "범용 1개: 어댑터", "신호원 1개", "위치오차mm") - pe(S, cond, "쌍극자 스캔 (자기 머리)", "신호원 1개", "위치오차mm")
        out += ["", f"어댑터 − 쌍극자 스캔: AUC(전체) {ci(d, 3)}, 위치 오차(1개) {ci(d2)} mm", ""]

# ---------- 3. 여러 신호원 (순차 제거)
A, S = read(HERE, "ranked_seed[0-9].csv"), read(HERE, "scan_ranked_seed[0-9].csv")
if S:
    seeds = sorted({r["시드"] for r in S})
    chk.append(check(A, S, ["시드", "배치", "방법", "머리", "신호원수"], "평균mm", "여러 신호원"))

    def pr(R, mont, m, k, col="평균mm"):
        return np.array([np.mean([float(r[col]) for r in R if r["시드"] == s and r["배치"] == mont and r["방법"] == m and r["신호원수"] == str(k)]) for s in seeds])
    M = [("sLORETA 순차 제거", A), ("쌍극자 스캔 순차 제거", S), ("MUSIC 순차 제거", S), ("어댑터 순차 제거 (일반 학습)", A), ("어댑터 순차 제거 (제거 학습)", A)]
    for mont in ("64채널", "32채널", "16채널"):
        out += [f"## 3. 여러 신호원, {mont} (시간 창 사용, 순차 제거)", "", "| 방법 | 1개 | 2개 | 3개 | 놓침 % (1 / 2 / 3) |", "| --- | --- | --- | --- | --- |"]
        for m, R in M:
            out.append(f"| {m} | " + " | ".join(ci(pr(R, mont, m, k)) for k in (1, 2, 3)) + " | " + " / ".join(f"{pr(R, mont, m, k, '놓침%').mean():.0f}" for k in (1, 2, 3)) + " |")
        best = lambda k: np.minimum(pr(A, mont, "어댑터 순차 제거 (일반 학습)", k), pr(A, mont, "어댑터 순차 제거 (제거 학습)", k))
        out += ["", "어댑터(둘 중 나은 쪽) − MUSIC: " + ", ".join(f"{k}개 {ci(best(k) - pr(S, mont, 'MUSIC 순차 제거', k))}" for k in (1, 2, 3)) + " mm", ""]
    H = {}
    for pat in ("ranked_hist_seed[0-9].csv", "scan_ranked_hist_seed[0-9].csv"):
        for f in sorted(glob.glob(os.path.join(HERE, pat))):
            scan = "scan_" in os.path.basename(f)
            with open(f, encoding="utf-8-sig") as fh:
                rd = csv.reader(fh)
                hdr = next(rd)
                for r in rd:
                    if scan != (r[1] in ("쌍극자 스캔 순차 제거", "MUSIC 순차 제거")):
                        continue
                    H[(r[0], r[1], r[2])] = H.get((r[0], r[1], r[2]), 0) + np.array(r[3:], float)
    edges = np.array(hdr[3:], float)
    out += ["## 3-1. 64채널, 신호원 2개 오차 분포", "", "| 방법 | 10 mm 이내 | 20 mm 이내 | 40 mm 넘음 |", "| --- | --- | --- | --- |"]
    for m, _ in M:
        h = H[("64채널", m, "2")]
        out.append(f"| {m} | {100 * h[edges < 10].sum() / h.sum():.0f}% | {100 * h[edges < 20].sum() / h.sum():.0f}% | {100 * h[edges >= 40].sum() / h.sum():.0f}% |")
    out.append("")

out += ["## 확인: 평가 샘플이 원래 실행과 같은가", ""] + [f"- {c}" for c in chk]
txt = "\n".join(out)
print(txt)
open(os.path.join(HERE, "scan_summary.md"), "w", encoding="utf-8").write(txt + "\n")
