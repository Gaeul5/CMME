"""Week 5 결과 그림(SVG). multihead_seed*.csv를 읽어 시드 평균으로 그림. 실행: python week05/make_figs.py"""
import csv
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = 'font-family="Noto Sans KR,Apple SD Gothic Neo,Malgun Gothic,sans-serif"'


def load(prefix):
    acc = collections.defaultdict(list)
    for s in (0, 1):
        with open(os.path.join(HERE, f"{prefix}_seed{s}.csv"), encoding="utf-8-sig") as f:
            for r in list(csv.reader(f))[1:]:
                acc[(r[0], r[1], r[2])].append((float(r[3]), float(r[5])))
    return {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in acc.items()}


def frame(w, h, title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT}>'
            f'<rect width="{w}" height="{h}" fill="#fff"/><text x="16" y="24" font-size="14" font-weight="600" fill="#222">{title}</text>{body}</svg>')


def grid(x0, y0, w, h, ymax, step, label):
    o = []
    n = int(ymax / step)
    for t in range(n + 1):
        v = step * t
        y = y0 + h - h * v / ymax
        o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.0f}" y2="{y:.0f}" stroke="#e6e6e6"/>'
                 f'<text x="{x0 - 6}" y="{y + 4:.0f}" font-size="10" text-anchor="end" fill="#666">{v:g}</text>')
    o.append(f'<text x="22" y="42" font-size="10" fill="#666">{label}</text>')
    return "".join(o)


def legend(items, x, y):
    return "".join(f'<rect x="{x}" y="{y + i * 16 - 9}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{x + 15}" y="{y + i * 16}" font-size="10" fill="#333">{n}</text>' for i, (n, c) in enumerate(items))


M = load("multihead")
METH = [("MLP 센서만", "센서만", "#9aa0a6"), ("MLP 센서 + 머리 벡터", "센서 + 머리 요약 벡터", "#e08a1e"),
        ("어댑터 (기준 머리 리드필드)", "어댑터 (기준 머리 리드필드)", "#7fb2e5"), ("어댑터 (자기 머리 리드필드)", "어댑터 (자기 머리 리드필드)", "#2b6cb0")]

# 그림 1: 학습 머리 수 K에 따른 새 머리 평균 오차 (선)
Ks = [1, 3, 6, 12]
x0, y0, w, h, ymax = 56, 50, 470, 220, 50
px = lambda i: x0 + 30 + i * (w - 60) / (len(Ks) - 1)
py = lambda v: y0 + h - h * min(v, ymax) / ymax
o = [grid(x0, y0, w, h, ymax, 10, "새 머리 6개 평균 위치 오차 mm")]
for i, K in enumerate(Ks):
    o.append(f'<text x="{px(i):.0f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle" fill="#333">학습 머리 {K}개</text>')
for k, lab, c in METH:
    pts = [(px(i), py(M[("E1", f"K={K} / 새 머리", k)][0])) for i, K in enumerate(Ks)]
    o.append(f'<polyline points="{" ".join(f"{a:.0f},{b:.0f}" for a, b in pts)}" fill="none" stroke="{c}" stroke-width="2.2"/>')
    o.append("".join(f'<circle cx="{a:.0f}" cy="{b:.0f}" r="3.2" fill="{c}"/>' for a, b in pts))
o.append(legend([(l, c) for _, l, c in METH], 56, 312))
o.append('<text x="300" y="312" font-size="10" fill="#555">학습 머리 1개일 때 머리 벡터는 상수라</text>'
         '<text x="300" y="326" font-size="10" fill="#555">"센서만"과 사실상 같음</text>')
fig1 = frame(560, 385, "머리 요약 벡터는 학습 머리가 여러 개가 되면 새 머리에서 무너진다", "".join(o))

# 그림 2: 학습 머리 / 새 머리 / 강한 변형 머리 (막대, K=12)
groups = [("E1", "K=12 / 학습 머리", "학습한 머리"), ("E1", "K=12 / 새 머리", "새 머리 (보통 변형)"), ("E3", "K=12 / 강한 변형 머리", "새 머리 (강한 변형)")]
x0, y0, w, h, ymax = 56, 50, 470, 220, 35
o = [grid(x0, y0, w, h, ymax, 5, "평균 위치 오차 mm (학습 머리 12개 모델)")]
gw = w / 3
bw = gw * 0.8 / 4
for g, (e, c, lab) in enumerate(groups):
    gx = x0 + g * gw + gw * 0.1
    for s, (k, l, col) in enumerate(METH):
        v = M[(e, c, k)][0]
        bh = h * v / ymax
        xx, yy = gx + s * bw, y0 + h - bh
        o.append(f'<rect x="{xx:.0f}" y="{yy:.0f}" width="{bw - 2:.0f}" height="{bh:.0f}" fill="{col}"/>'
                 f'<text x="{xx + (bw - 2) / 2:.0f}" y="{yy - 3:.0f}" font-size="9" text-anchor="middle" fill="#333">{v:.1f}</text>')
    o.append(f'<text x="{x0 + g * gw + gw / 2:.0f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle" fill="#333">{lab}</text>')
o.append(legend([(l, c) for _, l, c in METH], 56, 300))
fig2 = frame(560, 370, "자기 머리 리드필드 어댑터는 머리가 바뀌어도 오차가 거의 그대로다", "".join(o))

# 그림 3: 몽타주 + 새 머리 (E2)
E2M = [("sLORETA (자기 머리)", "sLORETA (자기 머리)", "#9aa0a6"), ("MLP 마스크", "범용 1개: 마스크", "#e08a1e"),
       ("어댑터 (자기 머리 리드필드)", "범용 1개: 어댑터 (자기 머리)", "#2b6cb0")]
conds = [("64채널", "64채널"), ("32채널 (Biosemi32)", "32채널"), ("16채널 (Biosemi16)", "16채널")]
x0, y0, w, h, ymax = 56, 50, 470, 220, 35
o = [grid(x0, y0, w, h, ymax, 5, "새 머리 6개 평균 위치 오차 mm")]
gw = w / 3
bw = gw * 0.8 / 3
for g, (c, lab) in enumerate(conds):
    gx = x0 + g * gw + gw * 0.1
    for s, (k, l, col) in enumerate(E2M):
        v = M[("E2", c, k)][0]
        bh = h * v / ymax
        xx, yy = gx + s * bw, y0 + h - bh
        o.append(f'<rect x="{xx:.0f}" y="{yy:.0f}" width="{bw - 2:.0f}" height="{bh:.0f}" fill="{col}"/>'
                 f'<text x="{xx + (bw - 2) / 2:.0f}" y="{yy - 3:.0f}" font-size="9" text-anchor="middle" fill="#333">{v:.1f}</text>')
    o.append(f'<text x="{x0 + g * gw + gw / 2:.0f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle" fill="#333">{lab}</text>')
o.append(legend([(l, c) for _, l, c in E2M], 56, 300))
fig3 = frame(560, 355, "머리와 몽타주가 둘 다 처음이어도 어댑터가 가장 작은 오차", "".join(o))

for name, svg in (("fig_multihead_K", fig1), ("fig_multihead", fig2), ("fig_multihead_montage", fig3)):
    with open(os.path.join(HERE, f"{name}.svg"), "w", encoding="utf-8") as f:
        f.write(svg)
    print(name, len(svg.encode()))
