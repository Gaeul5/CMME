"""Week 5-4: OASIS-3 피험자를 학습 / 평가(hold-out)로 나누는 스크립트.

11/4 메모: "50개 정도 골고루 뽑고, 50개 밖의 새 데이터에서도 잘 뽑는지 확인"
OASIS-3는 종단 데이터라 한 사람에게 MR 세션이 여러 개 있다. 세션 단위로 나누면 같은 사람의 머리가
학습과 평가 양쪽에 들어가서 "새 머리" 평가가 무의미해진다. 그래서
  1) 피험자당 세션 1개만 고른다 (기본: 가장 이른 세션)
  2) 층(나이 구간 × 인지 상태 × 자기장 세기)을 만들고
  3) 평가 피험자를 먼저 층별 비율대로 떼어 둔 뒤
  4) 남은 피험자에서 학습 피험자를 층마다 최대한 같은 수로 뽑는다 ("골고루")
  5) 학습과 평가에 같은 피험자가 없는지 확인한다

OASIS-3 표의 열 이름은 내려받은 파일마다 다를 수 있어 인자로 받는다. 실제 열 이름을 확인하고 지정할 것.

실행 예:
  python week05/oasis_split.py --table oasis3_mr_sessions.csv --heads-dir heads_oasis \
      --col-subject OASISID --col-session session --col-age age --col-cdr cdr --col-field field \
      --n-train 50 --n-test 20 --out split.json
출력 split.json은 multihead.py에 SPLIT=split.json 으로 넣는다. heads-dir 안의 npz 이름은 <피험자ID>.npz 로 가정.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


def pick_sessions(df, c, how):
    df = df.dropna(subset=[c["subject"], c["age"]]).copy()
    df = df.sort_values([c["subject"], c["age"]])
    g = df.groupby(c["subject"])
    return (g.head(1) if how == "first" else g.tail(1)).reset_index(drop=True)  # 한 행 통째로 고름 (열이 섞이지 않게)


def strata(df, c, age_edges):
    age_bin = pd.cut(df[c["age"]], bins=[-np.inf, *age_edges, np.inf], labels=False).astype(int)
    parts = [age_bin.map(lambda b: f"나이{b}")]
    if c.get("cdr") and c["cdr"] in df:
        parts.append(np.where(df[c["cdr"]].fillna(0) > 0, "CDR>0", "CDR0"))
    if c.get("field") and c["field"] in df:
        parts.append(df[c["field"]].astype(str).radd("자기장"))
    if c.get("sex") and c["sex"] in df:
        parts.append(df[c["sex"]].astype(str).radd("성별"))
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key + "|" + pd.Series(p, index=df.index).astype(str)
    return key


def split(df, key, n_train, n_test, rng):
    subj = df.index.to_numpy()
    groups = {k: rng.permutation(subj[key.to_numpy() == k]).tolist() for k in sorted(key.unique())}
    total = sum(len(v) for v in groups.values())
    if n_train + n_test > total:
        raise ValueError(f"피험자 {total}명으로는 학습 {n_train} + 평가 {n_test}를 만들 수 없음")
    # 평가: 층 크기 비율대로 (최대 나머지 방식)
    quota = {k: n_test * len(v) / total for k, v in groups.items()}
    take = {k: int(np.floor(q)) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[: n_test - sum(take.values())]:
        take[k] += 1
    test = []
    for k, v in groups.items():
        test += v[: take[k]]
        groups[k] = v[take[k]:]
    # 학습: 층마다 같은 수를 목표로 돌아가며 한 명씩
    train = []
    while len(train) < n_train:
        moved = False
        for k in groups:
            if groups[k] and len(train) < n_train:
                train.append(groups[k].pop(0))
                moved = True
        if not moved:
            raise ValueError("학습 피험자가 부족함")
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="세션 단위 표(csv). 한 행 = MR 세션 하나")
    ap.add_argument("--heads-dir", required=True, help="oasis_pipeline.py 출력 npz 폴더 (<피험자ID>.npz)")
    ap.add_argument("--col-subject", default="subject")
    ap.add_argument("--col-session", default="session")
    ap.add_argument("--col-age", default="age")
    ap.add_argument("--col-cdr", default="cdr")
    ap.add_argument("--col-field", default="field_strength")
    ap.add_argument("--col-sex", default="")
    ap.add_argument("--age-edges", type=float, nargs="*", default=[65, 75])
    ap.add_argument("--pick", choices=["first", "last"], default="first")
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--require-npz", action="store_true", help="npz가 있는 피험자만 후보로 사용")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = dict(subject=a.col_subject, session=a.col_session, age=a.col_age, cdr=a.col_cdr, field=a.col_field, sex=a.col_sex)

    raw = pd.read_csv(a.table)
    df = pick_sessions(raw, c, a.pick).set_index(c["subject"])
    npz = lambda s: os.path.join(a.heads_dir, f"{s}.npz")
    if a.require_npz:
        df = df[[os.path.exists(npz(s)) for s in df.index]]
    print(f"세션 {len(raw)}개 → 피험자 {len(df)}명 (피험자당 1세션, {a.pick})")
    key = strata(df, c, a.age_edges)
    train, test = split(df, key, a.n_train, a.n_test, np.random.default_rng(a.seed))
    assert not set(train) & set(test), "학습과 평가에 같은 피험자가 있음"

    tab = pd.DataFrame({"층": key, "학습": key.index.isin(train), "평가": key.index.isin(test)})
    summ = tab.groupby("층").agg(전체=("층", "size"), 학습=("학습", "sum"), 평가=("평가", "sum"))
    print(summ.to_string())
    for name, ids in (("학습", train), ("평가", test)):
        sub = df.loc[ids]
        msg = f"{name} {len(ids)}명: 나이 {sub[c['age']].mean():.1f} ± {sub[c['age']].std():.1f}"
        if c["cdr"] in sub:
            msg += f", CDR>0 비율 {100 * (sub[c['cdr']].fillna(0) > 0).mean():.0f}%"
        print(msg)
    json.dump(dict(train=[npz(s) for s in train], test=[npz(s) for s in test],
                   train_subjects=list(map(str, train)), test_subjects=list(map(str, test)),
                   sessions={str(s): str(df.loc[s, c["session"]]) for s in train + test} if c["session"] in df else {},
                   seed=a.seed, strata=summ.reset_index().to_dict("records")),
              open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
