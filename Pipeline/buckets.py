# -*- coding: utf-8 -*-
"""base 취약도별 이득 분해.

논문의 축이 base 로 바뀌면서 병합의 역할도 바뀌었다.

    base 가 주축이고, 병합은 **base 가 덮지 못한 영역을 덮는다.**

이건 평균의 주장이 아니라 **분포**의 주장이다. 평균만 보면 "병합이 조금 낫다"
와 "병합이 base 의 구멍을 메운다" 를 구분할 수 없다. 그래서 질의를 base 의
점수로 줄 세워 버킷으로 나누고, 버킷마다 이득을 따로 낸다.

    성립: 이득이 **낮은 버킷에 몰린다**   -> base 가 못 하는 데서 병합이 산다
    기각: 이득이 버킷에 고르게 퍼진다     -> 그냥 평균 효과지 보정이 아니다

이 그림이 기여 1 의 본체다. 그리고 예측이 하나 따라 나온다. base 가 커지면
메울 구멍이 줄어드니 **이득이 줄어야 한다**. 크기 축의 두 점에서 이 기울기가
같은 방향이면 기전 주장이 선다.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def by_base_weakness(base: Sequence[float], arms: Dict[str, Sequence[float]],
                     ref: str, n_bucket: int = 5) -> dict:
    """질의를 base 점수로 n_bucket 등분하고 버킷마다 `arm - ref` 를 낸다.

    base   BASE(Basic theta)의 질의별 점수. 버킷을 가르는 기준이다
    arms   비교할 arm 들의 질의별 점수
    ref    기준선 arm 이름(보통 최고 단일 surrogate)
    """
    b = np.asarray(base, dtype=np.float64)
    n = len(b)
    order = np.argsort(b, kind="stable")
    edges = [order[int(round(i * n / n_bucket)):int(round((i + 1) * n / n_bucket))]
             for i in range(n_bucket)]
    r = np.asarray(arms[ref], dtype=np.float64)

    rows = []
    for i, idx in enumerate(edges):
        if len(idx) == 0:
            continue
        row = {"bucket": i, "n": int(len(idx)),
               "base_mean": float(b[idx].mean()),
               "base_range": [float(b[idx].min()), float(b[idx].max())],
               "ref_mean": float(r[idx].mean()), "gain": {}}
        for nm, v in arms.items():
            if nm == ref:
                continue
            d = np.asarray(v, dtype=np.float64)[idx] - r[idx]
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            row["gain"][nm] = [float(d.mean()), float(d.mean() - 1.96 * se),
                               float(d.mean() + 1.96 * se)]
        rows.append(row)

    # 이득이 base 점수와 음의 상관이면 "구멍을 메운다" 는 뜻이다.
    slope = {}
    for nm, v in arms.items():
        if nm == ref:
            continue
        d = np.asarray(v, dtype=np.float64) - r
        lo = np.array([x["gain"][nm][0] for x in rows])
        bm = np.array([x["base_mean"] for x in rows])
        c = float(np.corrcoef(b, d)[0, 1]) if b.std() > 0 and d.std() > 0 else 0.0
        slope[nm] = {"corr_query": c,
                     "bucket_first_minus_last": float(lo[0] - lo[-1]),
                     "bucket_gain": lo.tolist(), "bucket_base": bm.tolist()}
    return {"ref": ref, "rows": rows, "slope": slope}


def report(bk: dict, label: str, log=print) -> None:
    rows, ref = bk["rows"], bk["ref"]
    if not rows:
        return
    names = list(rows[0]["gain"])
    log(f"\n[버킷 {label}]  질의를 BASE 점수로 5 등분. 기준선 = {ref}")
    log(f"  {'버킷':>4s} {'n':>4s} {'BASE':>8s} {ref[:14]:>14s}  "
        + "  ".join(f"{n[:14]:>14s}" for n in names))
    for r in rows:
        log(f"  {r['bucket']:4d} {r['n']:4d} {r['base_mean']:8.4f} "
            f"{r['ref_mean']:14.4f}  "
            + "  ".join(f"{r['gain'][n][0]:+14.4f}" for n in names))
    log(f"  {'기울기 (낮은버킷 - 높은버킷)':>34s}  "
        + "  ".join(f"{bk['slope'][n]['bucket_first_minus_last']:+14.4f}"
                    for n in names))
    log(f"  {'질의별 상관 (BASE, 이득)':>34s}  "
        + "  ".join(f"{bk['slope'][n]['corr_query']:+14.4f}" for n in names))
    best = max(names, key=lambda n: bk["slope"][n]["bucket_first_minus_last"])
    sl = bk["slope"][best]
    if sl["bucket_first_minus_last"] > 0 and sl["corr_query"] < 0:
        log(f"  -> {best}: 이득이 낮은 버킷에 몰린다. "
            f"'base 가 놓친 것을 병합이 잡는다' 와 맞는다.")
    else:
        log(f"  -> 이득이 낮은 버킷에 몰리지 않는다. 보정이 아니라 평균 효과다.")
