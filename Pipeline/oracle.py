# -*- coding: utf-8 -*-
"""상보성 측정. 병합을 시험하기 전에 **뽑을 것이 있는지**부터 본다.

    oracle(x) = max_k  m_k(x)          질의마다 최고 Local 을 골랐다면

`oracle` 은 어떤 결합 규칙도 넘을 수 없는 상한이다(선택 기반 결합에 대해).
그래서 두 수의 차가 전부를 결정한다.

    headroom = mean(oracle) - max_k mean(m_k)

  headroom ~ 0   조각이 IID 다. 한 Local 이 거의 모든 질의에서 이기고 있다.
                 어떤 병합도 최고 단일을 못 이긴다. 기전을 고칠 문제가 아니라
                 조각을 바꿀 문제다.
  headroom >> 0  상보성이 있다. 이제 "얼마나 회수했나" 가 기전의 성적이다.

    recovery = (mean(arm) - mean(best_local)) / headroom

지난 실행에서 이것을 재지 않은 것이 실수였다. 기전을 여섯 군데 고치기 전에
뽑을 것이 있는지부터 확인해야 했다. toy 에서는 oracle 0.8405 / best_local
0.4341 로 headroom 이 0.41 이었는데, 그 조건에서도 칸 argmax 는 균등 평균에
졌다. 실제 런의 headroom 이 얼마인지가 아이디어의 생사다.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def complementarity(per_query: Dict[str, Sequence[float]],
                    locals_: Sequence[str], best: str,
                    arms: Dict[str, Sequence[float]] | None = None,
                    higher_better: bool = True) -> dict:
    """질의별 값에서 oracle, headroom, 승자 점유, 회수율을 낸다."""
    M = np.stack([np.asarray(per_query[n], dtype=np.float64) for n in locals_])
    if not higher_better:
        M = -M
    orc = M.max(0)
    means = M.mean(1)
    top = float(means.max())
    head = float(orc.mean() - top)
    win = M.argmax(0)
    occ = np.bincount(win, minlength=len(locals_)).tolist()
    # 상위 하나가 몇 %의 질의에서 이기나. 1/K 면 완전 상보, 1 이면 지배다.
    dom = float(max(occ)) / len(orc)
    bm = float(np.asarray(per_query[best], dtype=np.float64).mean())
    out = {"oracle": float(orc.mean()), "best_local_mean": top,
           "headroom": head, "occupancy": occ, "dominance": dom,
           "best_named": best, "best_named_mean": bm if higher_better else -bm,
           "recovery": {}}
    if arms and head > 1e-12:
        for nm, v in arms.items():
            a = float(np.asarray(v, dtype=np.float64).mean())
            if not higher_better:
                a = -a
            out["recovery"][nm] = (a - top) / head
    return out


def report(o: dict, label: str, k: int, log=print) -> None:
    log(f"\n[상보성 {label}]")
    log(f"  oracle(질의별 최고 Local) {o['oracle']:.5f}   "
        f"최고 단일 {o['best_local_mean']:.5f}   "
        f"headroom {o['headroom']:+.5f}")
    log(f"  승자 점유 {o['occupancy']}")
    log(f"  최상위 Local 이 이긴 질의 {o['dominance']*100:.1f}%   "
        f"(완전 상보 {100.0/k:.1f}%, 완전 지배 100%)")
    if o["recovery"]:
        s = "  ".join(f"{nm} {r*100:+.1f}%"
                      for nm, r in sorted(o["recovery"].items(),
                                          key=lambda kv: -kv[1]))
        log(f"  headroom 회수율  {s}")
    if o["headroom"] <= 1e-4:
        log("  *** headroom 이 0 이다. 한 Local 이 모든 질의에서 이기고 있어")
        log("      선택으로 얻을 것이 없다. 기전이 아니라 조각(shard)의 문제다.")
    elif o["dominance"] > 0.5:
        log("  *** 한 Local 이 질의 과반을 이긴다. 조각이 아직 덜 갈렸다.")
