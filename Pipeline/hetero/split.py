# -*- coding: utf-8 -*-
"""질의 X 를 theta 와 Local 에 나누는 방식.

    full      (기존)  theta 와 모든 Local 이 X 전체 (train 4096) 를 쓴다.
    disjoint          theta (그리고 theta') 는 X_theta 만, Local k 는 서로 겹치지 않는
                      X_k 만 쓴다. theta' 의 소유자는 Local 의 가중치만 받고 그들의
                      질의·응답은 보지 못한다. z 가 MLE 항에 없는 정보를 나를 수 있는
                      설정이다.

나누기는 manifest 에 한 번 기록하고 모든 단계가 그것을 읽는다.
    man["split"] = {"mode", "seed", "n_x", "theta": [idx...], "locals": {name: [idx...]}}
idx 는 load_dataset_file 의 items 순서 (0 .. n_x-1). sel / check / test 는 그대로
n_x 뒤의 위치다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def make_split(n_x: int, theta_n: int, names: Sequence[str], seed: int) -> Dict:
    if theta_n >= n_x:
        raise SystemExit(f"theta_n {theta_n} >= n_x {n_x}")
    perm = np.random.default_rng(seed).permutation(n_x)
    rest = perm[theta_n:]
    shards = np.array_split(rest, len(names))
    return {"mode": "disjoint", "seed": int(seed), "n_x": int(n_x),
            "theta": sorted(int(i) for i in perm[:theta_n]),
            "locals": {n: sorted(int(i) for i in s) for n, s in zip(names, shards)}}


def theta_idx(man: Dict, n_x: int) -> List[int]:
    sp = man.get("split")
    return list(range(n_x)) if not sp else list(sp["theta"])


def local_idx(man: Dict, name: str, n_x: int) -> List[int]:
    sp = man.get("split")
    return list(range(n_x)) if not sp else list(sp["locals"][name])


def pick(items: Sequence[dict], idx: Sequence[int]) -> List[dict]:
    return [items[i] for i in idx]


def describe(man: Dict) -> str:
    sp = man.get("split")
    if not sp:
        return "split full (theta 와 Local 모두 X 전체)"
    loc = ", ".join(f"{n.split('/')[-1]} {len(v)}" for n, v in sp["locals"].items())
    return (f"split disjoint (seed {sp['seed']}): theta {len(sp['theta'])} 질의, "
            f"Local {loc}  (서로 겹치지 않음)")
