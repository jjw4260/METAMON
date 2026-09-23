# -*- coding: utf-8 -*-
"""조립 방식. 모두 같은 후보 집합에서 만들어야 비교가 성립한다.

  metamon_layer   eq:layer_selection + eq:representative_update
  metamon_cell    (layer, role) 마다 argmax. 집중도 대조군
  weighted        softmax(PartialScore / T). T -> inf 면 soup, T -> 0 이면 argmax
  soup            균등 평균
  random          칸마다 균등 무작위 선택
  shuffle         metamon 의 점유율을 유지한 채 위치만 섞는다
                  (선택의 기여와 집중의 효과를 분리한다)
  loo_k           k 번째 Local 을 뺀 평균. 겹침이 있어 참고용이다
  fleet_g         겹치지 않는 Local 묶음 g 의 평균. 종속성 판정용
                  같은 조각을 합쳐 학습한 union_g 와 데이터량이 같다
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import Config
from .contribution import Contribution, representative
from .modeling import Key, WeightSpace
from .weightspace import Candidates

Source = Callable[[Key], torch.Tensor]


def soup(cand: Candidates) -> Source:
    k = len(cand.names)
    return lambda key: sum(cand.get(n, key) for n in cand.names) / k


def soup_raw(cand: Candidates) -> Source:
    k = len(cand.names)
    return lambda key: sum(cand.raw(n, key) for n in cand.names) / k


def random_pick(cand: Candidates, keys: Sequence[Key], seed: int) -> Source:
    rng = random.Random(seed)
    m = {k: rng.randrange(len(cand.names)) for k in keys}
    return lambda key: cand.by_index(m[key], key)


def shuffle_pick(cand: Candidates, keys: Sequence[Key],
                 owners: Sequence[int], seed: int) -> Source:
    """점유율은 그대로 두고 위치만 섞는다."""
    perm = list(owners)
    random.Random(seed).shuffle(perm)
    m = dict(zip(keys, perm))
    return lambda key: cand.by_index(m[key], key)


def weighted(cand: Candidates, con: Contribution, keys: Sequence[Key],
             temperature: float) -> Source:
    w: Dict[Key, np.ndarray] = {}
    for key in keys:
        g = np.asarray(con.score[key], dtype=np.float64)
        e = np.exp((g - g.max()) / max(temperature, 1e-30))
        w[key] = e / e.sum()
    return lambda key: sum(float(w[key][j]) * cand.by_index(j, key)
                           for j in range(len(cand.names)))


def weight_spread(con: Contribution, keys: Sequence[Key]) -> float:
    """온도 격자를 스케일 없이 잡기 위한 기준값."""
    return float(np.median([max(con.score[k]) - min(con.score[k]) for k in keys]))


def leave_one_out(cand: Candidates, drop: int) -> Source:
    rest = [n for i, n in enumerate(cand.names) if i != drop]
    return lambda key: sum(cand.get(n, key) for n in rest) / len(rest)


def fleet_soup(deltas: Dict[str, Dict[Key, torch.Tensor]], names: Sequence[str],
               device: str) -> Source:
    """겹치지 않는 Local 묶음 하나의 균등 평균. 종속성 대조용.

    정규화 없이 원본 변화량을 평균한다. 같은 데이터로 학습한 union_g 와
    맞대어 볼 것이므로 추가 기구를 끼워 넣지 않는다.
    """
    n = len(names)
    return lambda key: sum(deltas[m][key].to(device) for m in names) / n


def single(deltas: Dict[str, Dict[Key, torch.Tensor]], name: str,
           device: str) -> Source:
    """개별 arm 은 정규화하지 않은 원본 변화량을 쓴다."""
    return lambda key: deltas[name][key].to(device)


def build_sources(ws: WeightSpace, cfg: Config, cand: Candidates,
                  con: Contribution, deltas: Dict[str, Dict[Key, torch.Tensor]],
                  win_layer: Dict[int, int], keep: set,
                  win_cell: Dict[Key, int]) -> Dict[str, Source]:
    keys = ws.keys
    src: Dict[str, Source] = {
        "metamon_layer": representative(cand, con, win_layer, ws, True, keep),
        "metamon_cell": representative(cand, con, win_cell, ws, False),
        "soup": soup(cand),
        "soup_raw": soup_raw(cand),
    }
    for r in range(cfg.n_random):
        src[f"random_{r}"] = random_pick(cand, keys, 100 + r)
    owners = [win_cell[k] for k in keys]
    for r in range(cfg.n_random):
        src[f"shuffle_{r}"] = shuffle_pick(cand, keys, owners, 300 + r)
    for i in range(cfg.k):
        src[f"loo_{i}"] = leave_one_out(cand, i)
    # 독립 fleet. fleet_g 와 union_g 는 본 데이터가 같다.
    for g, members in enumerate(cfg.fleets):
        src[f"fleet_{g}"] = fleet_soup(
            deltas, [cfg.local_names[i] for i in members], ws.device)
    for name in cfg.arm_names:
        src[name] = single(deltas, name, ws.device)
    return src
