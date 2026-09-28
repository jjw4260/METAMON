# -*- coding: utf-8 -*-
"""조립 방식. 모두 같은 후보 집합에서 만들어야 비교가 성립한다.

설계 문서(전체 실험 구성)의 표에 실제로 들어가는 것만 남긴다. 지난 실행은
62 개 arm 을 돌렸는데 그 중 24 개가 문서에 없는 것이었고, 정작 문서에 있는
`loo_k` 는 빠져 있었다.

  metamon_layer   eq:layer_selection + eq:representative_update   (E1 Ours)
  metamon_cell    (layer, role) 마다 argmax                        (E4 w/o Layer-wise)
  greedy          최고 단일에서 출발해 좋아질 때만 채택            (E1 Ours)
  soup            균등 평균                                        (E4 w/o Selection—Uniform)
  soup_raw        정규화 없는 균등 평균                            (E4 w/o Norm Matching)
  random_r        칸마다 균등 무작위                               (E4 w/o Selection—Random)
  shuffle_0       metamon 점유율 유지, 위치만 섞음                 (선택 자체의 값어치)
  weighted_tT     softmax(PartialScore / T)                        (E5 온도 민감도)
  loo_k           k 번째 Local 을 뺀 평균                          (E2 Leave-One-Out)
  fleet_g         겹치지 않는 묶음 g 의 평균. union_g 와 데이터량이 같다 (E2 대조)

**greedy 의 목적함수는 호출자가 준다.** 지난 실행은 채택 여부를 check 의
avgBF 로 정했는데, 병합 arm 에서 avgBF 와 생성 ROUGE-L 의 Spearman 이
-0.821 이었다. 확률을 올리는 방향이 생성을 내리는 방향이어서, 최고 단일에서
출발하고도 생성에서 졌다. 보고할 지표로 채택해야 그 지표에서 최고 단일 이상이
구성상 보장된다.
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
Scorer = Callable[[], float]        # 모델이 이미 적용된 상태에서 점수 하나


# ---------------------------------------------------------------- 기본 조립
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


def mean_src(store, names: Sequence[str]) -> Source:
    """정규화 없는 원본 Δw 의 균등 평균. 데이터량을 맞춘 대조에 쓴다."""
    n = len(names)
    return lambda key: sum(store.raw(m, key) for m in names) / n


def single(store, name: str) -> Source:
    """개별 arm 은 정규화하지 않은 원본 Δw 를 쓴다."""
    return lambda key: store.raw(name, key)


def build_sources(ws: WeightSpace, cfg: Config, cand: Candidates,
                  con: Contribution, store, win_layer: Dict[int, int],
                  keep: set, win_cell: Dict[Key, int]) -> Dict[str, Source]:
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
    for r in range(cfg.n_shuffle):
        src[f"shuffle_{r}"] = shuffle_pick(cand, keys, owners, 300 + r)
    for i in range(cfg.k):
        src[f"loo_{i}"] = leave_one_out(cand, i)
    # fleet_g 와 union_g 는 본 데이터가 같다. E2 의 판정이 이 짝 위에 선다.
    for g, members in enumerate(cfg.fleets):
        src[f"fleet_{g}"] = mean_src(
            store, [cfg.local_names[i] for i in members])
    for name in cfg.arm_names:
        src[name] = single(store, name)
    return src


# ---------------------------------------------------------------- greedy
#
# soup 은 K 개 균등 평균이라 나쁜 Local 이 좋은 Local 을 끌어내린다. 구조적으로
# 최고 단일을 못 이긴다. greedy 는 **최고 단일에서 출발**하므로 `score` 가
# 재는 지표에서 최고 단일 이하로 내려가지 않는다. 그래서 `score` 는 반드시
# 보고할 지표여야 한다.

def greedy_soup(ws: WeightSpace, store, order: Sequence[str],
                score: Scorer, scales: Sequence[float], log=print):
    """Model Soup 의 greedy (Wortsman et al.). order 는 내림차순이어야 한다."""
    def ev(members):
        best = -1e30
        src = mean_src(store, members)
        for s in scales:
            ws.apply(src, s)
            best = max(best, score())
        ws.reset()
        return best

    kept = [order[0]]
    cur = ev(kept)
    log(f"[greedy soup] 시작 {order[0]}  {cur:.5f}")
    for n in order[1:]:
        v = ev(kept + [n])
        if v > cur:
            kept, cur = kept + [n], v
            log(f"  + {n:10s} {v:.5f}  채택 ({len(kept)}개)")
        else:
            log(f"  + {n:10s} {v:.5f}  버림")
    log(f"[greedy soup] 최종 {len(kept)}개 {kept}  {cur:.5f}")
    return mean_src(store, kept), kept, cur


def metamon_greedy(ws: WeightSpace, cand: Candidates, con: Contribution,
                   store, start: str, win_cell: Dict[Key, int], scale: float,
                   score: Scorer, max_cells: int = 40, log=print):
    """기여도 유도 greedy.

    최고 단일에서 출발해 PartialScore 가 큰 칸부터 그 칸의 argmax 후보로
    바꿔 보고 `score` 가 좋아질 때만 채택한다. 생성으로 채점하면 칸 하나마다
    문장을 만들어야 하므로 상위 `max_cells` 칸만 본다.
    """
    keys = list(ws.keys)
    cur = {k: store.raw(start, k).clone() for k in keys}
    src = lambda k: cur[k]
    ws.apply(src, scale)
    best = score()
    log(f"[metamon greedy] 시작 {start} @배율 {scale}  {best:.5f}")

    order = sorted(keys, key=lambda k: -max(con.score[k]))
    n_try = n_ok = 0
    for key in order:
        if n_try >= max_cells:
            break
        j = win_cell[key]
        name = cand.names[j]
        if name == start or con.score[key][j] <= 0:
            continue
        n_try += 1
        old = cur[key]
        cur[key] = (con.alpha[key][j] * cand.get(name, key)).clone()
        ws.apply(src, scale, subset=[key])
        v = score()
        if v > best:
            best = v
            n_ok += 1
        else:
            cur[key] = old
            ws.apply(src, scale, subset=[key])
    ws.reset()
    log(f"[metamon greedy] 칸 {n_ok}/{n_try} 채택 (상위 {max_cells} 중)  {best:.5f}")
    return (lambda k: cur[k]), n_ok, n_try, best
