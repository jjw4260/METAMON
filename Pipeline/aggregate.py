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
    # 독립 fleet 은 curve_sources 가 cm{m}_{g} 로 만든다.
    # cm{fleet_size}_{g} 가 union_g 와 데이터량이 같은 짝이다.
    for name in cfg.arm_names:
        src[name] = single(deltas, name, ws.device)
    return src


# ---------------------------------------------------------------- greedy 계열
#
# soup 은 K 개를 균등 평균하므로 나쁜 Local 이 좋은 Local 을 끌어내린다.
# 구조적으로 최고 단일을 못 이긴다. 아래 둘은 **최고 단일에서 출발**하므로
# check 기준으로 최고 단일 이하로 내려가지 않는다.

def _mean_src(deltas, names, device):
    n = len(names)
    return lambda key: sum(deltas[m][key].to(device) for m in names) / n


def greedy_soup(ws: WeightSpace, cfg: Config,
                deltas: Dict[str, Dict[Key, torch.Tensor]],
                order: Sequence[str], check, scales: Sequence[float],
                log=print):
    """Model Soup 의 greedy (Wortsman et al.).

    check 에서 좋은 순으로 하나씩 더하고, 좋아질 때만 남긴다.
    order 는 이미 check 기준 내림차순이어야 한다.
    """
    from .metrics import sim

    def ev(members):
        best_v, best_s = -1.0, 0.0
        src = _mean_src(deltas, members, ws.device)
        for s in scales:
            ws.apply(src, s)
            v = float(np.mean(sim(ws.model, check)))
            if v > best_v:
                best_v, best_s = v, s
        ws.reset()
        return best_v, best_s

    kept = [order[0]]
    cur, _ = ev(kept)
    log(f"[greedy soup] 시작 {order[0]}  check {cur:.5f}")
    for n in order[1:]:
        v, _ = ev(kept + [n])
        if v > cur:
            kept, cur = kept + [n], v
            log(f"  + {n:10s} check {v:.5f}  채택 ({len(kept)}개)")
        else:
            log(f"  + {n:10s} check {v:.5f}  버림")
    log(f"[greedy soup] 최종 {len(kept)}개 {kept}  check {cur:.5f}")
    return _mean_src(deltas, kept, ws.device), kept


def metamon_greedy(ws: WeightSpace, cfg: Config, cand: Candidates,
                   con: Contribution, deltas: Dict[str, Dict[Key, torch.Tensor]],
                   start: str, win_cell: Dict[Key, int], scale: float,
                   check, log=print):
    """기여도 유도 greedy.

    최고 단일 surrogate 에서 출발해, PartialScore 가 큰 칸부터 그 칸의
    기여도 argmax 후보로 바꿔 보고 check 가 좋아질 때만 채택한다.
    eq:assembly_verification 의 greedy 복구를 조립 자체에 적용한 것이다.
    """
    from .metrics import sim

    keys = list(ws.keys)
    cur = {k: deltas[start][k].to(ws.device) for k in keys}
    src = lambda k: cur[k]
    ws.apply(src, scale)
    best = float(np.mean(sim(ws.model, check)))
    log(f"[metamon greedy] 시작 {start} @배율 {scale}  check {best:.5f}")

    order = sorted(keys, key=lambda k: -max(con.score[k]))
    n_try = n_ok = 0
    for key in order:
        j = win_cell[key]
        name = cand.names[j]
        if name == start or con.score[key][j] <= 0:
            continue
        n_try += 1
        old = cur[key]
        cur[key] = con.alpha[key][j] * cand.get(name, key)
        ws.apply(src, scale, subset=[key])
        v = float(np.mean(sim(ws.model, check)))
        if v > best:
            best = v
            n_ok += 1
        else:
            cur[key] = old
            ws.apply(src, scale, subset=[key])
    ws.reset()
    log(f"[metamon greedy] 칸 {n_ok}/{n_try} 채택  check {best:.5f}")
    return (lambda k: cur[k]), n_ok, n_try


def curve_sources(cfg: Config, deltas: Dict[str, Dict[Key, torch.Tensor]],
                  device: str) -> Dict[str, Source]:
    """종속성 곡선. m 개씩 겹치지 않게 묶은 균등 평균."""
    out: Dict[str, Source] = {}
    for m in cfg.curve:
        for g, members in enumerate(cfg.fleets_of(m)):
            out[f"cm{m}_{g}"] = _mean_src(
                deltas, [cfg.local_names[i] for i in members], device)
    return out
