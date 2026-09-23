# -*- coding: utf-8 -*-
"""공통 가중치 공간.

같은 BASE 를 공유하는 fleet 에서는 (layer, role) 위치가 이미 대응된다.
남는 일은 크기를 맞추는 것뿐이다.

  eq:norm_matching (수정판)
      후보의 크기를 w_theta 의 norm 이 아니라 그 위치에서 K 개 후보 norm 의
      중앙값에 맞춘다. w_theta 에 맞추면 실제 Δw 보다 수십 배 큰 섭동이 되어
      비교가 크기에 지배된다.

  측정에 쓴 후보를 그대로 조립에 쓴다. 고른 뒤 크기를 바꾸지 않는다.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .modeling import Key


class Candidates:
    """위치별 norm 을 맞춘 후보 집합. GPU 에 상주한다."""

    def __init__(self, deltas: Dict[str, Dict[Key, torch.Tensor]],
                 names: Sequence[str], keys: Sequence[Key], device: str):
        self.names = list(names)
        self.keys = list(keys)
        self.norm = {n: {k: float(deltas[n][k].norm()) for k in keys}
                     for n in names}
        self.median = {k: float(np.median([self.norm[n][k] for n in names]))
                       for k in keys}
        self.cand: Dict[str, Dict[Key, torch.Tensor]] = {
            n: {k: (deltas[n][k] * (self.median[k] / max(self.norm[n][k], 1e-12))
                    ).to(device) for k in keys}
            for n in names
        }

    def __getitem__(self, name: str) -> Dict[Key, torch.Tensor]:
        return self.cand[name]

    def get(self, name: str, key: Key) -> torch.Tensor:
        return self.cand[name][key]

    def by_index(self, i: int, key: Key) -> torch.Tensor:
        return self.cand[self.names[i]][key]

    def raw(self, name: str, key: Key) -> torch.Tensor:
        """정규화 이전 크기로 되돌린다(사본을 두지 않는다)."""
        s = self.norm[name][key] / max(self.median[key], 1e-12)
        return self.cand[name][key] * s

    # ------------------------------------------------ 다양성 관문
    def pairwise_cosine(self) -> List[Tuple[str, str, float, float, float]]:
        rows = []
        for a, b in itertools.combinations(self.names, 2):
            c = [float(F.cosine_similarity(self.cand[a][k].flatten(),
                                           self.cand[b][k].flatten(), dim=0))
                 for k in self.keys]
            rows.append((a, b, float(np.median(c)),
                         float(np.percentile(c, 10)), float(np.percentile(c, 90))))
        return rows

    def diversity_gate(self, cos_max: float, log=print) -> float:
        rows = self.pairwise_cosine()
        log("[다양성] 칸별 코사인")
        for a, b, med, p10, p90 in rows:
            log(f"  {a}-{b}  median={med:.4f}  p10={p10:.4f}  p90={p90:.4f}")
        worst = max(r[2] for r in rows)
        if worst > cos_max:
            raise SystemExit(
                f"칸별 코사인 중앙값 최대 {worst:.3f} > {cos_max}. 합칠 것이 없다.")
        return worst
