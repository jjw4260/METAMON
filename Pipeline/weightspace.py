# -*- coding: utf-8 -*-
"""공통 가중치 공간.

같은 BASE 를 공유하는 fleet 에서는 (layer, role) 위치가 이미 대응된다.
남는 일은 크기를 맞추는 것뿐이다.

  eq:norm_matching (수정판)
      후보의 크기를 w_theta 의 norm 이 아니라 그 위치에서 K 개 후보 norm 의
      중앙값에 맞춘다. w_theta 에 맞추면 실제 Δw 보다 수십 배 큰 섭동이 되어
      비교가 크기에 지배된다.

  측정에 쓴 후보를 그대로 조립에 쓴다. 고른 뒤 크기를 바꾸지 않는다.

후보는 **CPU 에 둔다.** K 개를 전부 GPU 에 올리면 K x 조립대상 만큼 든다.
TinyLlama-1.1B 는 조립 대상이 969M 이라 fp32 로 후보 하나가 3.9GB,
K=8 이면 31GB 다. 모델(4.4) 과 BASE 사본(3.9) 까지 더하면 A100 40GB 도
한계에 붙고 16GB 는 그 자리에서 터진다.

접근은 전부 **칸 단위**다(`apply` 도 `measure` 도 key 하나씩 돈다). 그래서
현재 칸의 K 개만 GPU 에 올려 두는 캐시 하나면 충분하다. 가장 큰 칸이
down_proj 5632x2048 = 46MB 이므로 K=8 에서 370MB 다.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .modeling import Key


class Candidates:
    """위치별 norm 을 맞춘 후보 집합. CPU 에 두고 칸 단위로 GPU 에 올린다."""

    def __init__(self, deltas: Dict[str, Dict[Key, torch.Tensor]],
                 names: Sequence[str], keys: Sequence[Key], device: str,
                 log=print):
        self.names = list(names)
        self.keys = list(keys)
        self.device = device
        # 원본 Δw 를 참조만 한다. 제자리로 고치면 local_k / union_g / fleet_g
        # arm 이 정규화된 Δ 를 쓰게 되고, 사본을 만들면 CPU 에 K 벌이 더 든다.
        # 정규화 계수만 갖고 있다가 전송 뒤 GPU 에서 곱한다.
        self.src = deltas
        self.norm = {n: {k: float(deltas[n][k].norm()) for k in keys}
                     for n in names}
        self.median = {k: float(np.median([self.norm[n][k] for n in names]))
                       for k in keys}
        self._ck: Optional[Key] = None          # 현재 GPU 에 올라온 칸
        self._buf: Dict[str, torch.Tensor] = {}
        big = max(deltas[names[0]][k].numel() for k in keys) * 4
        log(f"[후보] K={len(self.names)}  CPU 상주  "
            f"칸 캐시 최대 {big*len(self.names)/1e6:.0f}MB")

    def factor(self, name: str, key: Key) -> float:
        """eq:norm_matching (수정판). 위치별 median norm 으로 맞추는 계수."""
        return self.median[key] / max(self.norm[name][key], 1e-12)

    # ---- 캐시는 칸 하나 분이다. apply / measure 가 key 하나씩 돌기 때문이다.
    #      이름별로 게으르게 채운다. soup 처럼 K 개를 다 쓰면 K 개가 올라오고,
    #      representative 처럼 하나만 쓰면 하나만 올라온다.
    def raw(self, name: str, key: Key) -> torch.Tensor:
        """정규화 이전 Δw. GPU 사본을 캐시한다."""
        if self._ck != key:
            self._ck, self._buf = key, {}
        t = self._buf.get(name)
        if t is None:
            t = self.src[name][key].to(self.device, non_blocking=True)
            self._buf[name] = t
        return t

    def get(self, name: str, key: Key) -> torch.Tensor:
        return self.raw(name, key) * self.factor(name, key)

    def __getitem__(self, name: str) -> Dict[Key, torch.Tensor]:
        return self.src[name]

    def by_index(self, i: int, key: Key) -> torch.Tensor:
        return self.get(self.names[i], key)

    def release(self) -> None:
        self._ck, self._buf = None, {}
        torch.cuda.empty_cache()

    # ------------------------------------------------ 다양성 관문
    def pairwise_cosine(self) -> List[Tuple[str, str, float, float, float]]:
        """코사인은 크기에 무관하므로 정규화 전 Δw 로 재도 같다.
        칸을 바깥 고리로 두어 칸당 한 번만 GPU 로 올린다."""
        pairs = list(itertools.combinations(self.names, 2))
        acc: Dict[Tuple[str, str], List[float]] = {p: [] for p in pairs}
        for k in self.keys:
            for a, b in pairs:
                acc[(a, b)].append(float(F.cosine_similarity(
                    self.raw(a, k).flatten(), self.raw(b, k).flatten(), dim=0)))
        self.release()
        return [(a, b, float(np.median(c)), float(np.percentile(c, 10)),
                 float(np.percentile(c, 90)))
                for (a, b), c in acc.items()]

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
