# -*- coding: utf-8 -*-
"""Target 응답 재현 기여도와 위치별 선택.

  eq:perturbed_loss       한 (layer, role) 만 치환했을 때의 L(theta; omega)
  eq:partial_score        [ (L0 - min_alpha L) / L0 ]_+   와  Scale = argmin alpha
  eq:layer_selection      LayerScore = sum_rho PartialScore, k_l = argmax,
                          Confidence = 1 위와 2 위의 차이
  eq:representative_update z = Scale * cand   (PartialScore > 0 인 위치만)
  eq:assembly_verification L(theta + z; 1) < L(theta; 1),
                          실패하면 LayerScore 순으로 greedy 복구

선택 기준은 avgBF 가 아니라 손실이다. 확률 평균과 로그 평균은 질의를 가로질러
평균하는 순간 순서가 달라진다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .metrics import EvalSet, loss
from .modeling import Key, WeightSpace
from .weightspace import Candidates


@dataclass
class Contribution:
    score: Dict[Key, List[float]] = field(default_factory=dict)   # PartialScore
    alpha: Dict[Key, List[float]] = field(default_factory=dict)   # Scale
    base_loss: float = 0.0                                        # L(theta; 1)
    base_per_candidate: List[float] = field(default_factory=list)  # L(theta; omega_k)

    # -------- 칸 단위 (대조군)
    def winner_cell(self) -> Dict[Key, int]:
        return {k: int(np.argmax(v)) for k, v in self.score.items()}

    # -------- eq:layer_selection
    def layer_score(self, layers: Sequence[int], roles: Sequence[str],
                    k: int) -> Dict[Tuple[int, int], float]:
        return {(l, j): sum(self.score[(l, r)][j] for r in roles)
                for l in layers for j in range(k)}

    def select_layer(self, layers: Sequence[int], roles: Sequence[str], k: int
                     ) -> Tuple[Dict[int, int], Dict[int, float],
                                Dict[Tuple[int, int], float]]:
        ls = self.layer_score(layers, roles, k)
        win = {l: max(range(k), key=lambda j: ls[(l, j)]) for l in layers}
        conf = {l: ls[(l, win[l])] - max(ls[(l, j)] for j in range(k) if j != win[l])
                for l in layers}
        return win, conf, ls

    def occupancy(self, winner: Dict, k: int) -> List[int]:
        return [sum(1 for v in winner.values() if v == j) for j in range(k)]

    def zero_cells(self) -> int:
        return sum(1 for v in self.score.values() if max(v) <= 0)


def measure(ws: WeightSpace, cfg: Config, cand: Candidates, sel: EvalSet,
            omegas: Optional[np.ndarray] = None, log=print) -> Contribution:
    """eq:perturbed_loss + eq:partial_score.

    omegas
        None       -> omega = 1 (모든 후보에 같은 가중치)
        (K, N)      -> eq:soft_weight. 후보 k 는 omega_k 로 재고,
                       기준 손실도 L(theta; omega_k) 로 후보마다 따로 잡는다.
                       (하나로 평균하면 eq:soft_weight 의 의미가 사라진다.)
    """
    ws.reset()
    if omegas is None:
        base = [loss(ws.model, sel)] * len(cand.names)
    else:
        if omegas.shape[0] != len(cand.names):
            raise SystemExit(
                f"omegas 는 (K, N) 이어야 한다. 받은 {omegas.shape}")
        base = [loss(ws.model, sel, omegas[j]) for j in range(len(cand.names))]

    out = Contribution(base_loss=float(np.mean(base)))
    out.base_per_candidate = list(base)
    t0 = time.time()
    total = len(ws.keys)
    log(f"[기여도] L(theta;omega) 상대 감소율.  A={cfg.alphas}, "
        f"질의 {sel.n}개, 후보 {len(cand.names)}개, "
        f"omega={'1' if omegas is None else 'soft'}")
    for ci, key in enumerate(ws.keys):
        scores, alphas = [], []
        for j, name in enumerate(cand.names):
            w = None if omegas is None else omegas[j]
            best, best_a = None, None
            for a in cfg.alphas:
                with torch.no_grad():
                    ws.lin[key].weight.copy_(
                        torch.add(ws.base[key], cand.get(name, key), alpha=a))
                v = loss(ws.model, sel, w)
                if best is None or v < best:
                    best, best_a = v, a
            scores.append(max(0.0, (base[j] - best) / base[j]))
            alphas.append(best_a)
        ws.reset([key])
        out.score[key], out.alpha[key] = scores, alphas
        if (ci + 1) % 22 == 0:
            el = time.time() - t0
            log(f"  {ci+1}/{total}  {el:.0f}s  "
                f"남은 {el / (ci + 1) * (total - ci - 1):.0f}s")
    return out


def representative(cand: Candidates, con: Contribution,
                   winner: Dict, ws: WeightSpace, by_layer: bool,
                   keep: Optional[set] = None) -> Callable[[Key], torch.Tensor]:
    """eq:representative_update. PartialScore <= 0 인 위치는 0 으로 둔다."""
    def src(key: Key) -> torch.Tensor:
        if by_layer:
            if keep is not None and key[0] not in keep:
                return ws.zeros(key)
            j = winner[key[0]]
        else:
            j = winner[key]
        if con.score[key][j] <= 0:
            return ws.zeros(key)
        return con.alpha[key][j] * cand.by_index(j, key)
    return src


def verify_assembly(ws: WeightSpace, sel: EvalSet, con: Contribution,
                    cand: Candidates, win_layer: Dict[int, int],
                    layer_score: Dict[Tuple[int, int], float],
                    log=print) -> set:
    """eq:assembly_verification.

    개별 기여도는 omega_k 로 쟀더라도 최종 조합은 전체 입력에서 성립해야 하므로
    검증에는 omega = 1 을 쓴다. 기준 손실도 여기서 다시 잰다.
    """
    ws.reset()
    L0 = loss(ws.model, sel)
    src = representative(cand, con, win_layer, ws, by_layer=True)
    ws.apply(src, 1.0)
    L_asm = loss(ws.model, sel)
    ws.reset()
    log(f"[통합 검증] L(theta+z;1) {L_asm:.5f}  vs  L(theta;1) {L0:.5f}  "
        f"{'통과' if L_asm < L0 else '실패 - greedy 복구'}")
    if L_asm < L0:
        return set(ws.layers)

    keep, cur = set(), L0
    for l in sorted(ws.layers, key=lambda x: -layer_score[(x, win_layer[x])]):
        trial = keep | {l}
        ws.apply(representative(cand, con, win_layer, ws, True, trial), 1.0)
        v = loss(ws.model, sel)
        ws.reset()
        if v < cur:
            keep, cur = trial, v
    log(f"  유지 layer {len(keep)}/{len(ws.layers)}   L {cur:.5f}")
    return keep
