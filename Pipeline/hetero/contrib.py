# -*- coding: utf-8 -*-
"""이기종 3 단계의 측정과 선택.

  eq:norm_matching          z~_{k,l,rho} = Δz_{k,l,rho} * ||w_theta,l,rho||_F / ||Δz_{k,l,rho}||_F
  eq:perturbed_loss         한 (l, rho) 에 a * z~ 만 얹은 L(theta; omega)
  eq:partial_score          PartialScore = [ (L0 - min_{a in A} L) / L0 ]_+ ,  Scale = argmin a
  eq:layer_selection        LayerScore_{l,k} = sum_rho PartialScore,  k_l = argmax,
                            Confidence_l = 1 위 - 2 위
  eq:representative_update  z_{l,rho} = Scale * z~_{k_l,l,rho}   (PartialScore > 0 인 곳만)
  eq:assembly_verification  L(theta + z; 1) < L(theta; 1). 실패하면 LayerScore 순 greedy

측정은 질의별 mean log p 를 그대로 남긴다 (lp[l, rho, k, a, n]). omega(균등 / soft,
beta) 와 LOO 선택은 이 배열에서 다시 계산할 뿐 모델을 다시 돌리지 않는다.

층 l 만 바꾸면 l 이전의 hidden state 는 그대로다. sel 질의의 층별 입력을 한 번
캐시하고, 측정 때는 l 번째 층부터만 forward 한다. 캐시 경로와 전체 forward 가
같은 값을 내는지는 `check` 가 확인한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..metrics import EvalSet, _chunk


class TailEval:
    """sel 위에서 질의별 mean log p. 층 l 부터만 forward 할 수 있다."""

    def __init__(self, arch, tok, items, bs: int, device: str):
        self.arch, self.device = arch, device
        self.es = EvalSet(items, tok, device, bs)
        self.n = self.es.n
        self.H: List[List[torch.Tensor]] = []
        self.ch = None

    @torch.inference_mode()
    def cache(self) -> None:
        """현재 가중치에서 층별 입력 hidden state 를 모은다."""
        layers = self.arch.layers
        self.H = []
        cur: Dict[int, torch.Tensor] = {}

        def mk(i):
            def pre(mod, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                cur[i] = h.detach().clone()
            return pre
        hs = [layers[i].register_forward_pre_hook(mk(i), with_kwargs=True)
              for i in range(len(layers))]
        try:
            for inp, att, msk, ntk, idx in self.es.batches:
                cur.clear()
                self.arch.model(input_ids=inp, attention_mask=att)
                self.H.append([cur[i] for i in range(len(layers))])
                if self.ch is None:
                    self.ch = _chunk(self.es, self.arch.model.config.vocab_size, inp.shape[1])
        finally:
            for h in hs:
                h.remove()

    @torch.inference_mode()
    def mlp(self, start: Optional[int] = None) -> np.ndarray:
        """질의별 mean log p (eq:mean_log_probability). start 가 있으면 그 층부터."""
        mm = self.arch.model.model
        full = mm.layers
        if start is not None:
            if not self.H:
                raise RuntimeError("cache() 를 먼저 부른다")
            mm.layers = torch.nn.ModuleList(list(full)[start:])
        out = np.empty(self.n)
        try:
            for b, (inp, att, msk, ntk, idx) in enumerate(self.es.batches):
                ch = self.ch or 8
                rows = []
                for s0 in range(0, inp.shape[0], ch):
                    s1 = min(s0 + ch, inp.shape[0])
                    if start is None:
                        z = self.arch.model(input_ids=inp[s0:s1],
                                            attention_mask=att[s0:s1]).logits[:, :-1]
                    else:
                        z = self.arch.model(inputs_embeds=self.H[b][start][s0:s1],
                                            attention_mask=att[s0:s1]).logits[:, :-1]
                    z = z.float()
                    g = z.gather(-1, inp[s0:s1, 1:].unsqueeze(-1)).squeeze(-1)
                    rows.append(g - torch.logsumexp(z, -1))
                    del z, g
                lp = torch.cat(rows, 0)
                out[idx] = ((lp * msk[:, 1:]).sum(1) / ntk).double().cpu().numpy()
        finally:
            mm.layers = full
        return out

    def check(self, w_of, log=print) -> None:
        """캐시 경로 == 전체 forward. 섭동 없이, 그리고 중간 층 하나를 섭동해서."""
        ref = self.mlp()
        worst = 0.0
        for l in (0, self.arch.L // 2, self.arch.L - 1):
            worst = max(worst, float(np.abs(self.mlp(l) - ref).max()))
        l = self.arch.L // 2
        W = w_of(l)
        W0 = W.detach().clone()
        with torch.no_grad():
            W.add_(torch.randn_like(W) * (W0.norm() / W0.numel() ** .5) * 1e-2)
        try:
            a, b = self.mlp(l), self.mlp()
            moved = float(np.abs(b - ref).max())
            worst = max(worst, float(np.abs(a - b).max()))
        finally:
            with torch.no_grad():
                W.copy_(W0)
        if worst > 1e-4 or moved < 1e-4:
            raise SystemExit(f"층별 캐시 자기검사 실패: 최대 차 {worst:.2e}, 섭동 반응 {moved:.2e}")
        log(f"    층별 캐시 자기검사 통과 (최대 차 {worst:.1e}, 섭동 반응 {moved:.1e})")


# ------------------------------------------------------------------ 선택 (numpy)
def weights(K: int, N: int, omegas: Optional[np.ndarray]) -> np.ndarray:
    return np.ones((K, N)) if omegas is None else np.asarray(omegas, dtype=np.float64)


def partial_scores(lp: np.ndarray, lp0: np.ndarray, omegas: Optional[np.ndarray]):
    """lp (L, R, K, A, N), lp0 (N,).  nan 은 그 후보에 Δz 가 없는 칸.

    반환 ps (L, R, K), ai (L, R, K) Scale 번호, base (K,) = L(theta; omega_k)"""
    L, R, K, A, N = lp.shape
    w = weights(K, N, omegas)
    base = -(w * lp0[None]).sum(1) / w.sum(1)
    with np.errstate(invalid="ignore"):
        loss = -(lp * w[None, None, :, None, :]).sum(-1) / w.sum(1)[None, None, :, None]
    loss = np.where(np.isfinite(loss), loss, np.inf)
    best, ai = loss.min(-1), loss.argmin(-1)
    ps = np.maximum(0.0, (base[None, None] - best) / base[None, None])
    ps = np.where(np.isfinite(best), ps, 0.0)
    return ps, ai, base


def select_layers(ps: np.ndarray, ks: Sequence[int]):
    """eq:layer_selection. ks 는 쓰는 후보 번호 (LOO 면 하나를 뺀 것)."""
    ks = list(ks)
    LS = ps[:, :, ks].sum(1)                         # (L, K')
    order = np.argsort(-LS, axis=1)
    win = np.asarray(ks)[order[:, 0]]
    top1 = LS[np.arange(len(LS)), order[:, 0]]
    top2 = LS[np.arange(len(LS)), order[:, 1]] if len(ks) > 1 else np.zeros(len(LS))
    LSfull = np.full((ps.shape[0], ps.shape[2]), np.nan)
    LSfull[:, ks] = LS
    return win, top1 - top2, top1, LSfull
