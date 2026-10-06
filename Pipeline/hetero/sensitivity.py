# -*- coding: utf-8 -*-
"""eq:behavior_signature, eq:sensitivity_profile.

    s_{k,i,rho}(x_n) = || d overline{log p}_k(x_n) / d w_{k,i,rho} ||_F^2

질의마다 역전파를 따로 하면 |X| 번이다. 선형 계층 y = W a 에서 질의 n 의 gradient 는
G_n = sum_t g_t a_t^T 이므로

    || G_n ||_F^2 = sum_{t,t'} (a_t . a_t') (g_t . g_t')

이다. 토큰 Gram 행렬 두 개(T x T)의 원소곱 합이라 d_out x d_in 행렬을 만들지 않고,
배치 역전파 한 번에 질의별 값이 나온다. 손실은 배치 안 overline{log p} 의 **합**으로
둬야 질의별 gradient 가 섞이지 않는다.

가중치에는 gradient 를 만들지 않는다. 임베딩 출력만 requires_grad 로 열어 activation
gradient 만 흐르게 하고, 그 값을 forward hook 이 단 tensor hook 으로 받는다.
합쳐진 행렬(Phi-3)은 출력 gradient 를 역할 경계로 잘라 역할별로 따로 계산한다.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from ..lord import mean_logp, token_logp
from .roles import ROLES, Arch


def profiles(arch: Arch, tok, items: Sequence[dict], bs: int, device: str,
             log=print) -> np.ndarray:
    """(L, 7, |X|) 의 s. 평균 제거는 하지 않는다 (ot.alignment_cost 가 한다)."""
    model = arch.model
    N = len(items)
    out = np.zeros((arch.L, len(ROLES), N), dtype=np.float64)
    cur = {"lo": 0}

    groups = []      # (layer, module, [(role 번호, lo, hi)])
    for i in range(arch.L):
        by = {}
        for ri, r in enumerate(ROLES):
            mod, lo, hi = arch.out_spec(i, r)
            by.setdefault(id(mod), (mod, []))[1].append((ri, lo, hi))
        for mod, roles in by.values():
            groups.append((i, mod, roles))

    def fwd_hook(i, roles):
        def h(mod, args, o):
            a = args[0].detach().float()
            Ga = torch.bmm(a, a.transpose(1, 2))            # (B, T, T)

            def bwd(g):
                g = g.float()
                for ri, lo, hi in roles:
                    gs = g if lo is None else g[..., lo:hi]
                    Gg = torch.bmm(gs, gs.transpose(1, 2))
                    s = (Ga * Gg).sum((1, 2)).double().cpu().numpy()
                    out[i, ri, cur["lo"]:cur["lo"] + len(s)] = s
            o.register_hook(bwd)
        return h

    def open_grad(mod, args, o):
        return o.requires_grad_(True)

    hooks = [mod.register_forward_hook(fwd_hook(i, roles)) for i, mod, roles in groups]
    hooks.append(model.get_input_embeddings().register_forward_hook(open_grad))
    pad = tok.pad_token_id
    try:
        model.eval()
        for s in range(0, N, bs):
            chunk = items[s:s + bs]
            cur["lo"] = s
            with torch.enable_grad():
                lp, m = token_logp(model, [x["pid"] for x in chunk],
                                   [x["gid"] for x in chunk], pad, device)
                mean_logp(lp, m).sum().backward()
            if (s // bs) % 64 == 0:
                log(f"    sensitivity {s + len(chunk)}/{N}")
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return out
