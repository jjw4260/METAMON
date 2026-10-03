# -*- coding: utf-8 -*-
"""이기종 Local 하나의 가중치 공간. `train_lord` 와 `save_delta` 가 그대로 쓴다.

`modeling.WeightSpace` 와 같은 인터페이스(model, device, keys, lin, base, params,
apply, reset, trainable)를 갖되 두 가지가 다르다.

1. 키가 **모듈** 단위다. 역할이 따로 있는 구조는 (i, 'Query') ... (i, 'Down') 7 개,
   Phi-3 는 (i, 'QKV'), (i, 'Output'), (i, 'GateUp'), (i, 'Down') 4 개다. 학습되는
   행렬이 그 모듈이므로 Δw 도 그 단위로 저장한다. 역할 단위로 자르는 것은
   투영 단계에서 `roles.Arch` 가 같은 경계로 한다.
2. BASE 사본을 **CPU** 에 둔다. Phi-3-mini(3.8B)를 fp32 로 전체 학습하면 모델·
   gradient·Adam 만 58GB 이고, BASE 사본까지 GPU 에 두면 80GB 를 넘는다.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM

from .roles import ROLES, Arch

FUSED_NAME = {"Query": "QKV", "Key": "QKV", "Value": "QKV",
              "Gate": "GateUp", "Up": "GateUp"}


def attn_impl(name: str) -> str:
    return "eager" if "gemma" in name.lower() else "sdpa"


class HeteroSpace:
    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base, dtype=torch.float32, attn_implementation=attn_impl(cfg.base)
        ).to(device)
        self.model.config.use_cache = False
        self.model.generation_config.max_length = None
        self.arch = Arch(self.model)

        self.lin: Dict[tuple, torch.nn.Linear] = {}
        for i in range(self.arch.L):
            for r in ROLES:
                mod, lo, _ = self.arch.out_spec(i, r)
                name = r if lo is None else FUSED_NAME[r]
                self.lin[(i, name)] = mod
        self.keys: List[tuple] = list(self.lin.keys())
        self.layers: List[int] = sorted({k[0] for k in self.keys})
        self.base = {k: m.weight.detach().float().cpu().clone()
                     for k, m in self.lin.items()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = [m.weight for m in self.lin.values()]

    @torch.no_grad()
    def apply(self, src, scale: float = 1.0, subset=None) -> None:
        for k in (subset if subset is not None else self.keys):
            w = self.base[k].to(self.device, non_blocking=True)
            if src is None or scale == 0.0:
                self.lin[k].weight.copy_(w)
            else:
                self.lin[k].weight.copy_(torch.add(w, src(k).to(self.device),
                                                   alpha=scale))

    def reset(self, subset=None) -> None:
        self.apply(None, 0.0, subset)

    def trainable(self, flag: bool) -> None:
        for p in self.params:
            p.requires_grad_(flag)
