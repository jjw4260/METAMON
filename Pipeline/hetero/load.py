# -*- coding: utf-8 -*-
"""학습이 끝난 모델 (BASE + Δw) 을 올리고, Δw 를 역할 단위로 꺼낸다.

Δw 파일의 키
    theta  (동종 run 의 lord_all)  "{i}.{rho}"            역할 단위
    Local  (h1_train)              "{i}.{모듈}"            모듈 단위. Phi-3 는 QKV / GateUp

역할로 자를 때는 `roles.Arch.out_spec` 의 출력 축 경계를 그대로 쓴다. 학습할 때
합쳐진 행렬의 행이 곧 그 역할의 출력이므로 같은 경계다.
"""
from __future__ import annotations

from typing import Dict

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .roles import ROLES, Arch
from .space import FUSED_NAME, attn_impl


def module_name(arch: Arch, role: str) -> str:
    _, lo, _ = arch.out_spec(0, role)
    return role if lo is None else FUSED_NAME[role]


def load_tok(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@torch.no_grad()
def load_trained(name: str, delta_path: str, device: str):
    """fp32 로 올리고 Δw 를 더한다. Δw 는 BASE 대비 1e-3 수준이라 bf16 이면 묻힌다."""
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation=attn_impl(name)).to(device).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    arch = Arch(model)
    n = 0
    with safe_open(delta_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for i in range(arch.L):
            seen = set()
            for r in ROLES:
                mod, lo, _ = arch.out_spec(i, r)
                if id(mod) in seen:
                    continue
                seen.add(id(mod))
                k = f"{i}.{module_name(arch, r)}"
                if k not in keys:
                    raise SystemExit(f"{delta_path} 에 {k} 가 없다 (키 예: {sorted(keys)[:3]})")
                mod.weight.add_(f.get_tensor(k).to(device).float())
                n += 1
    return arch, n


class RoleDelta:
    """Local 의 Δw 를 역할 단위 (d_out x d_in) 로 꺼낸다. GPU fp32."""

    def __init__(self, arch: Arch, delta_path: str, device: str):
        self.arch, self.device = arch, device
        self.f = safe_open(delta_path, framework="pt", device="cpu")

    def get(self, i: int, role: str) -> torch.Tensor:
        _, lo, hi = self.arch.out_spec(i, role)
        t = self.f.get_tensor(f"{i}.{module_name(self.arch, role)}")
        if lo is not None:
            t = t[lo:hi]
        return t.to(self.device).float()
