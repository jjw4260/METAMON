# -*- coding: utf-8 -*-
"""구조가 다른 Transformer 의 선형 계층을 7 역할 rho 로 맞춘다.

    rho in {Query, Key, Value, Output, Gate, Up, Down}

Llama / Qwen2 / Gemma2 / SmolLM2 는 역할마다 nn.Linear 가 따로 있다.
Phi-3 는 둘이 합쳐져 있다.
    self_attn.qkv_proj      출력 = [Query ; Key ; Value]
    mlp.gate_up_proj        출력 = [Gate ; Up]          (Phi3MLP 가 chunk(2) 로 나눈다)
합쳐진 행렬은 **출력 축**으로 잘라 역할 하나씩으로 본다. 가중치·출력 activation·
Δw 를 모두 같은 경계로 자르므로 한 역할의 (입력, 가중치, 출력) 관계가 유지된다.
`check_roles` 가 실제 모델에서 out_slice == in @ W_slice^T 를 확인한다.

입력은 역할이 아니라 **입력 종류**로 묶인다. 같은 입력을 보는 역할은 InputMap 을
공유한다.
    attn_in   Query, Key, Value     (attention 앞 정규화 출력)
    o_in      Output                (head 출력을 이어 붙인 것)
    mlp_in    Gate, Up              (MLP 앞 정규화 출력)
    down_in   Down                  (Gate·Up 을 지난 중간 표현)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch

ROLES = ("Query", "Key", "Value", "Output", "Gate", "Up", "Down")
IN_KIND = {"Query": "attn_in", "Key": "attn_in", "Value": "attn_in",
           "Output": "o_in", "Gate": "mlp_in", "Up": "mlp_in", "Down": "down_in"}
KINDS = ("attn_in", "o_in", "mlp_in", "down_in")


class StopForward(Exception):
    pass


class Arch:
    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.L = len(self.layers)
        cfg = model.config
        a0, m0 = self.layers[0].self_attn, self.layers[0].mlp
        self.fused_qkv = hasattr(a0, "qkv_proj")
        self.fused_gu = hasattr(m0, "gate_up_proj")
        n_h = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_h
        if self.fused_qkv:
            op = a0.qkv_proj.out_features
            hd = op // (n_h + 2 * n_kv)
            if hd * (n_h + 2 * n_kv) != op:
                raise SystemExit(f"qkv_proj 출력 {op} 를 head 로 나눌 수 없다")
            self.q_dim, self.kv_dim = n_h * hd, n_kv * hd
        else:
            self.q_dim = a0.q_proj.out_features
            self.kv_dim = a0.k_proj.out_features
        if self.fused_gu:
            self.inter = m0.gate_up_proj.out_features // 2
        else:
            self.inter = m0.gate_proj.out_features

    # ---------------------------------------------------------------- 위치
    def out_spec(self, l: int, role: str) -> Tuple[torch.nn.Module, Optional[int], Optional[int]]:
        """(모듈, 출력 시작, 출력 끝). 끝이 None 이면 전체."""
        a, m = self.layers[l].self_attn, self.layers[l].mlp
        q, kv = self.q_dim, self.kv_dim
        if role in ("Query", "Key", "Value"):
            if self.fused_qkv:
                lo = {"Query": 0, "Key": q, "Value": q + kv}[role]
                hi = {"Query": q, "Key": q + kv, "Value": q + 2 * kv}[role]
                return a.qkv_proj, lo, hi
            return {"Query": a.q_proj, "Key": a.k_proj, "Value": a.v_proj}[role], None, None
        if role == "Output":
            return a.o_proj, None, None
        if role in ("Gate", "Up"):
            if self.fused_gu:
                n = self.inter
                return (m.gate_up_proj, 0, n) if role == "Gate" else (m.gate_up_proj, n, 2 * n)
            return (m.gate_proj if role == "Gate" else m.up_proj), None, None
        if role == "Down":
            return m.down_proj, None, None
        raise KeyError(role)

    def in_module(self, l: int, kind: str) -> torch.nn.Module:
        role = {"attn_in": "Query", "o_in": "Output", "mlp_in": "Gate",
                "down_in": "Down"}[kind]
        return self.out_spec(l, role)[0]

    def weight(self, l: int, role: str) -> torch.Tensor:
        """역할의 가중치 행렬 (d_out x d_in). 합쳐진 행렬은 행으로 자른 view."""
        mod, lo, hi = self.out_spec(l, role)
        w = mod.weight
        return w if lo is None else w[lo:hi]

    def out_dim(self, role: str) -> int:
        return self.weight(0, role).shape[0]

    def in_dim(self, kind: str) -> int:
        return self.in_module(0, kind).in_features

    def describe(self) -> str:
        return (f"L={self.L}  hidden={self.in_dim('attn_in')}  q={self.q_dim}  "
                f"kv={self.kv_dim}  inter={self.inter}  "
                f"{'qkv합침 ' if self.fused_qkv else ''}"
                f"{'gate_up합침' if self.fused_gu else ''}")

    # ---------------------------------------------------------------- 채집
    @contextmanager
    def capture(self, ins: Iterable[Tuple[int, str]], outs: Iterable[Tuple[int, str]],
                stop_after: Optional[int] = None):
        """forward 동안 필요한 activation 만 모은다.

        ins   [(layer, kind)]     선형 계층의 입력
        outs  [(layer, role)]     선형 계층의 출력 (합쳐진 것은 잘라서)
        stop_after                이 layer 가 끝나면 forward 를 멈춘다 (뒤는 안 쓴다)
        """
        store: Dict[tuple, torch.Tensor] = {}
        hooks = []
        need_in: Dict[int, List[Tuple[int, str]]] = {}
        need_out: Dict[int, List[Tuple[int, str, Optional[int], Optional[int]]]] = {}
        for l, k in ins:
            mod = self.in_module(l, k)
            need_in.setdefault(id(mod), []).append((l, k))
        for l, r in outs:
            mod, lo, hi = self.out_spec(l, r)
            need_out.setdefault(id(mod), []).append((l, r, lo, hi))
        mods = {}
        for l, k in ins:
            mods[id(self.in_module(l, k))] = self.in_module(l, k)
        for l, r in outs:
            mods[id(self.out_spec(l, r)[0])] = self.out_spec(l, r)[0]

        def mk(mid):
            def hook(mod, args, out):
                for l, k in need_in.get(mid, []):
                    store[("in", l, k)] = args[0].detach()
                for l, r, lo, hi in need_out.get(mid, []):
                    o = out.detach()
                    store[("out", l, r)] = o if lo is None else o[..., lo:hi]
            return hook

        for mid, mod in mods.items():
            hooks.append(mod.register_forward_hook(mk(mid)))
        if stop_after is not None:
            def stop(mod, args, out):
                raise StopForward()
            hooks.append(self.layers[stop_after].register_forward_hook(stop))
        try:
            yield store
        finally:
            for h in hooks:
                h.remove()


@torch.no_grad()
def check_roles(arch: Arch, input_ids: torch.Tensor, attn: torch.Tensor,
                layers: Optional[List[int]] = None, tol: float = 2e-2) -> None:
    """모든 역할에서 출력 == 입력 @ W^T 인지 본다. 역할을 잘못 자르면 여기서 걸린다."""
    layers = layers or [0, arch.L - 1]
    ins = [(l, k) for l in layers for k in KINDS]
    outs = [(l, r) for l in layers for r in ROLES]
    with arch.capture(ins, outs, stop_after=max(layers)) as st:
        try:
            arch.model(input_ids=input_ids, attention_mask=attn)
        except StopForward:
            pass
    for l in layers:
        for r in ROLES:
            x = st[("in", l, IN_KIND[r])].float()
            w = arch.weight(l, r).float()
            mod, _, _ = arch.out_spec(l, r)
            y = x @ w.T
            if getattr(mod, "bias", None) is not None:
                lo = arch.out_spec(l, r)[1] or 0
                y = y + mod.bias.float()[lo:lo + w.shape[0]]
            o = st[("out", l, r)].float()
            rel = float((o - y).norm() / (o.norm() + 1e-6))
            if rel > tol:
                raise SystemExit(f"역할 분해 오류: layer {l} {r}  상대오차 {rel:.3e}")
