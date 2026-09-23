# -*- coding: utf-8 -*-
"""모델과 가중치 공간.

METAMON 의 조립 대상은 Transformer block 의 학습 가능한 선형 계층 7 종이다.
    rho in {Query, Key, Value, Output, Gate, Up, Down}
한 인스턴스만 GPU 에 두고 7 종 가중치만 갈아끼우며 모든 arm 을 평가한다.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config

Key = Tuple[int, str]
Source = Callable[[Key], torch.Tensor]


def setup_precision() -> str:
    """진단은 FP32. TF32 는 끈다."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # 손실 계산용. 생성 시에만 left 로 바꾼다
    return tok


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.mem_get_info()[0] / 1e9


class WeightSpace:
    """BASE 가중치를 기준으로 7 종 역할 가중치를 적용/복원한다."""

    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base, dtype=torch.float32, attn_implementation="sdpa").to(device)
        self.model.config.use_cache = False
        self.model.generation_config.max_length = None

        self.lin = self._linears(self.model)
        self.keys: List[Key] = list(self.lin.keys())
        self.layers: List[int] = sorted({k[0] for k in self.keys})
        self.base = {k: m.weight.detach().clone().float()
                     for k, m in self.lin.items()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = [m.weight for m in self.lin.values()]
        self._zeros: Dict[tuple, torch.Tensor] = {}     # 모양 7 종만 캐시

    @staticmethod
    def _linears(model) -> Dict[Key, torch.nn.Linear]:
        d: Dict[Key, torch.nn.Linear] = {}
        for i, layer in enumerate(model.model.layers):
            d[(i, "Query")] = layer.self_attn.q_proj
            d[(i, "Key")] = layer.self_attn.k_proj
            d[(i, "Value")] = layer.self_attn.v_proj
            d[(i, "Output")] = layer.self_attn.o_proj
            d[(i, "Gate")] = layer.mlp.gate_proj
            d[(i, "Up")] = layer.mlp.up_proj
            d[(i, "Down")] = layer.mlp.down_proj
        return d

    def zeros(self, key: Key) -> torch.Tensor:
        shape = tuple(self.base[key].shape)
        if shape not in self._zeros:
            self._zeros[shape] = torch.zeros(
                shape, device=self.device, dtype=torch.float32)
        return self._zeros[shape]

    @torch.no_grad()
    def apply(self, src: Optional[Source], scale: float = 1.0,
              subset: Optional[List[Key]] = None) -> None:
        for k in (subset if subset is not None else self.keys):
            w = self.base[k]
            if src is None or scale == 0.0:
                self.lin[k].weight.copy_(w)
            else:
                self.lin[k].weight.copy_(torch.add(w, src(k).to(self.device),
                                                   alpha=scale))

    def reset(self, subset: Optional[List[Key]] = None) -> None:
        self.apply(None, 0.0, subset)

    @torch.no_grad()
    def load_state(self, state: Dict[Key, torch.Tensor]) -> None:
        for k in self.keys:
            self.lin[k].weight.copy_(state[k].to(self.device).float())

    def snapshot(self) -> Dict[Key, torch.Tensor]:
        return {k: self.lin[k].weight.detach().float().cpu().clone()
                for k in self.keys}

    def trainable(self, flag: bool) -> None:
        for p in self.params:
            p.requires_grad_(flag)


def save_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def ckpt_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.ckpt_dir, f"{name}.pt")
