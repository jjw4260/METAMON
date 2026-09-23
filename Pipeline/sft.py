# -*- coding: utf-8 -*-
"""SFT fleet. LoRD 대조군이자 병합 기전만 따로 보려 할 때 쓰는 안전한 경로.

Target 응답에 대한 토큰 평균 음의 log-probability 를 최소화한다.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Optional, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from .config import Config
from .lord import token_logp
from .modeling import WeightSpace, ckpt_path, save_atomic


def train_sft(ws: WeightSpace, tok, cfg: Config, name: str,
              data: Sequence[dict], seed: int,
              health: Optional[Callable[[], float]] = None,
              log=print) -> None:
    pad = tok.pad_token_id
    ws.reset()
    ws.trainable(True)
    opt = torch.optim.AdamW(ws.params, lr=cfg.sft_lr, foreach=True)
    jl = open(os.path.join(cfg.log_dir, f"{name}.jsonl"), "a", encoding="utf-8")
    upd = 0
    ws.model.train()
    try:
        for e in range(cfg.sft_epochs):
            order = list(range(len(data)))
            random.Random(seed + e).shuffle(order)
            for s in range(0, len(order) - cfg.acc + 1, cfg.acc):
                sl = order[s:s + cfg.acc]
                lp, m = token_logp(ws.model, [data[j]["pid"] for j in sl],
                                   [data[j]["gid"] for j in sl], pad, ws.device)
                L = -((lp * m).sum(1) / m.sum(1).clamp(min=1)).mean()
                if not torch.isfinite(L):
                    raise SystemExit(f"{name}: 비정상 손실")
                opt.zero_grad(set_to_none=True)
                L.backward()
                gn = float(clip_grad_norm_(ws.params, cfg.grad_clip))
                opt.step()
                upd += 1
                jl.write(json.dumps({"epoch": e, "update": upd,
                                     "nll": float(L.detach()), "gnorm": gn}) + "\n")
                if upd % 32 == 0:
                    jl.flush()
                    log(f"    [{name}] e{e+1} u{upd} nll {float(L.detach()):.4f} "
                        f"gnorm {gn:.2f}")
            if health is not None:
                ws.model.eval()
                log(f"  [{name}] e{e+1} 상태 L(theta;1) = {health():.5f}")
                ws.model.train()
    finally:
        jl.close()
        ws.model.eval()
        ws.trainable(False)
        del opt
        torch.cuda.empty_cache()

    save_atomic(ws.snapshot(), ckpt_path(cfg, name))
