# -*- coding: utf-8 -*-
"""fleet 준비: Local K 개 + 전체 질의 단일 모델.

  - 이미 있는 arm 은 건너뛴다
  - arm 마다 학습 직후 생존 관문을 통과해야 다음으로 넘어간다
    (배율 1.0 또는 0.5 중 하나에서 L(theta;1) 이 BASE 보다 작아야 한다)
  - 저장 가중치를 다시 읽어 질의별 log-probability 가 일치하는지 검증한다
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .data import Splits
from .lord import train_lord
from .metrics import EvalSet, loss, token_avg_logp
from .modeling import Key, WeightSpace, ckpt_path
from .sft import train_sft


def build_fleet(ws: WeightSpace, tok, cfg: Config, sp: Splits,
                sel: EvalSet, log=print) -> None:
    trainer = train_lord if cfg.fleet_method == "lord" else train_sft
    ws.reset()
    base_loss = loss(ws.model, sel)
    log(f"[fleet] {cfg.fleet_method}.  base L(theta;1) = {base_loss:.5f}")

    def health() -> float:
        return loss(ws.model, sel)

    for name in cfg.arm_names:
        if os.path.exists(ckpt_path(cfg, name)):
            log(f"  {name} 건너뜀")
            continue
        idx, seed = _assignment(cfg, sp, name)
        data = sp.train if idx is None else [sp.train[i] for i in idx]
        log(f"  {name} 학습 질의 {len(data)}")
        trainer(ws, tok, cfg, name, data, seed, health=health, log=log)
        _gate(ws, cfg, sel, name, base_loss, log)
    ws.reset()


def _assignment(cfg: Config, sp: Splits, name: str):
    """arm 이름 -> (학습 질의 index, 시드). index 가 None 이면 전체 질의."""
    if name == cfg.all_name:
        return None, cfg.seed
    if name in cfg.local_names:
        i = cfg.local_names.index(name)
        return sp.shard[name], cfg.seed + 1 + i
    if name in cfg.union_names:
        g = cfg.union_names.index(name)
        idx: List[int] = []
        for i in cfg.fleets[g]:
            idx += sp.shard[cfg.local_names[i]]
        # union_g 는 fleet g 의 병합과 본 데이터가 같아야 한다. 시드는 따로 준다.
        return idx, cfg.seed + 101 + g
    raise SystemExit(f"알 수 없는 arm: {name}")


def _gate(ws: WeightSpace, cfg: Config, sel: EvalSet, name: str,
          base_loss: float, log) -> None:
    state = torch.load(ckpt_path(cfg, name), map_location="cpu")
    delta = {k: state[k].float() - ws.base[k].cpu() for k in ws.keys}
    best = None
    for s in (1.0, 0.5):
        ws.apply(lambda k: delta[k], s)
        v = loss(ws.model, sel)
        best = v if best is None else min(best, v)
    ws.reset()
    log(f"  {name} 생존 L@1.0/0.5 최소 {best:.5f}  vs base {base_loss:.5f}  "
        f"{'통과' if best < base_loss else '실패'}")
    if best >= base_loss:
        raise SystemExit(
            f"{name}: 학습이 BASE 를 개선하지 못했다. 다음 arm 으로 넘어가지 않는다.")


def load_deltas(ws: WeightSpace, cfg: Config, log=print
                ) -> Dict[str, Dict[Key, torch.Tensor]]:
    """저장 가중치 - BASE. CPU 에 둔다(메모리)."""
    out: Dict[str, Dict[Key, torch.Tensor]] = {}
    base_cpu = {k: v.cpu() for k, v in ws.base.items()}
    frob = math.sqrt(sum(float((v ** 2).sum()) for v in base_cpu.values()))
    log(f"[적재] {'arm':12s} {'유한':>5s} {'상대Δ':>11s}")
    for name in cfg.arm_names:
        st = torch.load(ckpt_path(cfg, name), map_location="cpu")
        if set(st.keys()) != set(ws.keys):
            raise SystemExit(f"{name}: key 불일치")
        if not all(torch.isfinite(v).all().item() for v in st.values()):
            raise SystemExit(f"{name}: NaN/Inf 가 저장돼 있다")
        dw = {k: st[k].float() - base_cpu[k] for k in ws.keys}
        rel = math.sqrt(sum(float((v ** 2).sum()) for v in dw.values())) / frob
        log(f"  {name:12s} {'True':>5s} {rel:11.3e}")
        out[name] = dw
        del st
    return out


def verify_restore(ws: WeightSpace, cfg: Config,
                   deltas: Dict[str, Dict[Key, torch.Tensor]],
                   sel: EvalSet, log=print) -> None:
    """저장 가중치 직접 적재 vs BASE + Δw 가 같은 결과를 내는지 확인한다."""
    ok_all = True
    for name, dw in deltas.items():
        st = torch.load(ckpt_path(cfg, name), map_location="cpu")
        ws.load_state(st)
        a = token_avg_logp(ws.model, sel)
        ws.apply(lambda k: dw[k].to(ws.device), 1.0)
        b = token_avg_logp(ws.model, sel)
        ws.reset()
        qm = float(np.max(np.abs(a - b)))
        ok = qm < 1e-4
        ok_all &= ok
        log(f"  {name:10s} 질의별 최대 차이 {qm:.3e}  {'통과' if ok else '실패'}")
        del st
    if not ok_all:
        raise SystemExit("복원 불일치. 성능 판정을 진행하지 않는다.")
