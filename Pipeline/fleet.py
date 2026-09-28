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
    iid = cfg.shard in ("iid", "disjoint", "bootstrap")
    if not iid:
        log(f"  조각이 비-IID({cfg.shard}) 다. local 의 생존은 **자기 조각**에서")
        log(f"  본다. 전체 혼합 sel 에서 못 오르는 것은 정상이고 기록만 한다.")
    failed: List[str] = []

    def health() -> float:
        return loss(ws.model, sel)

    for name in cfg.arm_names:
        if os.path.exists(ckpt_path(cfg, name)):
            log(f"  {name} 건너뜀")
            continue
        idx, seed = _assignment(cfg, sp, name)
        data = sp.train if idx is None else [sp.train[i] for i in idx]
        log(f"  {name} 학습 질의 {len(data)}")

        # 비-IID local 은 자기 조각에서 재는 것이 맞는 건강 검사다. 좁게 배운
        # 모델이 전체 혼합에서 안 오르는 것은 고장이 아니라 그 설계의 결과다.
        own = None
        if not iid and name in cfg.local_names:
            own = EvalSet(data, tok, ws.device, cfg.eval_bs)
            ws.reset()
            base_own = loss(ws.model, own)

        trainer(ws, tok, cfg, name, data, seed, health=health, log=log)
        ok = _gate(ws, cfg, sel, name, base_loss, log,
                   own=own, base_own=base_own if own is not None else None)
        if not ok:
            failed.append(name)
        if len(failed) >= 4:
            raise SystemExit(
                f"local {len(failed)}개가 자기 조각에서도 개선이 없다: {failed}\n"
                f"학습 자체가 고장났다. lord_variant / lord_lr / period_chunk 를 볼 것.")
    if failed:
        log(f"\n  *** 생존 관문을 못 넘은 arm {failed}. 조립에는 들어가지만")
        log(f"      기여도 점유가 0 에 가까울 것이다.")
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
          base_loss: float, log, own: EvalSet | None = None,
          base_own: float | None = None) -> bool:
    """생존 관문. 통과 여부를 돌려준다.

    `own` 이 있으면(비-IID local) **자기 조각**이 판정 기준이고 전체 혼합 sel
    은 참고로만 찍는다. 좁게 배운 모델이 전체에서 안 오르는 것은 고장이 아니라
    설계의 결과다. `own` 이 없으면(union / all / IID local) 전체 혼합이 기준이며
    거기서 못 오르면 학습이 고장난 것이므로 그 자리에서 멈춘다.
    """
    state = torch.load(ckpt_path(cfg, name), map_location="cpu")
    delta = {k: state[k].float() - ws.base[k].cpu() for k in ws.keys}
    best, best_own = None, None
    for s in (1.0, 0.5):
        ws.apply(lambda k: delta[k], s)
        v = loss(ws.model, sel)
        best = v if best is None else min(best, v)
        if own is not None:
            w = loss(ws.model, own)
            best_own = w if best_own is None else min(best_own, w)
    ws.reset()

    if own is not None:
        ok = best_own < base_own
        log(f"  {name} 생존 자기조각 {best_own:.5f} vs base {base_own:.5f}  "
            f"{'통과' if ok else '실패'}   "
            f"(전체 sel {best:.5f} vs {base_loss:.5f} "
            f"{'+' if best < base_loss else '-'})")
        if not ok:
            log(f"    *** 자기 조각에서도 개선이 없다. 학습이 안 붙었다.")
        return ok

    ok = best < base_loss
    log(f"  {name} 생존 L@1.0/0.5 최소 {best:.5f}  vs base {base_loss:.5f}  "
        f"{'통과' if ok else '실패'}")
    if not ok:
        raise SystemExit(
            f"{name}: 전체 질의로 학습했는데 BASE 를 개선하지 못했다. "
            f"학습이 고장났다. 다음 arm 으로 넘어가지 않는다.")
    return True


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
