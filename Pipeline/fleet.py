# -*- coding: utf-8 -*-
"""fleet 준비: Local K 개 + 전체 질의 단일 모델.

  - 이미 있는 arm 은 건너뛴다
  - arm 마다 학습 직후 생존 관문을 재고 **기록한다**. 못 넘어도 멈추지 않는다
    (배율 1.0 또는 0.5 중 하나에서 L(theta;1) 이 BASE 보다 작아야 통과)
    저장되는 가중치가 sel 최적 시점이므로, 그래도 못 넘는 arm 은 학습이 안 붙은
    arm 이고 그것 자체가 결과다. 4 개가 못 넘으면 학습 설정이 틀린 것이라 멈춘다
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
from .deltastore import DeltaStore, st_path
from .modeling import Key, WeightSpace
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
        if os.path.exists(st_path(cfg, name)):
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
                f"arm {len(failed)}개가 기준선을 못 넘었다: {failed}\n"
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
    설계의 결과다. `own` 이 없으면(union / all / IID local) 전체 혼합이 기준이다.
    어느 쪽이든 판정은 기록이고, 여기서 run 을 멈추지 않는다.
    """
    st = DeltaStore(ws, cfg, [name], log=lambda *_: None)
    src = lambda k: st.raw(name, k)
    # **판정은 배율 1.0 에서만 한다.** 실행 3 의 single(arm) 이 쓰는 배율이
    # 1.0 이다. 예전에는 1.0 과 0.5 중 더 좋은 쪽으로 판정해서, 13 arm 전부가
    # 1.0 에서 BASE 보다 나쁜데도 전부 "통과" 로 찍혔다. 0.5 / 0.25 는 참고로만
    # 남긴다 (Δw 가 방향은 맞고 크기만 큰지 보는 눈).
    scales = (1.0, 0.5, 0.25)
    v, v_own = {}, {}
    for s in scales:
        ws.apply(src, s)
        v[s] = loss(ws.model, sel)
        if own is not None:
            v_own[s] = loss(ws.model, own)
    ws.reset()
    st.close()
    ref = "  ".join(f"@{s}={v[s]:.5f}" for s in scales)

    if own is not None:
        ok = v_own[1.0] < base_own
        log(f"  {name} 생존 자기조각 @1.0 {v_own[1.0]:.5f} vs base "
            f"{base_own:.5f}  {'통과' if ok else '실패'}   "
            f"(자기조각 " + "  ".join(f"@{s}={v_own[s]:.5f}" for s in scales)
            + f" / 전체 sel {ref} vs {base_loss:.5f})")
        if not ok:
            log(f"    *** 자기 조각에서도 배율 1.0 이 BASE 를 못 넘었다.")
        return ok

    ok = v[1.0] < base_loss
    log(f"  {name} 생존 @1.0 {v[1.0]:.5f}  vs base {base_loss:.5f}  "
        f"{'통과' if ok else '실패'}   (sel {ref})")
    if not ok:
        # 예전에는 여기서 멈췄다. arm 하나가 6 시간 run 을 죽였다. 저장되는 것이
        # 이제 마지막 상태가 아니라 sel 최적 시점이므로, 그래도 BASE 를 못 넘는
        # arm 은 "학습이 안 붙은 arm" 이고 그 사실 자체가 결과다. 기록만 하고
        # 남은 arm 을 학습한다. 조립에서는 기여도가 0 에 가까울 것이다.
        log(f"    *** 전체 질의로 학습했는데 BASE 를 못 넘었다. 기록만 하고 "
            f"다음 arm 으로 간다.")
    return ok


def open_store(ws: WeightSpace, cfg: Config, log=print) -> DeltaStore:
    """Δw 저장소를 연다. 디스크에 두고 칸 단위로 읽는다.

    예전 `load_deltas` 는 arm 전부를 CPU 에 올렸다. 3B 21 arm 이면 237GB 라
    Colab 에서 시작도 못 한다. 여기서는 열기만 하고 상주 메모리는 칸 하나 분이다.
    """
    store = DeltaStore(ws, cfg, cfg.arm_names, log=log)
    log(f"[적재] {'arm':12s} {'유한':>5s} {'상대Δ':>11s}")
    for name in cfg.arm_names:
        if not store.finite(name):
            raise SystemExit(f"{name}: NaN/Inf 가 저장돼 있다")
        log(f"  {name:12s} {'True':>5s} {store.rel_delta(name):11.3e}")
    store.release()
    return store


def verify_restore(ws: WeightSpace, cfg: Config, store: DeltaStore,
                   sel: EvalSet, log=print) -> None:
    """BASE + Δw 를 적용한 것과, 저장된 Δw 를 그대로 쓴 것이 같은지 본다.

    Δw 를 fp16 으로 저장하므로 여기서 걸리는 것이 그 손실이다. 질의별
    log-probability 차이가 1e-4 를 넘으면 delta_dtype 을 float32 로 올린다.
    """
    ok_all = True
    for name in cfg.arm_names:
        ws.load_state({k: store.state(name, k) for k in ws.keys})
        a = token_avg_logp(ws.model, sel)
        ws.apply(lambda k: store.raw(name, k), 1.0)
        b = token_avg_logp(ws.model, sel)
        ws.reset()
        qm = float(np.max(np.abs(a - b)))
        ok = qm < 1e-4
        ok_all &= ok
        log(f"  {name:10s} 질의별 최대 차이 {qm:.3e}  {'통과' if ok else '실패'}")
    store.release()
    if not ok_all:
        raise SystemExit(
            "복원 불일치. cfg.delta_dtype 을 'float32' 로 올리고 다시 저장할 것.")
