# -*- coding: utf-8 -*-
"""배율 결정과 비교.

  - 배율은 check 에서만 고른다. 고른 뒤 test 를 한 번 잰다.
  - 모든 arm 에 같은 격자를 준다.
  - 격자 상한을 고른 arm 이 있으면 경고한다(최적점을 지나지 못한 것).
  - 비교는 질의별 값에 대한 paired bootstrap 으로 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .metrics import EvalSet, paired_bootstrap, sim, verdict
from .modeling import Key, WeightSpace

Source = Callable[[Key], torch.Tensor]


@dataclass
class ArmResult:
    scale: float
    check: float
    test: np.ndarray
    curve: List[Tuple[float, float]]

    @property
    def test_mean(self) -> float:
        return float(np.mean(self.test))


def scale_curve(ws: WeightSpace, cfg: Config, src: Source, tag: str,
                check: EvalSet, test: EvalSet,
                base_check: np.ndarray, base_test: np.ndarray,
                log=print) -> ArmResult:
    best_s, best_v, rows = 0.0, float(np.mean(base_check)), []
    for s in cfg.scales:
        if s == 0.0:
            v = base_check
        else:
            ws.apply(src, s)
            v = sim(ws.model, check)
        m = float(np.mean(v))
        rows.append((s, m))
        if m > best_v:
            best_v, best_s = m, s
    if best_s == 0.0:
        q = base_test
    else:
        ws.apply(src, best_s)
        q = sim(ws.model, test)
    ws.reset()
    log(f"  {tag:15s} 배율 {best_s:<7} check {best_v:.5f}  test {np.mean(q):.5f}   "
        + " ".join(f"{s}:{v:.4f}" for s, v in rows))
    return ArmResult(best_s, best_v, q, rows)


def run_all(ws: WeightSpace, cfg: Config, sources: Dict[str, Source],
            names: Sequence[str], check: EvalSet, test: EvalSet,
            log=print) -> Dict[str, ArmResult]:
    ws.reset()
    base_check, base_test = sim(ws.model, check), sim(ws.model, test)
    log(f"[배율] 격자 {cfg.scales}")
    log(f"  {'base':15s} 배율 0       check {np.mean(base_check):.5f}  "
        f"test {np.mean(base_test):.5f}")
    out: Dict[str, ArmResult] = {}
    for n in names:
        out[n] = scale_curve(ws, cfg, sources[n], n, check, test,
                             base_check, base_test, log)
        torch.cuda.empty_cache()
    top = cfg.scales[-1]
    hit = [n for n, r in out.items() if r.scale == top]
    if hit:
        log(f"  *** 격자 상한 {top} 을 고른 arm: {hit}. 격자를 더 늘릴 것.")
    out["__base__"] = ArmResult(0.0, float(np.mean(base_check)), base_test, [])
    return out


def compare(res: Dict[str, ArmResult], pairs: Sequence[Tuple[str, str, str]],
            boot: int, log=print) -> Dict[str, Tuple[float, float, float]]:
    table: Dict[str, Tuple[float, float, float]] = {}
    for tag, a, b in pairs:
        d, lo, hi = paired_bootstrap(res[a].test, res[b].test, n=boot)
        table[tag] = (d, lo, hi)
        log(f"  {tag:34s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {verdict(lo, hi)}")
    return table


def mean_of(res: Dict[str, ArmResult], prefix: str, n: int) -> np.ndarray:
    return np.mean(np.stack([res[f"{prefix}{i}"].test for i in range(n)]), axis=0)


def pick_best_local(res: Dict[str, ArmResult], locals_: Sequence[str]) -> str:
    """check 에서 고른다. test 에서 고르면 그 값은 oracle 이다."""
    return max(locals_, key=lambda n: res[n].check)
