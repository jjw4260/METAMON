# -*- coding: utf-8 -*-
"""Surrogate Dependency 와 앙상블 baseline.

  eq:single_dependency       Dependency({theta_Single,k}) = Var_k(avgBF_k)
  eq:dependency_mitigation   Dependency(병합) < Dependency(단일)

종속성 주장은 대조 설계가 전부다. 세 집합을 비교한다.

    single   개별 Local K 개.            조각 1 개 분량을 봤다
    union    fleet 마다 조각 fleet_size 개를 합쳐 학습한 단일 모델. 병합 없음
    merged   같은 조각들을 따로 학습한 뒤 병합.                  병합 있음

union 과 merged 는 **본 데이터가 같다.** 둘의 분산 차이가 병합 자체의 효과다.
이 대조가 없으면 "데이터를 더 봐서 분산이 줄었다" 로 반박당한다.

leave-one-out 분산도 함께 내지만 판정에는 쓰지 않는다. LOO 병합들은 K 개 중
K-1 개를 공유하므로 분산이 줄어드는 것이 구성상 거의 자동이다.

앙상블은 K 개 모델을 모두 유지하고 출력 확률을 평균한다. 추론 비용이 K 배라
METAMON 의 설계 목표(단일 모델)와 다르지만, "비용 1 배로 앙상블 이득의 몇 %
를 회수하는가" 를 말하려면 반드시 필요한 기준선이다.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config
from .evaluate import ArmResult
from .metrics import EvalSet, dependency, ensemble_sim, paired_bootstrap, token_probs
from .modeling import WeightSpace


def _row(tag: str, vals: Sequence[float], log) -> float:
    d = dependency(vals)
    log(f"  {tag:34s} {d:.3e}   n={len(vals)}  "
        f"평균 {np.mean(vals):.5f}  avgBF {[round(v, 5) for v in vals]}")
    return d


def surrogate_dependency(res: Dict[str, ArmResult], cfg: Config, log=print
                         ) -> Dict[str, object]:
    """세 집합의 종속성. 판정은 union 대 merged 로 한다."""
    single = [res[n].test_mean for n in cfg.local_names]
    union = [res[n].test_mean for n in cfg.union_names if n in res]
    merged = [res[f"cm{cfg.fleet_size}_{g}"].test_mean
              for g in range(cfg.n_fleet)
              if f"cm{cfg.fleet_size}_{g}" in res]
    loo = [res[f"loo_{i}"].test_mean for i in range(cfg.k) if f"loo_{i}" in res]

    log("[Surrogate Dependency]  분산이 작을수록 surrogate 선택에 덜 휘둘린다")
    d_single = _row("single  (조각 1개, 병합 없음)", single, log)
    d_union = _row(f"union   (조각 {cfg.fleet_size}개, 병합 없음)", union, log) \
        if union else float("nan")
    d_merged = _row(f"merged  (조각 {cfg.fleet_size}개, 병합 있음)", merged, log) \
        if merged else float("nan")
    d_loo = _row("loo     (겹침. 참고용, 판정 제외)", loo, log) if loo else float("nan")

    ok = bool(merged and union and d_merged < d_union)
    if union and merged:
        log(f"  [판정] merged < union  {'성립' if ok else '불성립'}  "
            f"({d_union / max(d_merged, 1e-30):.1f}배).  "
            f"데이터량이 같으므로 이 차이는 병합 자체의 효과다.")
        log(f"         참고: single 대비 {d_single / max(d_merged, 1e-30):.1f}배")
    else:
        log("  [판정] union 또는 merged arm 이 없다. fleet_size 설정을 확인할 것.")
    if len(merged) < 3:
        log(f"  *** merged 가 {len(merged)} 개뿐이다. 분산 추정이 얇다. "
            f"k 를 늘려 n_fleet 을 3 이상으로 할 것.")

    return {"single": d_single, "union": d_union, "merged": d_merged,
            "loo": d_loo, "ok": ok,
            "values": {"single": single, "union": union,
                       "merged": merged, "loo": loo}}


def dependency_curve(res: Dict[str, ArmResult], cfg: Config, log=print) -> dict:
    """병합 크기 m 에 따른 종속성.

    m = 1 은 개별 Local 이고, m 이 커질수록 묶음 수가 줄어든다. 묶음끼리는
    구성원이 겹치지 않으므로 leave-one-out 처럼 "겹쳐서 분산이 준" 것이
    아니다. 크기가 커지며 분산이 단조로 줄면 그것이 종속성 완화의 증거다.

    fleet_size 를 2 로 두고 한 점만 보면 뒤집힌다(좋은 것과 나쁜 것을 1:1 로
    섞으면 희석이 안 된다). 곡선으로 봐야 한다.
    """
    rows = []
    for m in cfg.curve:
        vals = [res[f"cm{m}_{g}"].test_mean for g in range(cfg.k // m)
                if f"cm{m}_{g}" in res]
        if len(vals) < 2:
            continue
        rows.append({"m": m, "n": len(vals), "var": dependency(vals),
                     "mean": float(np.mean(vals)), "values": vals})
    if not rows:
        log("[종속성 곡선] arm 이 없다. cfg.curve 를 확인할 것.")
        return {"rows": [], "monotone": False}

    log("[종속성 곡선]  병합 크기 m 별 분산. 묶음끼리 구성원이 겹치지 않는다.")
    log(f"  {'m':>3s} {'묶음':>4s} {'분산':>11s} {'평균':>9s}   값")
    for r in rows:
        log(f"  {r['m']:3d} {r['n']:4d} {r['var']:11.3e} {r['mean']:9.5f}   "
            f"{[round(v, 4) for v in r['values']]}")
    mono = all(rows[i]["var"] >= rows[i + 1]["var"] for i in range(len(rows) - 1))
    first, last = rows[0], rows[-1]
    log(f"  m={first['m']} -> m={last['m']}  분산 "
        f"{first['var'] / max(last['var'], 1e-30):.1f}배 감소   "
        f"{'단조 감소' if mono else '단조가 아니다'}")
    return {"rows": rows, "monotone": mono}


def output_ensemble(ws: WeightSpace, cfg: Config, sources, res: Dict[str, ArmResult],
                    test: EvalSet, log=print) -> np.ndarray:
    """각 Local 을 check 에서 고른 배율로 켠 뒤 토큰 확률을 평균한다."""
    sets: List[List[np.ndarray]] = []
    for n in cfg.local_names:
        ws.apply(sources[n], res[n].scale)
        sets.append(token_probs(ws.model, test))
        ws.reset()
    ens = ensemble_sim(sets)
    log(f"[앙상블] K={cfg.k}, 추론 비용 {cfg.k}배   test {np.mean(ens):.5f}")
    return ens


def recovery_ratio(ens: np.ndarray, merge: np.ndarray, best_local: np.ndarray,
                   log=print) -> float:
    """비용 1 배의 병합이 앙상블 이득의 몇 % 를 회수하는가."""
    gain = float(np.mean(ens) - np.mean(best_local))
    got = float(np.mean(merge) - np.mean(best_local))
    r = 100.0 * got / gain if abs(gain) > 1e-12 else float("nan")
    log(f"  앙상블 이득 {gain:+.5f}, 병합 이득 {got:+.5f}  ->  회수율 {r:.1f}%")
    return r
