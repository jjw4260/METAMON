# -*- coding: utf-8 -*-
"""실행 3. 배율 결정(check)과 최종 비교(test).

    python run/03_evaluate.py --out runs/gpt35 --ensemble --text

실행 2 의 결과만 읽는다. 재측정 없음.

주장이 둘이므로 재는 것도 둘이다.
    충실도   병합 - 최고 단일 surrogate  가 양수인가
             확률(eq:sim) 과 생성 문장(BLEU/ROUGE-L) 양쪽에서 본다
    종속성   Dependency(merged) < Dependency(union) 인가
             둘은 본 데이터가 같다. 차이가 병합 자체의 효과다

--text 는 생성을 돌리므로 느리다. cfg.text_n 으로 질의 수를 줄인다.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.aggregate import build_sources, weight_spread, weighted
from Pipeline.config import assert_same_data, load as load_cfg
from Pipeline.contribution import Contribution
from Pipeline.data import Splits, load_dataset_file
from Pipeline.dependency import output_ensemble, recovery_ratio, surrogate_dependency
from Pipeline.evaluate import compare, mean_of, pick_best_local, run_all, scale_curve
from Pipeline.fleet import load_deltas
from Pipeline.metrics import EvalSet, paired_bootstrap, sim, verdict
from Pipeline.modeling import WeightSpace, load_tokenizer, setup_precision
from Pipeline.textgen import cost_table, generate, score_text
from Pipeline.weightspace import Candidates


def _keys(d):
    return {ast.literal_eval(k): v for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="*",
                    default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--text", action="store_true",
                    help="생성 문장을 Target 응답과 비교한다(BLEU/ROUGE-L)")
    ap.add_argument("--text-arms", nargs="*", default=None,
                    help="생성할 arm. 기본은 base/최고Local/soup/metamon_cell")
    a = ap.parse_args()

    cfg_path = os.path.join(a.out, "ckpt", "config.json")
    cfg = load_cfg(cfg_path)
    cfg.out_root = a.out
    device = setup_precision()
    tok = load_tokenizer(cfg)
    sp = Splits(cfg, load_dataset_file(cfg.dataset_path, cfg, tok))
    sp.assert_clean()
    assert_same_data(cfg_path, sp.query_hash(), sp.shard_hash())

    ws = WeightSpace(cfg, device)
    check = EvalSet(sp.check, tok, device, cfg.eval_bs)
    test = EvalSet(sp.test, tok, device, cfg.eval_bs)

    raw = json.load(open(os.path.join(cfg.log_dir, "02_contribution.json"),
                         encoding="utf-8"))
    con = Contribution(score=_keys(raw["partial_score"]),
                       alpha=_keys(raw["alpha"]),
                       base_loss=raw["base_loss"])
    win_cell = _keys(raw["winner_cell"])
    win_layer = {int(k): v for k, v in raw["winner_layer"].items()}
    keep = set(raw["kept_layers"])

    deltas = load_deltas(ws, cfg)
    cand = Candidates(deltas, cfg.local_names, ws.keys, device)
    torch.cuda.empty_cache()

    src = build_sources(ws, cfg, cand, con, deltas, win_layer, keep, win_cell)
    spread = weight_spread(con, ws.keys)
    for t in a.taus:
        src[f"weighted_t{t}"] = weighted(cand, con, ws.keys, t * spread)

    names = ([cfg.all_name] + cfg.local_names + cfg.union_names
             + ["metamon_layer", "metamon_cell", "soup", "soup_raw"]
             + [f"random_{i}" for i in range(cfg.n_random)]
             + [f"shuffle_{i}" for i in range(cfg.n_random)]
             + [f"loo_{i}" for i in range(cfg.k)]
             + [f"fleet_{g}" for g in range(cfg.n_fleet)]
             + [f"weighted_t{t}" for t in a.taus])
    res = run_all(ws, cfg, src, names, check, test)

    from Pipeline.evaluate import ArmResult
    rand = mean_of(res, "random_", cfg.n_random)
    shuf = mean_of(res, "shuffle_", cfg.n_random)
    res["random_mean"] = ArmResult(0.0, 0.0, rand, [])
    res["shuffle_mean"] = ArmResult(0.0, 0.0, shuf, [])
    best_local = pick_best_local(res, cfg.local_names)
    best_tau = max(a.taus, key=lambda t: res[f"weighted_t{t}"].check)
    res["weighted"] = res[f"weighted_t{best_tau}"]

    print(f"\n[결과] test {len(sp.test)} 질의, paired bootstrap 95%   "
          f"best_local={best_local} (check 기준),  weighted tau={best_tau}")
    pairs = []
    for nm in ("metamon_layer", "metamon_cell", "weighted"):
        pairs += [
            (f"{nm} - {cfg.all_name}", nm, cfg.all_name),
            (f"{nm} - best_local", nm, best_local),
            (f"{nm} - soup", nm, "soup"),
            (f"{nm} - random 평균", nm, "random_mean"),
        ]
    pairs += [
        ("metamon_layer - shuffle 평균 (위치 선택)", "metamon_layer", "shuffle_mean"),
        ("shuffle 평균 - random 평균 (집중 효과)", "shuffle_mean", "random_mean"),
        ("soup - best_local (병합 자체)", "soup", best_local),
    ]
    # 데이터량을 맞춘 짝. fleet_g 와 union_g 는 같은 조각을 봤다.
    for g in range(cfg.n_fleet):
        pairs.append((f"fleet_{g} - union_{g} (데이터 동일)",
                      f"fleet_{g}", f"union_{g}"))
    table = compare(res, pairs, cfg.boot)
    print(f"  random 편차 "
          f"{np.std([res[f'random_{i}'].test_mean for i in range(cfg.n_random)]):.5f}")

    dep = surrogate_dependency(res, cfg)

    ens_mean = None
    if a.ensemble:
        ens = output_ensemble(ws, cfg, src, res, test)
        d, lo, hi = paired_bootstrap(ens, res["soup"].test, n=cfg.boot)
        print(f"  {'ensemble - soup':34s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  "
              f"{verdict(lo, hi)}")
        recovery_ratio(ens, res["soup"].test, res[best_local].test)
        ens_mean = float(np.mean(ens))

    # ---- 텍스트 수준. 생성 문장을 Target 응답과 비교한다.
    text = {}
    if a.text:
        pool = sp.test[: cfg.text_n] if cfg.text_n else sp.test
        gold = [x["gold"].strip() for x in pool]
        arms = a.text_arms or ["__base__", best_local, "soup", "metamon_cell",
                               cfg.all_name]
        print(f"\n[텍스트] {len(pool)} 질의 생성, Target 응답과 비교 "
              f"({'greedy' if cfg.text_temp <= 0 else f'T={cfg.text_temp}'})")
        print(f"  *** 비교 대상은 데이터셋 정답이 아니라 Target 응답이다.")
        for nm in arms:
            if nm == "__base__":
                ws.reset()
            elif nm in src:
                ws.apply(src[nm], res[nm].scale if nm in res else 1.0)
            else:
                print(f"  {nm} 없음. 건너뜀")
                continue
            hyp = generate(ws.model, tok, pool, cfg, device)
            text[nm] = score_text(hyp, gold, cfg, tag=nm)
            text[nm]["sample"] = [{"prompt": pool[i]["prompt"],
                                   "target": gold[i], "surrogate": hyp[i]}
                                  for i in range(min(3, len(pool)))]
            ws.reset()
            torch.cuda.empty_cache()

    cost = cost_table(ws.model, cfg, len(sp.all))

    out = {
        "best_local": best_local, "best_tau": best_tau,
        "scale": {n: r.scale for n, r in res.items() if r.curve},
        "check": {n: r.check for n, r in res.items() if r.curve},
        "test_mean": {n: r.test_mean for n, r in res.items()},
        "test_per_query": {n: list(map(float, r.test)) for n, r in res.items()},
        "curves": {n: r.curve for n, r in res.items() if r.curve},
        "compare": {k: list(v) for k, v in table.items()},
        "dependency": dep, "text": text, "cost": cost,
        "ensemble_mean": ens_mean,
    }
    path = os.path.join(cfg.log_dir, "03_evaluate.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
