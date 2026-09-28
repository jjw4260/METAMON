# -*- coding: utf-8 -*-
"""실행 3. 배율 결정(check)과 최종 비교(test).

    python run/03_evaluate.py --out runs/gpt35 --ensemble --text

실행 2 의 결과만 읽는다. 재측정 없음.

논문의 축이 base 로 바뀌었다. base 가 주축이고 병합은 **base 가 덮지 못한
영역을 덮는** 보정항이다. 그래서 재는 것이 셋이다.

  E1  추출 성능      병합이 최고 단일 surrogate 를, 그리고 전체 질의 단일
                     모델(lord_all)을 이기는가. **생성**에서 본다
  E2  종속성         fleet_g 와 union_g 는 본 데이터가 같다. 그 짝의 분산 차이
  버킷 분해          질의를 BASE 점수로 나눠 이득이 **낮은 버킷에 몰리는가**.
                     이것이 "병합이 base 의 구멍을 메운다" 의 본 증거다

**greedy 의 채택 기준은 보고할 지표와 같아야 한다.** 지난 실행은 check 의
avgBF 로 채택했는데, 병합 arm 에서 avgBF 와 생성 ROUGE-L 의 Spearman 이
-0.821 이었다. 확률을 올리는 방향이 생성을 내리는 방향이어서 최고 단일에서
출발하고도 생성에서 졌다. `--greedy-on gen` 이 기본이다.
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

from Pipeline.aggregate import (build_sources, greedy_soup, metamon_greedy,
                                weight_spread, weighted)
from Pipeline.buckets import by_base_weakness, report as bucket_report
from Pipeline.config import assert_same_data, load as load_cfg
from Pipeline.contribution import Contribution
from Pipeline.data import Splits, load_dataset_file
from Pipeline.dependency import (output_ensemble, recovery_ratio,
                                 surrogate_dependency)
from Pipeline.evaluate import ArmResult, compare, mean_of, run_all
from Pipeline.fleet import open_store
from Pipeline.metrics import EvalSet, loss, paired_bootstrap, sim, verdict
from Pipeline.modeling import WeightSpace, load_tokenizer, setup_precision
from Pipeline.oracle import complementarity, report as oracle_report
from Pipeline.textgen import (cost_table, generate, score_text, text_compare,
                              victim_scores, _rouge_each)
from Pipeline.weightspace import Candidates


def _keys(d):
    return {ast.literal_eval(k): v for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.3, 1.0, 3.0])
    ap.add_argument("--arm", default="greedy",
                    help="주 결과로 보고할 arm")
    ap.add_argument("--greedy-on", default="gen", choices=["gen", "sim"],
                    help="greedy 채택 기준. gen=check 생성 ROUGE-L (기본)")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--text", action="store_true")
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

    store = open_store(ws, cfg)
    cand = Candidates(store, cfg.local_names, ws.keys, device)

    src = build_sources(ws, cfg, cand, con, store, win_layer, keep, win_cell)
    spread = weight_spread(con, ws.keys)
    for t in a.taus:
        src[f"weighted_t{t}"] = weighted(cand, con, ws.keys, t * spread)

    # ---------------------------------------------------------------- 1 단계
    names = ([cfg.all_name] + cfg.local_names + cfg.union_names
             + [f"fleet_{g}" for g in range(cfg.n_fleet)]
             + ["metamon_layer", "metamon_cell", "soup", "soup_raw"]
             + [f"random_{i}" for i in range(cfg.n_random)]
             + [f"shuffle_{i}" for i in range(cfg.n_shuffle)]
             + [f"loo_{i}" for i in range(cfg.k)]
             + [f"weighted_t{t}" for t in a.taus])
    print(f"[arm] {len(names)}개 평가한다")
    res = run_all(ws, cfg, src, names, check, test)

    # ---- 최고 단일. 기준마다 다른 Local 이 뽑히므로 셋 다 고른다.
    print("\n[최고 단일 선택]  check 에서 고른다. test 를 보고 고르면 oracle 이다.")
    ck_loss = {}
    for n in cfg.local_names:
        ws.apply(src[n], res[n].scale)
        ck_loss[n] = loss(ws.model, check)
    ws.reset()
    best = {"sim": max(cfg.local_names, key=lambda n: res[n].check),
            "loss": min(cfg.local_names, key=lambda n: ck_loss[n])}
    print(f"  avgBF 기준   {best['sim']}   check {res[best['sim']].check:.5f}")
    print(f"  손실 기준    {best['loss']}   check L {ck_loss[best['loss']]:.5f}")

    oc_sim = complementarity({n: res[n].test for n in cfg.local_names},
                             cfg.local_names, best["loss"])
    oracle_report(oc_sim, "확률 (eq:sim)", cfg.k)

    # ---------------------------------------------------------------- 생성 준비
    # greedy 를 생성으로 채점하려면 check 생성이 먼저 있어야 한다.
    cpool = sp.check[: cfg.greedy_n] if cfg.greedy_n else sp.check
    cgold = [x["gold"].strip() for x in cpool]

    def gen(nm_or_src, items, scale=1.0):
        s = src[nm_or_src] if isinstance(nm_or_src, str) else nm_or_src
        if nm_or_src == "__base__":
            ws.reset()
        else:
            ws.apply(s, scale)
        h = generate(ws.model, tok, items, cfg, device)
        ws.reset()
        torch.cuda.empty_cache()
        return h

    ck_gen = {}
    if a.text or a.greedy_on == "gen":
        print(f"\n[생성] check {len(cpool)} 질의로 최고 단일을 고른다")
        for n in cfg.local_names:
            ck_gen[n] = float(np.mean(_rouge_each(
                gen(n, cpool, res[n].scale), cgold)))
            print(f"    {n:10s} check ROUGE-L {ck_gen[n]:.4f}")
        best["gen"] = max(cfg.local_names, key=lambda n: ck_gen[n])
        print(f"  생성 기준    {best['gen']}   check ROUGE-L "
              f"{ck_gen[best['gen']]:.4f}")

    # ---------------------------------------------------------------- 2 단계
    greedy_info = {}
    if not a.no_greedy:
        if a.greedy_on == "gen":
            start = best["gen"]
            order = sorted(cfg.local_names, key=lambda n: -ck_gen[n])
            def score():
                h = generate(ws.model, tok, cpool, cfg, device)
                return float(np.mean(_rouge_each(h, cgold)))
        else:
            start = best["loss"]
            order = sorted(cfg.local_names, key=lambda n: -res[n].check)
            def score():
                return float(np.mean(sim(ws.model, check)))
        print(f"\n[greedy]  채택 기준 = {a.greedy_on}   시작 {start}")
        gs, kept, v_gs = greedy_soup(ws, store, order, score, cfg.greedy_scales)
        mg, n_ok, n_try, v_mg = metamon_greedy(
            ws, cand, con, store, start, win_cell, res[start].scale, score,
            max_cells=cfg.greedy_cells)
        # 둘 중 check 에서 좋은 쪽을 본 결과 arm 으로 삼는다. 둘 다 최고 단일
        # 에서 출발했으므로 어느 쪽이든 그 기준에서 최고 단일 이상이다.
        src["greedy"] = gs if v_gs >= v_mg else mg
        src["greedy_soup"], src["metamon_greedy"] = gs, mg
        greedy_info = {"on": a.greedy_on, "start": start, "soup_members": kept,
                       "soup_check": v_gs, "metamon_check": v_mg,
                       "cells_taken": n_ok, "cells_tried": n_try,
                       "picked": "greedy_soup" if v_gs >= v_mg else "metamon_greedy"}
        print(f"  -> {greedy_info['picked']} 채택 "
              f"(soup {v_gs:.5f} vs metamon {v_mg:.5f})")
        res.update(run_all(ws, cfg, src,
                           ["greedy", "greedy_soup", "metamon_greedy"],
                           check, test))

    res["random_mean"] = ArmResult(0.0, 0.0, mean_of(res, "random_", cfg.n_random), [])
    res["local_mean"] = ArmResult(0.0, 0.0, mean_of(res, "local_", cfg.k), [])
    best_tau = max(a.taus, key=lambda t: res[f"weighted_t{t}"].check)
    res["weighted"] = res[f"weighted_t{best_tau}"]

    arm = a.arm if a.arm in res else "soup"
    ours = [n for n in ("greedy", "soup", "metamon_layer", "weighted")
            if n in res]

    oc_sim = complementarity({n: res[n].test for n in cfg.local_names},
                             cfg.local_names, best["loss"],
                             arms={n: res[n].test for n in ours})
    oracle_report(oc_sim, "확률 (eq:sim) — 회수율 포함", cfg.k)

    # ---------------------------------------------------------------- 비교
    print(f"\n[결과] test {len(sp.test)} 질의, paired bootstrap 95%   "
          f"weighted tau={best_tau}")
    pairs = []
    for nm in ours:
        for tag, other in (("best_local(sim)", best["sim"]),
                           ("best_local(loss)", best["loss"]),
                           (cfg.all_name, cfg.all_name),
                           ("local 평균", "local_mean"),
                           ("soup", "soup")):
            if nm != other:
                pairs.append((f"{nm} - {tag}", nm, other))
    pairs += [
        ("metamon_layer - shuffle_0 (위치 선택)", "metamon_layer", "shuffle_0"),
        ("shuffle_0 - random 평균 (집중 효과)", "shuffle_0", "random_mean"),
    ]
    for g in range(cfg.n_fleet):                    # 데이터량을 맞춘 짝
        if f"fleet_{g}" in res and f"union_{g}" in res:
            pairs.append((f"fleet_{g} - union_{g} (데이터 동일)",
                          f"fleet_{g}", f"union_{g}"))
    table = compare(res, pairs, cfg.boot)

    dep = surrogate_dependency(res, cfg)

    ens_mean = None
    if a.ensemble:
        ens = output_ensemble(ws, cfg, src, res, test)
        d, lo, hi = paired_bootstrap(ens, res[arm].test, n=cfg.boot)
        print(f"  {'ensemble - ' + arm:34s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  "
              f"{verdict(lo, hi)}")
        recovery_ratio(ens, res[arm].test, res[best["loss"]].test)
        ens_mean = float(np.mean(ens))

    cost = cost_table(ws.model, cfg, len(sp.all))
    skew = sp.skew(cfg)
    text, victim, tcmp, oc_gen, bk_sim, bk_gen = {}, None, {}, None, None, None

    # ---- BASE 의 질의별 확률. 버킷을 가르는 기준이다.
    ws.reset()
    base_q = list(map(float, sim(ws.model, test)))
    bk_sim = by_base_weakness(base_q, {n: res[n].test for n in ours + [best["loss"]]},
                              best["loss"])
    bucket_report(bk_sim, "확률 (eq:sim)")

    def save() -> str:
        out = {
            "arm": arm, "best_local": best, "best_tau": best_tau,
            "check_loss": ck_loss, "check_gen": ck_gen, "greedy": greedy_info,
            "scale": {n: r.scale for n, r in res.items() if r.curve},
            "check": {n: r.check for n, r in res.items() if r.curve},
            "test_mean": {n: r.test_mean for n, r in res.items()},
            "test_per_query": {n: list(map(float, r.test)) for n, r in res.items()},
            "base_per_query": base_q,
            "curves": {n: r.curve for n, r in res.items() if r.curve},
            "compare": {k: list(v) for k, v in table.items()},
            "text_compare": {k: {m: list(v) for m, v in d.items()}
                             for k, d in tcmp.items()},
            "dependency": dep, "oracle_sim": oc_sim, "oracle_gen": oc_gen,
            "bucket_sim": bk_sim, "bucket_gen": bk_gen, "shard_skew": skew,
            "text": text, "victim_text": victim, "cost": cost,
            "ensemble_mean": ens_mean,
        }
        p = os.path.join(cfg.log_dir, "03_evaluate.json")
        json.dump(out, open(p, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        return p

    print(f"\n중간 저장: {save()}")

    # ---------------------------------------------------------------- 텍스트
    if a.text:
        st = {"text": {}, "victim": None, "tcmp": {}, "oc_gen": None,
              "bk_gen": None, "best": best,
              "hyps_path": os.path.join(cfg.log_dir, "03_hyps.json"),
              "query_hash": sp.query_hash()}
        try:
            _text_stage(ws, tok, cfg, sp, src, res, best, ours, device, base_q,
                        best_tau, gen, st)
        except Exception:
            import traceback
            print("\n*** 텍스트 단계가 실패했다. 확률 결과는 이미 저장돼 있다.")
            traceback.print_exc()
        text, victim, tcmp = st["text"], st["victim"], st["tcmp"]
        oc_gen, bk_gen, best = st["oc_gen"], st["bk_gen"], st["best"]

    print(f"\n저장: {save()}")


def _text_stage(ws, tok, cfg, sp, src, res, best, ours, device, base_q,
                best_tau, gen, st) -> None:
    """생성과 텍스트 지표. 생성 결과는 arm 마다 디스크에 캐시한다."""
    pool = sp.test[: cfg.text_n] if cfg.text_n else sp.test
    gold = [x["gold"].strip() for x in pool]
    ref = [str(x["ref"]).strip() for x in pool]

    arms = (["__base__"] + cfg.local_names + ours
            + [cfg.all_name, "metamon_cell", "soup_raw",
               f"weighted_t{best_tau}"])
    arms = [n for i, n in enumerate(arms) if n not in arms[:i]]

    cache = {}
    if os.path.exists(st["hyps_path"]):
        try:
            c = json.load(open(st["hyps_path"], encoding="utf-8"))
            if c.get("query_hash") == st["query_hash"]:
                cache = c
                print(f"\n[생성 캐시] arm {len(c.get('hyps', {}))}개 재사용")
        except Exception:
            cache = {}

    print(f"\n[텍스트] test {len(pool)} 질의 생성 "
          f"({'greedy' if cfg.text_temp <= 0 else f'T={cfg.text_temp}'})   "
          f"BERTScore {cfg.bert_score_model or '끔'}")
    hyps = dict(cache.get("hyps") or {})
    for nm in arms:
        if nm in hyps or (nm != "__base__" and nm not in src):
            continue
        hyps[nm] = gen(nm, pool, res[nm].scale if nm in res else 1.0)
        json.dump({"query_hash": st["query_hash"], "hyps": hyps},
                  open(st["hyps_path"], "w", encoding="utf-8"),
                  ensure_ascii=False)
    print(f"  생성 완료 {len(hyps)}개 arm")

    victim = victim_scores(gold, ref, cfg)
    st["victim"] = victim
    text = {}
    for nm in arms:
        if nm not in hyps:
            continue
        text[nm] = score_text(hyps[nm], gold, ref, cfg, victim=victim, tag=nm)
        text[nm]["sample"] = [
            {"prompt": pool[i]["prompt"], "target": gold[i], "ref": ref[i],
             "surrogate": hyps[nm][i]} for i in range(min(3, len(pool)))]
    st["text"] = text

    # ---- 생성에서의 상보성과 버킷. 이것이 본 진단이므로 비교보다 먼저 낸다.
    rq = {n: _rouge_each(hyps[n], gold) for n in cfg.local_names if n in hyps}
    arm_q = {n: _rouge_each(hyps[n], gold) for n in ours if n in hyps}
    if len(rq) == cfg.k:
        st["oc_gen"] = complementarity(rq, cfg.local_names, best["gen"],
                                       arms=arm_q)
        oracle_report(st["oc_gen"], "생성 (ROUGE-L, Target 응답 대비)", cfg.k)
    if "__base__" in hyps and best["gen"] in hyps:
        b = _rouge_each(hyps["__base__"], gold)
        d = dict(arm_q)
        d[best["gen"]] = _rouge_each(hyps[best["gen"]], gold)
        st["bk_gen"] = by_base_weakness(b, d, best["gen"])
        bucket_report(st["bk_gen"], "생성 (ROUGE-L)")

    print(f"\n[생성 비교] paired bootstrap 95%  (Target 응답 대비)")
    tcmp = {}
    for nm in ours:
        for tag, other in (("best_local(gen)", best["gen"]),
                           ("best_local(loss)", best["loss"]),
                           (cfg.all_name, cfg.all_name)):
            if nm in hyps and other in hyps and nm != other:
                tcmp[f"{nm} - {tag}"] = text_compare(
                    hyps[nm], hyps[other], gold, tag=f"{nm} - {tag}")
    st["tcmp"] = tcmp
    st["best"] = best


if __name__ == "__main__":
    main()
