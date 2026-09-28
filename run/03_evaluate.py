# -*- coding: utf-8 -*-
"""실행 3. 배율 결정(check)과 최종 비교(test).

    python run/03_evaluate.py --out runs/gpt35 --ensemble --text

실행 2 의 결과만 읽는다. 재측정 없음.

주장이 둘이므로 재는 것도 둘이다.

  주장 1 (충실도)  병합 - 최고 단일 surrogate > 0
      확률(eq:sim) 과 생성(BLEU/ROUGE-L) 양쪽에서 본다. 그리고 **최고 단일을
      고르는 기준도 보고하는 지표와 같아야 한다.** avgBF 로 고르면 생성에서
      5 위인 Local 이 뽑히는 것을 확인했다. 그래서 세 가지로 다 고른다.
          best_sim   check 의 avgBF
          best_loss  check 의 L(theta;1)      <- 기여도와 같은 기준
          best_gen   check 의 생성 ROUGE-L    <- 생성으로 보고할 때의 기준
      soup 은 K 개 균등 평균이라 구조적으로 최고 단일을 못 이긴다.
      greedy 계열이 그 자리를 맡는다(최고 단일에서 출발한다).

  주장 2 (종속성)  병합 크기가 커지면 종속성이 준다
      cm{m}_{g} 는 Local 을 m 개씩 겹치지 않게 묶은 평균이다. 겹치지 않으므로
      leave-one-out 처럼 "공유해서 분산이 준" 것이 아니다.
      union_g 는 같은 조각을 합쳐 학습한 단일 모델이다(데이터량 대조).
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

from Pipeline.aggregate import (build_sources, curve_sources, greedy_soup,
                                metamon_greedy, weight_spread, weighted)
from Pipeline.config import assert_same_data, load as load_cfg
from Pipeline.contribution import Contribution
from Pipeline.data import Splits, load_dataset_file
from Pipeline.dependency import (dependency_curve, output_ensemble,
                                 recovery_ratio, surrogate_dependency)
from Pipeline.evaluate import ArmResult, compare, mean_of, run_all
from Pipeline.fleet import load_deltas
from Pipeline.metrics import EvalSet, loss, paired_bootstrap, sim, verdict
from Pipeline.modeling import WeightSpace, load_tokenizer, setup_precision
from Pipeline.oracle import complementarity, report as oracle_report
from Pipeline.textgen import (cost_table, generate, score_text,
                              text_compare, victim_scores)
from Pipeline.weightspace import Candidates


def _keys(d):
    return {ast.literal_eval(k): v for k, v in d.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--taus", type=float, nargs="*",
                    default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--scales", type=float, nargs="*", default=None)
    ap.add_argument("--arm", default="metamon_greedy",
                    help="04 가 'Ours' 로 볼 arm")
    a = ap.parse_args()

    cfg_path = os.path.join(a.ckpt or a.out, "ckpt", "config.json")
    cfg = load_cfg(cfg_path)
    cfg.out_root = a.out
    cfg.ckpt_root = a.ckpt or ""
    if a.scales:
        cfg.scales = sorted(set([0.0] + list(a.scales)))
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
    src.update(curve_sources(cfg, deltas, device))
    spread = weight_spread(con, ws.keys)
    for t in a.taus:
        src[f"weighted_t{t}"] = weighted(cand, con, ws.keys, t * spread)

    # ---------------------------------------------------------------- 1 단계
    names = ([cfg.all_name] + cfg.local_names + cfg.union_names
             + ["metamon_layer", "metamon_cell", "soup", "soup_raw"]
             + [f"random_{i}" for i in range(cfg.n_random)]
             + [f"shuffle_{i}" for i in range(cfg.n_random)]
             + [f"cm{m}_{g}" for m in cfg.curve for g in range(cfg.k // m)
                if m > 1]
             + [f"weighted_t{t}" for t in a.taus])
    res = run_all(ws, cfg, src, names, check, test)
    for i, n in enumerate(cfg.local_names):          # m=1 은 개별 Local 이다
        res[f"cm1_{i}"] = res[n]
        src[f"cm1_{i}"] = src[n]

    # ---- 최고 단일을 세 기준으로 고른다. 기준이 다르면 다른 Local 이 뽑힌다.
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

    # ---- 상보성. 뽑을 것이 있는지부터 본다. headroom 이 0 이면 그 뒤가 무의미하다.
    oc_sim = complementarity({n: res[n].test for n in cfg.local_names},
                             cfg.local_names, best["loss"])
    oracle_report(oc_sim, "확률 (eq:sim)", cfg.k)

    # ---------------------------------------------------------------- 2 단계
    # greedy 계열은 1 단계 결과(순서, 시작점)가 있어야 만들 수 있다.
    greedy_info = {}
    if not a.no_greedy:
        print()
        order = sorted(cfg.local_names, key=lambda n: -res[n].check)
        gs, kept_names = greedy_soup(
            ws, cfg, deltas, order, check, [0.5, 0.75, 1.0, 1.5])
        src["greedy_soup"] = gs
        start = best["loss"]
        mg, n_ok, n_try = metamon_greedy(
            ws, cfg, cand, con, deltas, start, win_cell,
            res[start].scale, check)
        src["metamon_greedy"] = mg
        greedy_info = {"soup_members": kept_names, "start": start,
                       "cells_taken": n_ok, "cells_tried": n_try}
        res.update(run_all(ws, cfg, src, ["greedy_soup", "metamon_greedy"],
                           check, test))

    rand = mean_of(res, "random_", cfg.n_random)
    shuf = mean_of(res, "shuffle_", cfg.n_random)
    res["random_mean"] = ArmResult(0.0, 0.0, rand, [])
    res["shuffle_mean"] = ArmResult(0.0, 0.0, shuf, [])
    res["local_mean"] = ArmResult(0.0, 0.0, mean_of(res, "local_", cfg.k), [])
    best_tau = max(a.taus, key=lambda t: res[f"weighted_t{t}"].check)
    res["weighted"] = res[f"weighted_t{best_tau}"]

    arm = a.arm if a.arm in res else "soup"
    ours = [n for n in ("greedy_soup", "metamon_greedy", "soup",
                        "metamon_layer", "weighted") if n in res]

    # 회수율까지 포함해 다시 낸다. headroom 의 몇 %를 기전이 가져왔는가.
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
        ("metamon_layer - shuffle 평균 (위치 선택)", "metamon_layer", "shuffle_mean"),
        ("shuffle 평균 - random 평균 (집중 효과)", "shuffle_mean", "random_mean"),
    ]
    for g in range(cfg.n_fleet):                    # 데이터량을 맞춘 짝
        nm = f"cm{cfg.fleet_size}_{g}"
        if nm in res and f"union_{g}" in res:
            pairs.append((f"{nm} - union_{g} (데이터 동일)", nm, f"union_{g}"))
    table = compare(res, pairs, cfg.boot)

    dep = surrogate_dependency(res, cfg)
    curve = dependency_curve(res, cfg)

    ens_mean = None
    if a.ensemble:
        ens = output_ensemble(ws, cfg, src, res, test)
        d, lo, hi = paired_bootstrap(ens, res[arm].test, n=cfg.boot)
        print(f"  {'ensemble - ' + arm:34s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  "
              f"{verdict(lo, hi)}")
        recovery_ratio(ens, res[arm].test, res[best["loss"]].test)
        ens_mean = float(np.mean(ens))

    # ---------------------------------------------------------------- 텍스트
    text, victim, tcmp, oc_gen = {}, None, {}, None
    text_state = {"text": {}, "victim": None, "tcmp": {}, "oc_gen": None,
                  "best": best, "hyps_path":
                  os.path.join(cfg.log_dir, "03_hyps.json"),
                  "query_hash": sp.query_hash()}

    cost = cost_table(ws.model, cfg, len(sp.all))
    skew = sp.skew(cfg)

    def save() -> str:
        """지금까지 나온 것을 전부 저장한다. 텍스트 단계에서 죽어도 확률
        결과는 남는다. 생성은 비싸므로 hyps 를 따로 캐시한다."""
        out = {
            "arm": arm, "best_local": best, "best_tau": best_tau,
            "check_loss": ck_loss, "greedy": greedy_info,
            "scale": {n: r.scale for n, r in res.items() if r.curve},
            "check": {n: r.check for n, r in res.items() if r.curve},
            "test_mean": {n: r.test_mean for n, r in res.items()},
            "test_per_query": {n: list(map(float, r.test)) for n, r in res.items()},
            "curves": {n: r.curve for n, r in res.items() if r.curve},
            "compare": {k: list(v) for k, v in table.items()},
            "text_compare": {k: {m: list(v) for m, v in d.items()}
                             for k, d in tcmp.items()},
            "dependency": dep, "dependency_curve": curve,
            "oracle_sim": oc_sim, "oracle_gen": oc_gen,
            "shard_skew": skew,
            "text": text, "victim_text": victim, "cost": cost,
            "ensemble_mean": ens_mean,
        }
        p = os.path.join(cfg.log_dir, "03_evaluate.json")
        json.dump(out, open(p, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        return p

    # 텍스트 단계 전에 한 번 저장한다. 생성에서 무슨 일이 나도 확률 결과와
    # 종속성 곡선은 이미 디스크에 있다.
    print(f"\n중간 저장: {save()}")

    # ---------------------------------------------------------------- 텍스트
    if a.text:
        try:
            _text_stage(ws, tok, cfg, sp, src, res, best, ours, arm, device,
                        best_tau, text_state)
        except Exception:
            import traceback
            print("\n*** 텍스트 단계가 실패했다. 확률 결과는 이미 저장돼 있다.")
            traceback.print_exc()
        text = text_state["text"]
        victim = text_state["victim"]
        tcmp = text_state["tcmp"]
        oc_gen = text_state["oc_gen"]
        best = text_state["best"]

    path = save()
    print(f"\n저장: {path}")


def _text_stage(ws, tok, cfg, sp, src, res, best, ours, arm, device,
                best_tau, st) -> None:
    """생성과 텍스트 지표. 생성 결과는 디스크에 캐시한다.

    생성이 이 단계 비용의 대부분이다. 뒤에서 무엇이 터지든 다시 생성하지
    않도록 hyps 를 먼저 저장하고, 다시 돌릴 때 같은 질의면 그대로 읽는다.
    """
    from Pipeline.textgen import _rouge_each
    pool = sp.test[: cfg.text_n] if cfg.text_n else sp.test
    cpool = sp.check[: cfg.text_n] if cfg.text_n else sp.check
    gold = [x["gold"].strip() for x in pool]
    ref = [str(x["ref"]).strip() for x in pool]
    cgold = [x["gold"].strip() for x in cpool]

    def gen(nm, items):
        if nm == "__base__":
            ws.reset()
        else:
            ws.apply(src[nm], res[nm].scale if nm in res else 1.0)
        h = generate(ws.model, tok, items, cfg, device)
        ws.reset()
        torch.cuda.empty_cache()
        return h

    arms = (["__base__"] + cfg.local_names + ours
            + [cfg.all_name, "metamon_cell", "soup_raw",
               f"weighted_t{best_tau}"])
    arms = [n for i, n in enumerate(arms) if n not in arms[:i]]

    # ---- 캐시. 같은 질의에 대해 이미 생성해 둔 것이 있으면 다시 안 만든다.
    cache = {}
    if os.path.exists(st["hyps_path"]):
        try:
            c = json.load(open(st["hyps_path"], encoding="utf-8"))
            if c.get("query_hash") == st["query_hash"]:
                cache = c
                print(f"\n[생성 캐시] {st['hyps_path']} 사용. "
                      f"arm {len(c.get('hyps', {}))}개, check {len(c.get('ck_gen', {}))}개")
        except Exception:
            cache = {}

    ck_gen = cache.get("ck_gen") or {}
    if len(ck_gen) < cfg.k:
        print(f"\n[생성] check {len(cpool)} 질의로 최고 단일을 고른다")
        for n in cfg.local_names:
            if n not in ck_gen:
                ck_gen[n] = float(np.mean(_rouge_each(gen(n, cpool), cgold)))
    best["gen"] = max(cfg.local_names, key=lambda n: ck_gen[n])
    for n in cfg.local_names:
        print(f"    {n:10s} check ROUGE-L {ck_gen[n]:.4f}"
              + ("   <- 최고" if n == best["gen"] else ""))

    print(f"\n[텍스트] test {len(pool)} 질의 생성 "
          f"({'greedy' if cfg.text_temp <= 0 else f'T={cfg.text_temp}'})   "
          f"BERTScore {cfg.bert_score_model or '끔'}")
    hyps = dict(cache.get("hyps") or {})
    todo = [n for n in arms if n not in hyps and (n == "__base__" or n in src)]
    for nm in todo:
        hyps[nm] = gen(nm, pool)
        # 한 arm 끝날 때마다 저장한다. 중간에 끊겨도 거기까지는 산다.
        json.dump({"query_hash": st["query_hash"], "hyps": hyps,
                   "ck_gen": ck_gen},
                  open(st["hyps_path"], "w", encoding="utf-8"),
                  ensure_ascii=False)
    print(f"  생성 완료 {len(hyps)}개 arm   캐시 {st['hyps_path']}")

    victim = victim_scores(gold, ref, cfg)
    st["victim"] = victim
    text = {}
    for nm in arms:
        if nm not in hyps:
            continue
        text[nm] = score_text(hyps[nm], gold, ref, cfg, victim=victim, tag=nm)
        text[nm]["sample"] = [
            {"prompt": pool[i]["prompt"], "target": gold[i],
             "ref": ref[i], "surrogate": hyps[nm][i]}
            for i in range(min(3, len(pool)))]
    st["text"] = text

    # ---- 생성에서의 상보성. 이 수가 본 진단이므로 비교보다 먼저 낸다.
    rq = {n: _rouge_each(hyps[n], gold) for n in cfg.local_names if n in hyps}
    if len(rq) == cfg.k:
        st["oc_gen"] = complementarity(
            rq, cfg.local_names, best["gen"],
            arms={n: _rouge_each(hyps[n], gold) for n in ours if n in hyps})
        oracle_report(st["oc_gen"], "생성 (ROUGE-L, Target 응답 대비)", cfg.k)

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
