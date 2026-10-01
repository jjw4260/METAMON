# -*- coding: utf-8 -*-
"""실행 4. 판정.

    python run/04_report.py --out runs/gpt35

축이 base 다. base 가 주축이고 병합은 **base 가 덮지 못한 영역을 덮는** 보정항
이다. 그래서 판정도 그 순서로 한다.

  전제   조각이 갈렸는가(shard_skew), 뽑을 것이 있는가(headroom)
         headroom 이 0 이면 주장 1 은 불성립이 아니라 **미시험**이다
  기여 1 병합이 최고 단일과 lord_all 을 **생성에서** 이기는가
  기전   이득이 BASE 가 약한 질의에 몰리는가 (버킷)
         이게 "병합이 base 의 구멍을 메운다" 의 본 증거다. 고르게 퍼져 있으면
         보정이 아니라 그냥 평균 효과다
  기여 2 fleet_g 와 union_g 는 본 데이터가 같다. 그 짝의 분산 차이
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.config import METRICS, load as load_cfg


def _mark(lo, hi):
    return "양수" if lo > 0 else ("음수" if hi < 0 else "불확실")


def _bucket(bk, label):
    if not bk or not bk.get("rows"):
        return None
    rows, ref = bk["rows"], bk["ref"]
    names = list(rows[0]["gain"])
    print(f"\n  [{label}] 질의를 BASE 점수로 5 등분. 기준선 = {ref}")
    print(f"    {'버킷':>4s} {'n':>4s} {'BASE':>8s} "
          + "".join(f"{n[:13]:>14s}" for n in names))
    for r in rows:
        print(f"    {r['bucket']:4d} {r['n']:4d} {r['base_mean']:8.4f} "
              + "".join(f"{r['gain'][n][0]:+14.4f}" for n in names))
    print(f"    {'낮은버킷 - 높은버킷':>18s} "
          + "".join(f"{bk['slope'][n]['bucket_first_minus_last']:+14.4f}"
                    for n in names))
    print(f"    {'상관(BASE, 이득)':>18s} "
          + "".join(f"{bk['slope'][n]['corr_query']:+14.4f}" for n in names))
    ok = {n: (bk["slope"][n]["bucket_first_minus_last"] > 0
              and bk["slope"][n]["corr_query"] < 0) for n in names}
    hit = [n for n, v in ok.items() if v]
    print(f"    -> 구멍을 메우는 arm: {hit or '없음'}")
    return bool(hit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()

    cfg = load_cfg(os.path.join(a.ckpt or a.out, "ckpt", "config.json"))
    cfg.out_root = a.out
    ev = json.load(open(os.path.join(cfg.log_dir, "03_evaluate.json"),
                        encoding="utf-8"))
    cmp_, tcmp = ev["compare"], ev.get("text_compare") or {}
    tm = ev["test_mean"]
    best = ev.get("best_local") or {}
    arm = a.arm or ev.get("arm") or "soup"

    print(f"[base] {cfg.base}")
    print(f"       K={cfg.k}  fleet_size={cfg.fleet_size}  shard={cfg.shard}  "
          f"질의 {cfg.n_train + cfg.n_sel + cfg.n_check + cfg.n_test}")
    if cfg.target_provider == "reference":
        print("*** Target 이 reference 다. 추출 충실도가 아니다. 보고하지 말 것.")

    g = ev.get("greedy") or {}
    if g:
        print(f"[greedy] 기준 {g.get('on')}  시작 {g.get('start')}  "
              f"채택 {g.get('picked')}")
        # 03 이 남기는 키는 build_check(만든 절반) 와 verify_check(남긴 절반)다.
        bc, vc = g.get("build_check") or {}, g.get("verify_check") or {}
        print(f"         soup {len(g.get('soup_members', []))}개 "
              f"{g.get('soup_members', [])}  |  "
              f"metamon 칸 {g.get('cells_taken')}/{g.get('cells_tried')}")
        print(f"         만든 절반  " + "  ".join(f"{k} {v:.5f}" for k, v in bc.items()))
        print(f"         검증 절반  " + "  ".join(f"{k} {v:.5f}" for k, v in vc.items()))

    # ------------------------------------------------ 전제
    sk = ev.get("shard_skew") or {}
    if sk:
        print(f"\n[전제] 조각 치우침   len_eta2 {sk['len_eta2']:.3f}   "
              f"feat_cosine {sk['feat_cosine']:.3f}")
        if sk["len_eta2"] < 0.05 and sk["feat_cosine"] > 0.90:
            print("       *** 조각이 IID 다. 아래 기여 1 은 시험된 것이 아니다.")
    ok_head = True
    for key, label in (("oracle_sim", "확률"), ("oracle_gen", "생성")):
        o = ev.get(key)
        if not o:
            continue
        print(f"  [상보성 {label}]  oracle {o['oracle']:.5f}   "
              f"최고 단일 {o['best_local_mean']:.5f}   "
              f"headroom {o['headroom']:+.5f}   "
              f"최상위 승률 {o['dominance']*100:.1f}% "
              f"(완전 상보 {100.0/cfg.k:.1f}%)")
        if o.get("recovery"):
            print("     회수율   " + "  ".join(
                f"{n} {r*100:+.1f}%" for n, r in
                sorted(o["recovery"].items(), key=lambda kv: -kv[1])))
        if o["headroom"] <= 1e-4:
            ok_head = False

    # ------------------------------------------------ 기여 1
    print(f"\n[기여 1] 병합({arm})이 최고 단일과 {cfg.all_name} 을 이기는가")
    print(f"  최고 단일  avgBF {best.get('sim')}   손실 {best.get('loss')}   "
          f"생성 {best.get('gen')}")

    print(f"\n  확률 (eq:sim)")
    ok_sim = {}
    for tag in ("best_local(sim)", "best_local(loss)", cfg.all_name,
                "local 평균", "soup"):
        k = f"{arm} - {tag}"
        if k in cmp_:
            d, lo, hi = cmp_[k]
            ok_sim[tag] = lo > 0
            print(f"    {k:38s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {_mark(lo, hi)}")

    print(f"\n  생성 (Target 응답 대비, paired bootstrap)   <- 본 판정")
    ok_gen = {}
    if tcmp:
        for tag in ("best_local(gen)", "best_local(loss)", cfg.all_name):
            k = f"{arm} - {tag}"
            if k in tcmp:
                for m, (d, lo, hi) in tcmp[k].items():
                    if m == "ROUGE-L":
                        ok_gen[tag] = lo > 0
                    print(f"    {k:30s} {m:8s} {d:+.5f}  "
                          f"[{lo:+.5f}, {hi:+.5f}]  {_mark(lo, hi)}")
    else:
        print("    생성 비교 없음. --text 로 다시 돌릴 것.")
    claim1 = bool(ok_gen.get("best_local(gen)")) and \
        bool(ok_gen.get(cfg.all_name))

    # ------------------------------------------------ 기전 (버킷)
    print(f"\n[기전] 이득이 BASE 가 약한 질의에 몰리는가")
    h1 = _bucket(ev.get("bucket_sim"), "확률")
    h2 = _bucket(ev.get("bucket_gen"), "생성")
    mech = bool(h2 if h2 is not None else h1)

    # ------------------------------------------------ 표
    text = ev.get("text") or {}
    victim = ev.get("victim_text") or {}
    if text and any("vs_ref" in (v or {}) for v in text.values()):
        head = [m for m in METRICS
                if any(m in (v.get("vs_ref") or {}) for v in text.values())]
        label = {"__base__": "Basic theta (BASE)", "soup": "Uniform Merge",
                 cfg.all_name: "All-query LoRD", "greedy": "METAMON"}
        order = ([n for n in text if n.startswith("local_")]
                 + [n for n in text if not n.startswith("local_")
                    and n != "__base__"])
        for key, title in (("vs_ref", "[E1] 데이터셋 정답(ref) 대비 — "
                                      "LoRD Table 1 과 같은 기준"),
                           ("vs_target", "[추출 충실도] Target 응답(gold) 대비")):
            print(f"\n  {title}")
            print(f"    {'Method':30s}" + "".join(f"{m:>10s}" for m in head)
                  + (f"{'F(ROUGE-L)':>12s}" if key == "vs_ref" else ""))
            if key == "vs_ref" and victim:
                print(f"    {'Target Model':30s}"
                      + "".join(f"{victim.get(m, float('nan')):>10.4f}"
                                for m in head) + f"{1.0:>12.3f}")
            for nm in ["__base__"] + order:
                v = (text.get(nm) or {}).get(key)
                if not v:
                    continue
                f = (text[nm].get("F") or {}).get("ROUGE-L", float("nan"))
                print(f"    {label.get(nm, nm):30s}"
                      + "".join(f"{v.get(m, float('nan')):>10.4f}" for m in head)
                      + (f"{f:>12.3f}" if key == "vs_ref" else ""))
        print(f"\n    LoRD 논문 (Llama-3-8B, 질의 16): "
              f"BLEU-4 0.249  ROUGE-L 0.538  BERT 0.906  F(ROUGE-L) 0.891")
        print(f"    LoRD 논문 BASE                : "
              f"BLEU-4 0.105  ROUGE-L 0.348  BERT 0.868  F(ROUGE-L) 0.576")

    # ------------------------------------------------ 기여 2
    print(f"\n[기여 2] 데이터량을 맞춘 짝: fleet_g 대 union_g")
    dep = ev.get("dependency") or {}
    pos = sum(1 for k, v in cmp_.items()
              if k.startswith("fleet_") and "union" in k and v[1] > 0)
    neg = sum(1 for k, v in cmp_.items()
              if k.startswith("fleet_") and "union" in k and v[2] < 0)
    # 두 가지를 따로 본다. 평균이 오르는 것과 분산이 주는 것은 다른 주장이다.
    claim2a = pos == cfg.n_fleet and cfg.n_fleet > 0      # 평균
    claim2b = bool(dep.get("ok"))                          # 분산
    claim2 = claim2a
    if dep:
        vals = dep.get("values") or {}
        mu = lambda k: (sum(vals[k]) / len(vals[k])) if vals.get(k) else float("nan")
        print(f"  (a) 성능   union 평균 {mu('union'):.5f}  ->  "
              f"merged 평균 {mu('merged'):.5f}   "
              f"{'병합이 높다' if mu('merged') > mu('union') else '병합이 낮다'}")
        print(f"      같은 데이터에서 병합이 이긴 묶음 {pos}/{cfg.n_fleet}"
              f"  (진 묶음 {neg})   {'성립' if claim2a else '불성립'}")
        print(f"  (b) 분산   union {dep['union']:.3e}  ->  "
              f"merged {dep['merged']:.3e}   "
              f"{'성립' if claim2b else '불성립'}   (single {dep['single']:.3e})")
        if cfg.n_fleet < 5:
            print(f"      *** 묶음이 {cfg.n_fleet}개다. (b) 의 분산 추정이 얇다. "
                  f"(a) 가 더 믿을 만하다.")

    # ------------------------------------------------ 비용
    c = ev.get("cost") or {}
    if c:
        print(f"\n[비용]  surrogate {c['surrogate_params']/1e9:.2f}B  "
              f"Target {c['target']}  질의 {c['queries_total']}  "
              f"추론 병합 1배 vs 앙상블 {c['inference_ensemble']}배")
    if ev.get("ensemble_mean"):
        print(f"       앙상블 {ev['ensemble_mean']:.5f}  vs  "
              f"{arm}(1배) {tm.get(arm, float('nan')):.5f}")

    # ------------------------------------------------ 결론
    print(f"\n[결론]  base = {cfg.base}")
    print(f"  전제 (상보성)     {'있다' if ok_head else '없다'}"
          + ("" if ok_head else "   조각이 IID 다. 기여 1 은 미시험"))
    print(f"  기여 1 (충실도)   {'성립' if claim1 else '불성립'}"
          + ("" if claim1 else "   생성에서 최고 단일과 all 을 이겨야 한다"))
    print(f"  기전  (구멍 메움) {'성립' if mech else '불성립'}"
          + ("" if mech else "   이득이 낮은 버킷에 몰려야 한다"))
    print(f"  기여 2 (같은 데이터) {'성립' if claim2 else '불성립'}"
          + f"   성능 {'O' if claim2 else 'X'} / 분산 {'O' if claim2b else 'X'}")
    if claim1 and mech:
        print("  -> 다음 크기 점으로 넘어갈 단계. 같은 표를 base 를 바꿔 채운다.")
    print(f"  구간은 고정된 checkpoint 의 표본 불확실성이다. "
          f"학습 시드 반복을 대신하지 않는다.")


if __name__ == "__main__":
    main()
