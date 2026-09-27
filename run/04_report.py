# -*- coding: utf-8 -*-
"""실행 4. 판정.

    python run/04_report.py --out runs/gpt35

주장이 둘이다. 둘 다 미리 정한 비교로만 판정한다.

  주장 1 (충실도)  병합 - 최고 단일 surrogate > 0
      **확률과 생성 양쪽에서** 봐야 하고, 최고 단일을 고르는 기준도 보고하는
      지표와 같아야 한다. avgBF 로 고르면 생성에서 5 위인 Local 이 뽑힌다.
      확률에서 이기고 생성에서 지면 성립이 아니다. LoRD Table 1 이 생성이다.

  주장 2 (종속성)  병합 크기가 커지면 종속성이 준다
      cm{m}_{g} 는 겹치지 않는 묶음이다. m 이 커지며 분산이 단조로 줄어야 한다.
      union_g 대조로 "데이터를 더 봐서 준 것" 을 배제한다.
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

    if cfg.target_provider == "reference":
        print("*** Target 이 reference 다. 추출 충실도가 아니다. 보고하지 말 것.\n")

    g = ev.get("greedy") or {}
    if g:
        print(f"[greedy] soup 구성 {len(g.get('soup_members', []))}개 "
              f"{g.get('soup_members')}")
        print(f"         metamon 시작 {g.get('start')}  "
              f"칸 {g.get('cells_taken')}/{g.get('cells_tried')} 채택")

    # ------------------------------------------------ 주장 1
    print(f"\n[주장 1] 병합({arm}) 이 최고 단일 surrogate 를 이기는가")
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

    print(f"\n  생성 (Target 응답 대비, paired bootstrap)")
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
        print("    생성 비교 없음. run/03_evaluate.py --text 로 다시 돌릴 것.")

    claim1 = bool(ok_sim.get("best_local(loss)")) and \
        bool(ok_gen.get("best_local(gen)"))

    # ------------------------------------------------ 표
    text = ev.get("text") or {}
    victim = ev.get("victim_text") or {}
    if text and any("vs_ref" in (v or {}) for v in text.values()):
        head = [m for m in METRICS
                if any(m in (v.get("vs_ref") or {}) for v in text.values())]
        label = {"__base__": "Basic theta (Local Model)",
                 "soup": "Uniform Merge", cfg.all_name: "All-query LoRD"}
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
        for nm in (arm, best.get("gen")):
            for s in (text.get(nm, {}).get("sample") or [])[:1]:
                print(f"    [{nm} 예시]")
                print(f"      Target    {s['target'][:86]}")
                print(f"      surrogate {s['surrogate'][:86]}")

    # ------------------------------------------------ 주장 2
    print(f"\n[주장 2] 병합 크기가 커지면 surrogate 선택 의존이 주는가")
    cv = ev.get("dependency_curve") or {}
    rows = cv.get("rows") or []
    claim2 = False
    if rows:
        print(f"    {'m':>3s} {'묶음':>4s} {'분산':>11s} {'평균':>9s}")
        for r in rows:
            print(f"    {r['m']:3d} {r['n']:4d} {r['var']:11.3e} {r['mean']:9.5f}")
        f0, fl = rows[0], rows[-1]
        print(f"    m={f0['m']} -> m={fl['m']}  "
              f"{f0['var'] / max(fl['var'], 1e-30):.1f}배 감소   "
              f"{'단조 감소' if cv.get('monotone') else '단조가 아니다'}")
        claim2 = bool(cv.get("monotone")) and fl["var"] < f0["var"]
    dep = ev.get("dependency") or {}
    if dep.get("union") and dep.get("merged"):
        print(f"    데이터량 대조  union {dep['union']:.3e}  "
              f"merged {dep['merged']:.3e}  "
              f"{'merged 가 작다' if dep['merged'] < dep['union'] else '불성립'}")
        pos = sum(1 for k, v in cmp_.items()
                  if k.startswith(f"cm{cfg.fleet_size}_") and "union" in k
                  and v[1] > 0)
        print(f"    같은 데이터에서 병합이 이긴 묶음 {pos}/{cfg.n_fleet}")

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
    print(f"\n[결론]")
    print(f"  주장 1 (충실도)   {'성립' if claim1 else '불성립'}"
          + ("" if claim1 else "   확률과 생성 양쪽에서 최고 단일을 이겨야 한다"))
    print(f"  주장 2 (종속성)   {'성립' if claim2 else '불성립'}"
          + ("" if claim2 else "   병합 크기에 따라 분산이 단조로 줄어야 한다"))
    if claim1 and claim2:
        print("  둘 다 성립. 시드 반복과 두 번째 subset 으로 확장할 단계.")
    print(f"  구간은 고정된 checkpoint 의 표본 불확실성이다. "
          f"학습 시드 반복을 대신하지 않는다.")


if __name__ == "__main__":
    main()
