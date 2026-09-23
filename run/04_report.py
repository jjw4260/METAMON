# -*- coding: utf-8 -*-
"""실행 4. 판정.

    python run/04_report.py --out runs/gpt35

주장이 둘이다. 둘 다 미리 정한 비교로만 판정한다. 보조 지표로 대신하지 않는다.

  주장 1 (충실도)
      병합 - 최고 단일 surrogate > 0
      확률(eq:sim) 에서 양수여야 하고, 생성 문장(BLEU/ROUGE-L) 에서도
      뒤지지 않아야 한다. 확률만 높고 문장이 안 비슷하면 주장이 아니다.

  주장 2 (종속성)
      Dependency(merged) < Dependency(union)
      둘은 본 데이터가 같다. 차이가 병합 자체의 효과다.
      union 대조가 없는 숫자는 "데이터를 더 봤을 뿐" 으로 반박당한다.

전체 질의 단일 모델(<fleet>_all) 이 앞서는 것은 한계로 보고하면 된다.
논지는 "병합이 최선" 이 아니라 "surrogate 를 하나 고르는 것이 불안정하고
병합이 그 불안정을 없앤다" 이므로 무너지지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.config import load as load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="soup",
                    choices=["soup", "metamon_cell", "metamon_layer", "weighted"])
    a = ap.parse_args()

    cfg = load_cfg(os.path.join(a.out, "ckpt", "config.json"))
    cfg.out_root = a.out
    ev = json.load(open(os.path.join(cfg.log_dir, "03_evaluate.json"),
                        encoding="utf-8"))
    cmp_ = ev["compare"]
    tm = ev["test_mean"]
    best_local = ev["best_local"]
    arm = a.arm

    if cfg.target_provider == "reference":
        print("*** Target 이 reference 다. 이 수치는 추출 충실도가 아니다. "
              "보고하지 말 것.\n")

    # ------------------------------------------------ 주장 1. 충실도
    print(f"[주장 1] 병합({arm}) 이 최고 단일 surrogate 보다 Target 을 잘 재현하는가")
    t = f"{arm} - best_local"
    ok1 = False
    if t in cmp_:
        d, lo, hi = cmp_[t]
        ok1 = lo > 0
        print(f"  eq:sim  {t:28s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  "
              f"{'양수' if lo > 0 else ('음수' if hi < 0 else '불확실')}")
    else:
        print(f"  {t} 비교가 없다. 03 의 --arm 설정을 확인할 것.")

    text = ev.get("text") or {}
    ok_text = None
    if text:
        keys = ("BLEU-1", "BLEU-4", "ROUGE-L", "BERT-F1")
        head = [k for k in keys if any(k in v for v in text.values())]
        print(f"  생성 문장 vs Target 응답")
        print(f"    {'arm':16s}" + "".join(f"{h:>10s}" for h in head))
        for nm, v in text.items():
            label = "base" if nm == "__base__" else nm
            print(f"    {label:16s}" +
                  "".join(f"{v.get(h, float('nan')):10.2f}" for h in head))
        if arm in text and best_local in text:
            ok_text = text[arm]["ROUGE-L"] >= text[best_local]["ROUGE-L"]
            print(f"    -> ROUGE-L 에서 {arm} 이 best_local 에 "
                  f"{'앞선다' if ok_text else '뒤진다'}")
        for nm in (arm, best_local):
            for s in (text.get(nm, {}).get("sample") or [])[:1]:
                print(f"    [{nm} 예시]")
                print(f"      Target    {s['target'][:90]}")
                print(f"      surrogate {s['surrogate'][:90]}")
    else:
        print("  생성 비교 없음. run/03_evaluate.py --text 로 다시 돌릴 것.")

    # ------------------------------------------------ 주장 2. 종속성
    dep = ev.get("dependency") or {}
    print(f"\n[주장 2] 병합이 surrogate 선택 의존을 줄이는가")
    ok2 = bool(dep.get("ok"))
    if dep:
        for tag, key in (("single  (조각 1개, 병합 없음)", "single"),
                         (f"union   (조각 {cfg.fleet_size}개, 병합 없음)", "union"),
                         (f"merged  (조각 {cfg.fleet_size}개, 병합 있음)", "merged"),
                         ("loo     (겹침, 판정 제외)", "loo")):
            v = dep.get(key)
            if v is not None:
                print(f"  Dependency {tag:34s} {v:.3e}")
        u, m = dep.get("union"), dep.get("merged")
        if u and m:
            print(f"  merged < union  {'성립' if ok2 else '불성립'}  "
                  f"({u / max(m, 1e-30):.1f}배)")
        n_f = len(dep.get("values", {}).get("merged", []))
        if n_f < 3:
            print(f"  *** merged 가 {n_f} 개뿐이다. 분산 추정이 얇다.")
        # 데이터량을 맞춘 짝별 충실도
        pos = sum(1 for g in range(cfg.n_fleet)
                  if cmp_.get(f"fleet_{g} - union_{g} (데이터 동일)", [0, 0, 0])[1] > 0)
        print(f"  같은 데이터에서 fleet > union 인 묶음 {pos}/{cfg.n_fleet}")
    else:
        print("  종속성 결과가 없다.")

    # ------------------------------------------------ 비용
    c = ev.get("cost") or {}
    if c:
        print(f"\n[비용]")
        print(f"  surrogate {c['surrogate']}  {c['surrogate_params']/1e9:.2f}B")
        print(f"  Target    {c['target']}")
        print(f"  질의      {c['queries_total']}")
        print(f"  추론      병합 {c['inference_merged']}배(모델 1개)  vs  "
              f"앙상블 {c['inference_ensemble']}배(모델 {c['models_kept_ensemble']}개)")
    if ev.get("ensemble_mean"):
        print(f"  앙상블 test {ev['ensemble_mean']:.5f}  vs  "
              f"{arm}(비용 1배) {tm.get(arm, float('nan')):.5f}")

    # ------------------------------------------------ 한계
    print(f"\n[한계로 보고할 것]")
    t = f"{arm} - {cfg.all_name}"
    if t in cmp_:
        d, lo, hi = cmp_[t]
        print(f"  {t:34s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]"
              + ("  전체 질의 단일 모델이 앞선다" if hi < 0 else ""))
    print(f"  구간은 고정된 checkpoint 의 표본 불확실성이다. "
          f"학습 시드 반복을 대신하지 않는다.")

    # ------------------------------------------------ 결론
    print(f"\n[결론]")
    if ok1 and ok2:
        print("  두 주장 모두 성립. 시드 반복과 두 번째 과제(subset) 로 확장할 단계.")
    elif ok2 and not ok1:
        print("  종속성만 성립. 충실도 개선은 주장에서 빼고 종속성 완화 단독으로 쓸 것.")
    elif ok1 and not ok2:
        print("  충실도만 성립. 종속성은 union 대조를 통과하지 못했다. "
              "n_fleet 을 늘리거나 주장을 충실도로 좁힐 것.")
    else:
        print("  둘 다 성립하지 않는다. 설정을 바꾸기 전에 fleet 학습 로그부터 볼 것.")
    if text and ok_text is False:
        print("  생성 문장에서 뒤진다. 확률만 높은 상태이므로 "
              "'비슷한 답변을 낸다' 는 주장은 아직 못 한다.")


if __name__ == "__main__":
    main()
