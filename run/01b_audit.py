# -*- coding: utf-8 -*-
"""저장된 arm 을 다시 재서 쓸 수 있는지 판정한다.

    python run/01b_audit.py --out runs/l32

`01_fleet.py` 의 생존 관문은 배율 1.0 과 0.5 중 **더 좋은 쪽**을 보고 통과를
찍는다. 그런데 실행 3 의 `single(arm)` 은 배율 1.0 을 쓴다. 그래서 "통과" 가
"배율 1.0 에서 BASE 보다 낫다" 를 뜻하지 않는다. 그 차이가 그대로 비교에
들어가면 union_g 가 쓰레기라서 fleet_g 가 이기는 일이 생긴다.

여기서는 arm 마다 배율별 L(theta;1) 을 전부 찍고, 배율 1.0 에서 BASE 를 넘지
못한 arm 을 이름으로 모아 돌려준다. 학습은 하지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline import config as C
from Pipeline.data import Splits, load_dataset_file
from Pipeline.deltastore import DeltaStore, st_path
from Pipeline.metrics import EvalSet, loss
from Pipeline.modeling import WeightSpace, load_tokenizer, setup_precision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="1.0,0.5,0.25")
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",")]

    cfg = C.load(os.path.join(a.out, "ckpt", "config.json"))
    cfg.out_root = a.out
    device = setup_precision()
    tok = load_tokenizer(cfg)
    sp = Splits(cfg, load_dataset_file(cfg.dataset_path, cfg, tok))
    ws = WeightSpace(cfg, device)
    sel = EvalSet(sp.sel, tok, device, cfg.eval_bs)

    ws.reset()
    base = loss(ws.model, sel)
    print(f"\n[감사] {cfg.base}   BASE L(theta;1) = {base:.5f}")
    print(f"  {'arm':12s} " + "  ".join(f"{'L@'+str(s):>9s}" for s in scales)
          + f"  {'최선':>9s}  판정")

    bad, missing = [], []
    for name in cfg.arm_names:
        if not os.path.exists(st_path(cfg, name)):
            missing.append(name)
            print(f"  {name:12s} 없음")
            continue
        st = DeltaStore(ws, cfg, [name], log=lambda *_: None)
        v = {}
        for s in scales:
            ws.apply(lambda k: st.raw(name, k), s)
            v[s] = loss(ws.model, sel)
        ws.reset()
        st.close()
        ok = v[1.0] < base
        if not ok:
            bad.append(name)
        print(f"  {name:12s} " + "  ".join(f"{v[s]:9.5f}" for s in scales)
              + f"  {min(v.values()):9.5f}  "
              + ("쓸 수 있다" if ok else "*** 배율 1.0 에서 BASE 보다 나쁘다"))

    print()
    if missing:
        print(f"  없는 arm: {missing}")
    if bad:
        print(f"  다시 학습할 arm {len(bad)}개:\n    {' '.join(bad)}")
        print(f"  지우는 명령:")
        print("    " + " ".join(f"rm {st_path(cfg, n)}" for n in bad))
    else:
        print(f"  모든 arm 이 배율 1.0 에서 BASE 보다 낫다. 실행 2 로 간다.")


if __name__ == "__main__":
    main()
