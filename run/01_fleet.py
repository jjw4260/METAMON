# -*- coding: utf-8 -*-
"""실행 1. fleet 학습.

    python run/01_fleet.py --out runs/lord_k4 --fleet lord
    python run/01_fleet.py --out runs/sft_k4  --fleet sft

arm 하나가 BASE 를 개선하지 못하면 그 자리에서 멈춘다. 다음 arm 으로 넘어가
시간을 버리지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.config import Config
from Pipeline.data import Splits, load_dataset_file
from Pipeline.fleet import build_fleet
from Pipeline.metrics import EvalSet
from Pipeline.modeling import WeightSpace, load_tokenizer, setup_precision, gpu_free_gb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--fleet", default="lord", choices=["lord", "sft"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fleet-size", type=int, default=2,
                    help="독립 fleet 하나에 묶을 Local 수. n_fleet = k / 이 값")
    ap.add_argument("--periods", type=int, default=None,
                    help="비우면 arm 의 질의 수에 맞춰 자동으로 잡는다")
    ap.add_argument("--lord-variant", default=None, choices=["code", "paper"],
                    help="code=train_pod2.py:963 그대로(기본), paper=Eq.10 의 clip 포함")
    ap.add_argument("--lambda1", type=float, default=None)
    # 분할 크기는 여기서 고정되어 ckpt/config.json 에 박힌다. 실행 2, 3 은
    # 그 파일을 읽으므로 나중에 바꿀 수 없다. 실행 0 의 질의 수와 맞출 것.
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-sel", type=int, default=None,
                    help="실행 2 비용이 여기에 비례한다")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    cfg = Config(out_root=a.out, fleet_method=a.fleet, k=a.k, seed=a.seed,
                 fleet_size=a.fleet_size)
    if a.n_train is not None:
        cfg.n_train = a.n_train
    if a.n_sel is not None:
        cfg.n_sel = a.n_sel
    if a.periods is not None:
        cfg.periods = a.periods
    if a.lord_variant is not None:
        cfg.lord_variant = a.lord_variant
    if a.lambda1 is not None:
        cfg.lambda1 = a.lambda1
    cfg.makedirs()
    if cfg.n_fleet < 3:
        print(f"  *** n_fleet = {cfg.n_fleet}. 종속성 분산 추정이 얇다. "
              f"k 를 늘리거나 fleet_size 를 줄일 것.")

    # Target 설정은 실행 0 이 남긴 것을 그대로 따른다. 설정 해시에 포함된다.
    tmeta_path = os.path.join(cfg.log_dir, "00_target.json")
    if not os.path.exists(cfg.dataset_path):
        raise SystemExit(
            f"Target 응답이 없다: {cfg.dataset_path}\n"
            f"  먼저 run/00_target.py 를 실행할 것.")
    tmeta = json.load(open(tmeta_path, encoding="utf-8")) \
        if os.path.exists(tmeta_path) else {}
    for k_, f_ in (("provider", "target_provider"), ("model", "target_model"),
                   ("temperature", "target_temperature"),
                   ("max_tokens", "target_max_tokens"), ("subset", "subset")):
        if k_ in tmeta:
            setattr(cfg, f_, tmeta[k_])

    device = setup_precision()
    tok = load_tokenizer(cfg)
    sp = Splits(cfg, load_dataset_file(cfg.dataset_path, cfg, tok))
    sp.assert_clean()
    cfg.guard({"query_hash": sp.query_hash(), "shard_hash": sp.shard_hash(),
               "target": tmeta})

    print("[설정]")
    for k, v in cfg.to_dict().items():
        print(f"  {k:16s} {v}")
    print(f"  분할             {sp.summary()}")
    print(f"  Target           {cfg.target_provider} / {cfg.target_model}")
    print(f"  독립 fleet        {cfg.n_fleet}개 x {cfg.fleet_size}   {cfg.fleets}")
    print(f"  학습할 arm        {len(cfg.arm_names)}개  {cfg.arm_names}")
    print(f"  GPU 여유          {gpu_free_gb():.1f}GB")
    if cfg.target_provider == "reference":
        print("  *** Target 응답이 데이터셋 정답 문장이다. 추출 충실도가 아니다.")

    ws = WeightSpace(cfg, device)
    sel = EvalSet(sp.sel, tok, device, cfg.eval_bs)
    build_fleet(ws, tok, cfg, sp, sel)

    json.dump({"cfg": cfg.to_dict(), "query_hash": sp.query_hash(),
               "shard_hash": sp.shard_hash(), "overlap": sp.overlap()},
              open(os.path.join(cfg.log_dir, "01_fleet.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n완료. checkpoint: {cfg.ckpt_dir}")


if __name__ == "__main__":
    main()
