# -*- coding: utf-8 -*-
"""실행 2. 공통 가중치 공간 구성과 기여도 측정, 선택, 통합 검증.

    python run/02_contribution.py --out runs/lord_k4

여기서 나온 PartialScore / Scale / 선택 결과를 저장한다. 실행 3 은 이것만
읽어서 배율과 최종 비교를 수행하므로, 조립 방식을 바꿔 볼 때 재측정이 필요 없다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metamon.config import Config, assert_same_data, load as load_cfg
from metamon.contribution import measure, verify_assembly
from metamon.data import Splits, load_dataset_file
from metamon.fleet import load_deltas, verify_restore
from metamon.metrics import EvalSet, loss, sim, soft_weight
from metamon.modeling import WeightSpace, gpu_free_gb, load_tokenizer, setup_precision
from metamon.weightspace import Candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--omega", default="uniform", choices=["uniform", "soft"],
                    help="eq:weighted_loss 의 omega. soft 는 eq:soft_weight")
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
    sel = EvalSet(sp.sel, tok, device, cfg.eval_bs)
    print(f"[시작] GPU 여유 {gpu_free_gb():.1f}GB   base L(theta;1) "
          f"{loss(ws.model, sel):.5f}")

    deltas = load_deltas(ws, cfg)
    verify_restore(ws, cfg, deltas, sel)

    cand = Candidates(deltas, cfg.local_names, ws.keys, device)
    worst_cos = cand.diversity_gate(cfg.cos_max)
    torch.cuda.empty_cache()

    # eq:soft_weight 는 후보마다 다른 omega_k 를 준다. 하나로 평균하면 안 된다.
    omegas = None
    if a.omega == "soft":
        sims = []
        for n in cfg.local_names:
            ws.apply(lambda k, d=deltas[n]: d[k].to(device), 1.0)
            sims.append(sim(ws.model, sel))
            ws.reset()
        omegas = soft_weight(np.stack(sims), cfg.beta)      # (K, N)
        print(f"  omega_k 행별 합 = {omegas.sum(axis=1).round(2).tolist()}  "
              f"(질의마다 K 에 대해 정규화되어 전체 합은 {sel.n})")

    con = measure(ws, cfg, cand, sel, omegas)

    win_cell = con.winner_cell()
    win_layer, conf, ls = con.select_layer(ws.layers, cfg.roles, cfg.k)
    occ_cell = con.occupancy(win_cell, cfg.k)
    occ_layer = con.occupancy(win_layer, cfg.k)
    agree = sum(1 for k in ws.keys if win_cell[k] == win_layer[k[0]]) / len(ws.keys)
    print(f"  [칸]   점유 {occ_cell}  PartialScore=0 인 칸 "
          f"{con.zero_cells()}/{len(ws.keys)}  "
          f"중앙값 {np.median([max(v) for v in con.score.values()]):.3e}")
    print(f"  [layer] 점유 {occ_layer}  Confidence 중앙값 "
          f"{np.median(list(conf.values())):.3e}  칸과 일치 {agree*100:.1f}%")

    keep = verify_assembly(ws, sel, con, cand, win_layer, ls)

    out = {
        "base_loss": con.base_loss, "omega": a.omega, "worst_cosine": worst_cos,
        "base_per_candidate": con.base_per_candidate,
        "query_hash": sp.query_hash(), "shard_hash": sp.shard_hash(),
        "partial_score": {str(k): v for k, v in con.score.items()},
        "alpha": {str(k): v for k, v in con.alpha.items()},
        "winner_cell": {str(k): v for k, v in win_cell.items()},
        "winner_layer": {str(k): v for k, v in win_layer.items()},
        "confidence": {str(k): v for k, v in conf.items()},
        "kept_layers": sorted(keep),
        "occupancy_cell": occ_cell, "occupancy_layer": occ_layer,
        "agree_cell_layer": agree, "zero_cells": con.zero_cells(),
        "cand_norm_median": {str(k): v for k, v in cand.median.items()},
    }
    path = os.path.join(cfg.log_dir, "02_contribution.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
