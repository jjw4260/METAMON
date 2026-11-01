# -*- coding: utf-8 -*-
"""E3 Resource Efficiency. 단일 대형 Local 하나를 LoRD 로 학습해 METAMON 과 비교한다.

    python run/h7_single_large.py --data .../<task>/target/dataset.json \\
        --out .../<task>/single_large --metamon .../<task>/hetero

질의 예산을 맞춘다. METAMON 의 공격자 쪽은 theta (X_theta 1024) 와 Local 3 개
(각 1024) 로 Target 에 모두 4096 번 질의했다. 단일 대형 Local 은 같은 4096 질의
(train 전체) 로 학습한다. 학습 설정은 h1 과 같다 (LoRD-VI, lord_lr, EOS, sel 최적 시점).
7B 이상은 fp32 AdamW 상태가 80GB 를 넘으므로 --optim paged_adamw8bit 로 상태만 8-bit 로 둔다.

보고 (out/)
  test 512 의 L, avgBF (자기 tokenizer), test 앞 256 생성의 BLEU-4 / ROUGE-L / BERTScore
  (Target 응답 대비, 데이터셋 정답 대비), METAMON 생성과의 paired bootstrap 차이,
  LoRD 학습 GPU 시간 (단일 대형 vs METAMON 의 theta + Local 학습 합) 과 추론 파라미터 수.
  report_single_large.md / .json, hyps.json
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from Pipeline.config import Config
from Pipeline.data import load_dataset_file
from Pipeline.hetero.load import load_tok, load_trained
from Pipeline.metrics import EvalSet, token_avg_logp
from Pipeline.modeling import setup_precision
from Pipeline.textgen import generate, score_text, text_compare, victim_scores


def short(name: str) -> str:
    return name.split("/")[-1]


def load_h1():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1_train.py")
    spec = importlib.util.spec_from_file_location("h1_train", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metamon", required=True, help="METAMON run 폴더 (manifest.json, h5/<plan>/hyps.json)")
    ap.add_argument("--plan", default="ot")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--optim", default="paged_adamw8bit", choices=["adamw", "adamw8bit", "paged_adamw8bit"])
    ap.add_argument("--lord-micro", type=int, default=2)
    ap.add_argument("--local-dir", default="/content/hetero_local/h7")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--text-n", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()
    s = short(a.model)

    # ---- 학습 (끝난 것이 있으면 건너뛴다)
    info_path = os.path.join(a.out, "train_info.json")
    if os.path.exists(info_path):
        info = json.load(open(info_path, encoding="utf-8"))
        print(f"[학습] 저장본 사용: {info['delta']}  ({info['sec'] / 3600:.2f} GPU-h)")
    else:
        h1 = load_h1()
        ns = SimpleNamespace(local_dir=a.local_dir, data=a.data, out=a.out, lord_lr=None,
                             lord_micro=a.lord_micro, optim=a.optim)
        print(f"[학습] {a.model}  X 전체 {a.n_x} 질의  optim {a.optim}  lord_micro {a.lord_micro}", flush=True)
        info = h1.train_one(a.model, ns, device, idx=None)
        json.dump(info, open(info_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 평가
    tok = load_tok(a.model)
    cfg = Config(base=a.model, dataset_file=os.path.abspath(a.data), gold_eos=True)
    items = load_dataset_file(cfg.dataset_path, cfg, tok, log=lambda *_: None)
    t0_ = a.n_x + a.n_sel + a.n_check
    test = items[t0_:t0_ + a.n_test]
    pool = items[t0_:t0_ + a.text_n]
    gold = [x["gold"].strip() for x in pool]
    ref = [x["ref"] for x in pool]

    arch, _ = load_trained(a.model, info["delta"], device)
    model = arch.model
    lp = token_avg_logp(model, EvalSet(test, tok, device, cfg.eval_bs))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[test] L {-lp.mean():.5f}  avgBF {np.exp(lp).mean():.4f}  params {n_params / 1e9:.2f}B")

    hyp_path = os.path.join(a.out, "hyps.json")
    hyps = json.load(open(hyp_path, encoding="utf-8")) if os.path.exists(hyp_path) else {}
    if hyps.get("_pool") != [t0_, len(pool)] or "single" not in hyps:
        model.config.use_cache = True
        model.generation_config.max_length = None
        print("  생성 중 ...", flush=True)
        hyps = {"_pool": [t0_, len(pool)], "single": generate(model, tok, pool, cfg, device)}
        json.dump(hyps, open(hyp_path, "w", encoding="utf-8"), ensure_ascii=False)
    del model, arch
    gc.collect()
    torch.cuda.empty_cache()

    victim = victim_scores(gold, ref, cfg)
    text = score_text(hyps["single"], gold, ref, cfg, victim=victim, tag=f"Single {s}"[:15])

    # ---- METAMON 과 비교
    mh_path = os.path.join(a.metamon, "h5", a.plan, "hyps.json")
    cmp_ = None
    if os.path.exists(mh_path):
        mh = json.load(open(mh_path, encoding="utf-8"))
        if mh.get("_pool") == [t0_, len(pool)] and "theta' (METAMON)" in mh:
            cmp_ = text_compare(mh["theta' (METAMON)"], hyps["single"], gold, tag="METAMON - Single Large")
    man = json.load(open(os.path.join(a.metamon, "manifest.json"), encoding="utf-8"))
    fleet_sec = float(man["theta"].get("sec", 0.0)) + sum(float(v.get("sec", 0.0)) for v in man["locals"].values())

    vt, vr, F = text["vs_target"], text["vs_ref"], text.get("F", {})
    md = [f"# E3 - Single Large ({a.model}) vs METAMON\n",
          f"- Single Large: LoRD on all {info.get('n_train', a.n_x)} train queries (same query budget as METAMON: "
          f"theta 1024 + 3 Locals x 1024); optim {a.optim}",
          f"- LoRD training GPU-hours: Single Large {info['sec'] / 3600:.2f} h, "
          f"METAMON theta + Locals {fleet_sec / 3600:.2f} h (h2-h4 not included)",
          f"- inference params: Single Large {n_params / 1e9:.2f}B\n",
          "| Metric | Single Large |", "|---|---|",
          f"| test L (own tokenizer) | {-lp.mean():.5f} |",
          f"| test avgBF (own tokenizer) | {np.exp(lp).mean():.4f} |",
          f"| vs Target BLEU-4 / ROUGE-L / BERTScore | {vt['BLEU-4']:.4f} / {vt['ROUGE-L']:.4f} / {vt.get('BERT-F1', float('nan')):.4f} |",
          f"| vs ref BLEU-4 / ROUGE-L / BERTScore / F | {vr['BLEU-4']:.4f} / {vr['ROUGE-L']:.4f} / "
          f"{vr.get('BERT-F1', float('nan')):.4f} / {F.get('ROUGE-L', float('nan')):.3f} |"]
    if cmp_:
        rl, b4 = cmp_["ROUGE-L"], cmp_["BLEU-4"]
        md += ["", f"- METAMON − Single Large (vs Target): ROUGE-L {rl[0]:+.4f} [{rl[1]:+.4f}, {rl[2]:+.4f}], "
                   f"BLEU-4 {b4[0]:+.4f} [{b4[1]:+.4f}, {b4[2]:+.4f}]"]
    from Pipeline import tasks as TK
    task_sc = None
    if not TK.is_wmt(cfg.task):
        metric = TK.TASKS[cfg.task]["metric"]
        if metric == "choice" and all("label" in x for x in pool):
            task_sc = TK.choice_scores([TK.parse_choice(h) for h in hyps["single"]], [int(x["label"]) for x in pool])
            md.append(f"- choice accuracy {task_sc['Accuracy']:.4f}  F1 {task_sc['F1']:.4f}")
        elif metric == "safety":
            task_sc = TK.toxicity(hyps["single"], device)
            md.append("- toxicity x100 " + ", ".join(f"{k} {v:.2f}" for k, v in task_sc.items()))
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)
    open(os.path.join(a.out, "report_single_large.md"), "w", encoding="utf-8").write(md_text)
    json.dump({"model": a.model, "train": info, "test_L": float(-lp.mean()),
               "test_avgBF": float(np.exp(lp).mean()), "params": n_params, "text": text,
               "compare_metamon_minus_single": {k: list(v) for k, v in (cmp_ or {}).items()},
               "fleet_train_sec": fleet_sec, "task": task_sc},
              open(os.path.join(a.out, "report_single_large.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"저장: {a.out}/report_single_large.md")


if __name__ == "__main__":
    main()
