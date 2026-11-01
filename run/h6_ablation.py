# -*- coding: utf-8 -*-
"""이기종 6 단계. Contribution Selection ablation 의 조립 recipe 를 만든다 (E4).

    python run/h6_ablation.py --manifest .../hetero_abl/manifest.json --data .../dataset.json

h3 (PartialScore / Scale 측정) 와 h3b (omega, A 선택) 가 끝난 run 에서, Target 기준
기여도 측정을 빼고 Local 을 layer 에 배정하는 두 대조를 만든다.

  uniform   layer 마다 Local 을 돌아가며 배정한다 (k = l mod K). 모든 Local 이 같은 수의 layer 를 받는다.
  random    layer 마다 Local 을 무작위로 배정한다 (seed 별).

두 대조 모두 PartialScore 로 고르지 않으므로 Scale 은 h3b 가 고른 A 의 최댓값을 모든 칸에
쓰고, Confidence 는 모든 layer 에서 1 이다. 통합 검증 (eq:assembly_verification) 은 METAMON 과
같다 (sel, omega=1; 실패하면 layer 순서대로 손실을 줄이는 layer 만 남긴다).
w/o Behavioral OT (Layer Ratio) 는 이 스크립트가 아니라 h2 --plan ratio 부터 같은 단계를 돌려 만든다.

저장 (h6/<plan>/)
  select_uniform.json, select_random.json   h4 --recipe 로 쓰는 조립 recipe (select_check.json 과 같은 형식)
  report_ablation.md, report_ablation.json  조립 단계 (학습 전) 의 check / test L, theta·METAMON 대비 차이
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from safetensors import safe_open

from Pipeline.config import Config
from Pipeline.data import load_dataset_file
from Pipeline.hetero.contrib import TailEval
from Pipeline.hetero.load import load_tok, load_trained
from Pipeline.hetero.roles import ROLES
from Pipeline.metrics import paired_bootstrap
from Pipeline.modeling import setup_precision


def short(name: str) -> str:
    return name.split("/")[-1]


def save_text(a, name: str, text: str) -> None:
    tmp = os.path.join(a.local_dir, name)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    dst = os.path.join(a.out, name)
    shutil.copyfile(tmp, dst)
    if os.path.getsize(dst) != os.path.getsize(tmp):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", default="ot")
    ap.add_argument("--out", default=None, help="기본: manifest 옆 h6/<plan>")
    ap.add_argument("--local-dir", default="/content/hetero_local/h6")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    root = os.path.dirname(a.manifest)
    a.out = a.out or os.path.join(root, "h6", a.plan)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()

    sc_path = os.path.join(root, "h3", a.plan, "select_check.json")
    if not os.path.exists(sc_path):
        raise SystemExit(f"{sc_path} 가 없다. h3b 를 먼저 돌릴 것.")
    sc = json.load(open(sc_path, encoding="utf-8"))
    names = sc["names"]
    K = len(names)
    a_max = float(max(sc["alphas"]))
    man = json.load(open(a.manifest, encoding="utf-8"))
    tinfo = man["theta"]
    print(f"[recipe 기준] {sc_path}  omega {sc['omega']}  A_max {a_max:g}  후보 {[short(n) for n in names]}")

    ttok = load_tok(tinfo["name"])
    cfg = Config(base=tinfo["name"], dataset_file=os.path.abspath(a.data), gold_eos=True)
    titems = load_dataset_file(cfg.dataset_path, cfg, ttok, log=lambda *_: None)
    s0, c0 = a.n_x, a.n_x + a.n_sel
    t0_ = c0 + a.n_check
    sel, chk, test = titems[s0:c0], titems[c0:t0_], titems[t0_:t0_ + a.n_test]
    th, _ = load_trained(tinfo["name"], tinfo["delta"], device)
    L_t = th.L

    ev = {nm: TailEval(th, ttok, it, a.bs, device) for nm, it in
          (("sel", sel), ("check", chk), ("test", test))}
    base = {nm: e.mlp() for nm, e in ev.items()}
    L0 = -float(base["sel"].mean())
    print("[theta] " + "  ".join(f"{nm} L {-v.mean():.5f}" for nm, v in base.items()))

    zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                    framework="pt", device="cpu") for n in names]
    zkeys = [set(f.keys()) for f in zf]
    base_W = {(l, r): th.weight(l, r).detach().cpu().clone() for l in range(L_t) for r in ROLES}
    base_norm = {kk: float(v.norm()) for kk, v in base_W.items()}

    def restore():
        with torch.no_grad():
            for (l, r), w in base_W.items():
                th.weight(l, r).copy_(w.to(device))

    def apply(cells):
        restore()
        with torch.no_grad():
            for c in cells:
                W = th.weight(c["l"], c["role"])
                d = zf[c["k"]].get_tensor(f"{c['l']}.{c['role']}").to(device).float()
                W.add_(d, alpha=c["a"] * base_norm[(c["l"], c["role"])] / max(float(d.norm()), 1e-30))

    def cells_for(win, keep):
        return [{"l": int(l), "role": r, "k": int(win[l]), "a": a_max}
                for l in keep for r in ROLES if f"{l}.{r}" in zkeys[int(win[l])]]

    def assemble(win):
        """기여도 없이 배정한 win 으로 조립하고 METAMON 과 같은 통합 검증을 한다."""
        cells = cells_for(win, range(L_t))
        apply(cells)
        L_asm = -float(ev["sel"].mlp().mean())
        keep, cur = list(range(L_t)), L_asm
        if L_asm >= L0:
            keep, cur = [], L0
            for l in range(L_t):                    # 점수가 없으므로 layer 순서대로
                apply(cells_for(win, keep + [l]))
                v = -float(ev["sel"].mlp().mean())
                if v < cur:
                    keep, cur = keep + [l], v
            cells = cells_for(win, keep)
            apply(cells)
        return {"win": [names[int(k)] for k in win], "confidence": [1.0] * L_t,
                "layer_score": [[0.0] * K for _ in range(L_t)], "kept": sorted(keep), "cells": cells,
                "L_asm_sel": L_asm, "L_sel": cur,
                "occupancy": {short(names[k]): int(sum(1 for w in win if int(w) == k)) for k in range(K)}}

    def measure():
        return {nm: ev[nm].mlp() for nm in ("check", "test")}

    t_start = time.time()
    res = {}
    # METAMON 조립 (기준)
    apply(sc["main"]["cells"])
    res["METAMON"] = {"lp": measure(), "kept": len(sc["main"]["kept"]), "cells": len(sc["main"]["cells"])}
    restore()
    print(f"  METAMON 조립      check {-res['METAMON']['lp']['check'].mean():.5f}  "
          f"test {-res['METAMON']['lp']['test'].mean():.5f}")

    recipes = {}
    # uniform
    win = np.array([l % K for l in range(L_t)])
    s = assemble(win)
    res["uniform"] = {"lp": measure(), "kept": len(s["kept"]), "cells": len(s["cells"]), "sel": s}
    restore()
    recipes["uniform"] = s
    print(f"  uniform 배정      check {-res['uniform']['lp']['check'].mean():.5f}  "
          f"test {-res['uniform']['lp']['test'].mean():.5f}  층 {len(s['kept'])}  칸 {len(s['cells'])}  "
          f"({time.time() - t_start:.0f}s)")
    # random
    rnd = []
    for sd in range(a.rand_seeds):
        win = np.random.default_rng(100 + sd).integers(0, K, L_t)
        s = assemble(win)
        m = measure()
        restore()
        rnd.append({"lp": m, "kept": len(s["kept"]), "cells": len(s["cells"]), "sel": s})
        if sd == 0:
            recipes["random"] = s
        print(f"  random 배정 s{sd}   check {-m['check'].mean():.5f}  test {-m['test'].mean():.5f}  "
              f"층 {len(s['kept'])}  칸 {len(s['cells'])}  점유 {s['occupancy']}  ({time.time() - t_start:.0f}s)")
    res["random"] = {"lp": {nm: np.mean([r["lp"][nm] for r in rnd], 0) for nm in ("check", "test")},
                     "kept": float(np.mean([r["kept"] for r in rnd])),
                     "cells": float(np.mean([r["cells"] for r in rnd]))}

    for nm, s in recipes.items():
        rec = {"L0_sel": L0, "omega": sc["omega"], "beta": sc.get("beta"), "alphas": [a_max],
               "names": names, "chosen_by": f"ablation-{nm}", "main": s, "loo": {}}
        save_text(a, f"select_{nm}.json", json.dumps(rec, indent=1, ensure_ascii=False))

    # ---- 표
    Lt0 = -float(base["test"].mean())

    def ci(x, y):
        m, lo_, hi_ = paired_bootstrap(-np.asarray(x), -np.asarray(y))
        return [m, lo_, hi_]

    rows, md = [], ["# METAMON heterogeneous - Contribution Selection ablation (assembly, no training)\n",
                    f"- plan {a.plan}; omega {sc['omega']}; Scale = A_max {a_max:g} for every cell; Confidence = 1",
                    f"- assembly verification on sel ({a.n_sel}); reported on check ({a.n_check}) / test ({a.n_test})",
                    "- ΔL = paired bootstrap mean [95% CI] on test (negative = better)\n",
                    "| Setting | check L | test L | ΔL vs theta | ΔL vs METAMON | test avgBF | layers | cells |",
                    "|---|---|---|---|---|---|---|---|",
                    f"| theta (LoRD) | {-base['check'].mean():.5f} | {Lt0:.5f} | - | - | "
                    f"{np.exp(base['test']).mean():.4f} | | |"]
    lab = {"METAMON": "METAMON assembly", "uniform": "w/o Contribution Selection (Uniform)",
           "random": f"w/o Contribution Selection (Random, {a.rand_seeds} seeds)"}
    for k_, r_ in res.items():
        lpt = r_["lp"]["test"]
        d_t = ci(lpt, base["test"])
        d_m = ci(lpt, res["METAMON"]["lp"]["test"]) if k_ != "METAMON" else None
        row = {"setting": k_, "check_L": -float(r_["lp"]["check"].mean()), "test_L": -float(lpt.mean()),
               "test_avgBF": float(np.exp(lpt).mean()), "d_theta": d_t, "d_metamon": d_m,
               "layers": r_["kept"], "cells": r_["cells"], "test_lp": np.asarray(lpt).tolist()}
        rows.append(row)
        dm = "-" if d_m is None else f"{d_m[0]:+.5f} [{d_m[1]:+.5f}, {d_m[2]:+.5f}]"
        md.append(f"| {lab[k_]} | {row['check_L']:.5f} | {row['test_L']:.5f} | "
                  f"{d_t[0]:+.5f} [{d_t[1]:+.5f}, {d_t[2]:+.5f}] | {dm} | {row['test_avgBF']:.4f} | "
                  f"{row['layers']} | {row['cells']} |")
    md += ["", f"- uniform occupancy {recipes['uniform']['occupancy']}; random s0 occupancy {recipes['random']['occupancy']}"]
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)
    save_text(a, "report_ablation.md", md_text)
    save_text(a, "report_ablation.json", json.dumps({"rows": rows, "a_max": a_max, "omega": sc["omega"],
                                                     "theta_test_lp": base["test"].tolist()},
                                                    indent=1, ensure_ascii=False))
    print(f"저장: {a.out}/select_uniform.json  select_random.json  report_ablation.md")


if __name__ == "__main__":
    main()
