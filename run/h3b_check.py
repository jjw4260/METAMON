# -*- coding: utf-8 -*-
"""이기종 3 단계 (b). omega 와 A 를 check 에서 고른다. 새 섭동 측정은 하지 않는다.

    python run/h3b_check.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

h3 는 PartialScore / Scale / 층 선택 / 통합 검증을 모두 sel 128 로 했고, omega 와 A
도 sel 결과를 보고 정한 셈이었다. 그러면 sel 에 맞춘 몫이 test 에서 빠진다.
여기서는 그 두 hyperparameter 를 선택에 쓰지 않은 **check** (sel 뒤 256) 로 정한다.

  격자   omega in {uniform, soft(beta)}  x  A = { a in A_all : a <= A_max },
         A_max in A_all
  설정마다  h3 의 lp (sel 측정) 로 eq:partial_score -> eq:layer_selection ->
           eq:representative_update -> eq:assembly_verification (sel, omega=1)
           -> check L, test L
  고르기   check L 이 가장 낮은 설정 (test 는 보지 않는다)
  고른 설정  LOO -k, Random direction (seed 3) 을 test 에서 잰다

저장 (h3/<plan>/): report_check.md, report_check.json,
                  select_check.json  (4 단계용 조립 recipe, select_soft.json 과 같은 형식)
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
from Pipeline.hetero.contrib import TailEval, partial_scores, select_layers
from Pipeline.hetero.load import load_tok, load_trained
from Pipeline.hetero.roles import ROLES
from Pipeline.metrics import paired_bootstrap, soft_weight
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
    ap.add_argument("--manifest", default="/content/drive/MyDrive/metamon_runs/hetero/manifest.json")
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--plan", default="ot")
    ap.add_argument("--out", default=None, help="기본: manifest 옆 h3/<plan> (h3 결과가 있는 곳)")
    ap.add_argument("--local-dir", default="/content/hetero_local/h3b")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    root = os.path.dirname(a.manifest)
    a.out = a.out or os.path.join(root, "h3", a.plan)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()

    meta = json.load(open(os.path.join(a.out, "meta.json"), encoding="utf-8"))
    names, alphas = meta["names"], meta["alphas"]
    lp = np.load(os.path.join(a.out, "lp.npy"))
    done = np.load(os.path.join(a.out, "done.npy"))
    sims = np.load(os.path.join(a.out, "sims.npy"))
    if not done.all():
        raise SystemExit(f"h3 측정이 끝나지 않았다 ({int(done.sum())}/{done.size}). h3 를 먼저 끝낼 것.")
    K, R = len(names), len(ROLES)
    print(f"[h3 측정] 후보 {[short(n) for n in names]}  A {alphas}  lp {lp.shape}")

    man = json.load(open(a.manifest, encoding="utf-8"))
    tinfo = man["theta"]
    ttok = load_tok(tinfo["name"])
    cfg = Config(base=tinfo["name"], dataset_file=os.path.abspath(a.data), gold_eos=True)
    titems = load_dataset_file(cfg.dataset_path, cfg, ttok, log=lambda *_: None)
    s0, c0 = a.n_x, a.n_x + a.n_sel
    t0_ = c0 + a.n_check
    sel, chk, test = titems[s0:c0], titems[c0:t0_], titems[t0_:t0_ + a.n_test]
    if len(chk) < a.n_check or len(test) < a.n_test:
        raise SystemExit(f"질의 부족: check {len(chk)} test {len(test)}")
    th, _ = load_trained(tinfo["name"], tinfo["delta"], device)
    L_t = th.L
    if lp.shape[0] != L_t:
        raise SystemExit(f"lp 층 수 {lp.shape[0]} != theta {L_t}")
    print(f"[분할] sel {s0}..{c0 - 1}  check {c0}..{t0_ - 1}  test {t0_}..{t0_ + a.n_test - 1}")

    ev = {nm: TailEval(th, ttok, it, a.bs, device) for nm, it in
          (("sel", sel), ("check", chk), ("test", test))}
    base = {nm: e.mlp() for nm, e in ev.items()}
    L0 = -float(base["sel"].mean())
    print("[theta] " + "  ".join(f"{nm} L {-v.mean():.5f}" for nm, v in base.items()))

    zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                    framework="pt", device="cpu") for n in names]
    base_W = {(l, r): th.weight(l, r).detach().cpu().clone() for l in range(L_t) for r in ROLES}
    base_norm = {kk: float(v.norm()) for kk, v in base_W.items()}

    def restore():
        with torch.no_grad():
            for (l, r), w in base_W.items():
                th.weight(l, r).copy_(w.to(device))

    def make_cells(win, ps, ai, als, keep):
        return [{"l": int(l), "role": r, "k": int(win[l]), "a": float(als[int(ai[l, ri, win[l]])])}
                for l in keep for ri, r in enumerate(ROLES) if ps[l, ri, win[l]] > 0]

    def apply(cells, seed=None):
        restore()
        g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            for c in cells:
                W = th.weight(c["l"], c["role"])
                if g is None:
                    d = zf[c["k"]].get_tensor(f"{c['l']}.{c['role']}").to(device).float()
                else:
                    d = torch.randn(W.shape, generator=g, device=device, dtype=W.dtype)
                W.add_(d, alpha=c["a"] * base_norm[(c["l"], c["role"])]
                       / max(float(d.norm()), 1e-30))

    def assemble(lp_, als, omegas, ks):
        """sel 로 선택 + 통합 검증 (h3 와 같은 절차)."""
        ps, ai, _ = partial_scores(lp_, base["sel"], omegas)
        win, conf, top, LS = select_layers(ps, ks)
        cells = make_cells(win, ps, ai, als, range(L_t))
        apply(cells)
        L_asm = -float(ev["sel"].mlp().mean())
        keep, cur = list(range(L_t)), L_asm
        if L_asm >= L0:
            keep, cur = [], L0
            for l in sorted(range(L_t), key=lambda x: -top[x]):
                apply(make_cells(win, ps, ai, als, keep + [l]))
                v = -float(ev["sel"].mlp().mean())
                if v < cur:
                    keep, cur = keep + [l], v
            cells = make_cells(win, ps, ai, als, keep)
            apply(cells)
        out = {"win": [names[int(k)] for k in win], "confidence": conf.tolist(),
               "layer_score": np.nan_to_num(LS).tolist(), "kept": sorted(keep), "cells": cells,
               "L_asm_sel": L_asm, "L_sel": cur,
               "occupancy": {short(names[k]): int((win == k).sum()) for k in ks}}
        return out

    def measure(nms):
        return {nm: ev[nm].mlp() for nm in nms}

    def dl(lpx, nm):
        m, lo_, hi_ = paired_bootstrap(-lpx, -base[nm])
        return m, lo_, hi_

    # ---- 격자
    om = {"uniform": None, f"soft(beta={a.beta:g})": soft_weight(sims, a.beta)}
    grid = []
    t0 = time.time()
    for oname, omg in om.items():
        for amax in alphas:
            js = [j for j, x in enumerate(alphas) if x <= amax + 1e-15]
            als = [alphas[j] for j in js]
            s = assemble(lp[:, :, :, js], als, omg, list(range(K)))
            m = measure(["check", "test"])
            restore()
            row = {"omega": oname, "A_max": amax, "A": als, "sel_L": s["L_sel"],
                   "check_L": -float(m["check"].mean()), "test_L": -float(m["test"].mean()),
                   "kept": len(s["kept"]), "cells": len(s["cells"]),
                   "test_lp": m["test"].tolist(), "check_lp": m["check"].tolist(), "sel": s}
            grid.append(row)
            print(f"  omega {oname:16s} A<= {amax:g}  sel {row['sel_L']:.5f}  check {row['check_L']:.5f}  "
                  f"test {row['test_L']:.5f}  층 {row['kept']}  칸 {row['cells']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    best = min(grid, key=lambda r: r["check_L"])
    bo, bA = best["omega"], best["A"]
    print(f"\n[선택] check L 최소: omega {bo}  A {bA}  check {best['check_L']:.5f}")

    # ---- 고른 설정: LOO, Random
    jsb = [alphas.index(x) for x in bA]
    loo = {}
    for kx in range(K):
        s = assemble(lp[:, :, :, jsb], bA, om[bo], [k for k in range(K) if k != kx])
        m = measure(["check", "test"])
        restore()
        loo[names[kx]] = {"sel": s, "check_L": -float(m["check"].mean()),
                          "test_L": -float(m["test"].mean()), "test_lp": m["test"].tolist()}
        print(f"  LOO -{short(names[kx]):24s} sel {s['L_sel']:.5f}  check {loo[names[kx]]['check_L']:.5f}  "
              f"test {loo[names[kx]]['test_L']:.5f}")
    rnd = []
    for sd in range(a.rand_seeds):
        apply(best["sel"]["cells"], seed=1000 + sd)
        rnd.append(measure(["sel", "check", "test"]))
        restore()
        print(f"  Random seed {sd}  sel {-rnd[-1]['sel'].mean():.5f}  check {-rnd[-1]['check'].mean():.5f}  "
              f"test {-rnd[-1]['test'].mean():.5f}")
    rnd_test = np.mean([r["test"] for r in rnd], 0)

    # ---- 표
    def ci(lpx):
        m, lo_, hi_ = dl(np.asarray(lpx), "test")
        return f"{m:+.5f} [{lo_:+.5f}, {hi_:+.5f}]"

    Lt0 = -float(base["test"].mean())

    def rel(x):
        return f"{(x - Lt0) / Lt0:+.2%}"

    md = ["# METAMON heterogeneous - assembly with omega / A chosen on check\n",
          f"- theta: {tinfo['name']} (+LoRD Δw); Locals: {', '.join(short(n) for n in names)}; plan {a.plan}",
          f"- PartialScore / Scale / layer selection / assembly verification: sel ({a.n_sel}); "
          f"omega, A: chosen by check L ({a.n_check}); reported: test ({a.n_test}, items {t0_}..{t0_ + a.n_test - 1})",
          "- L = mean over queries of -mean token log p (omega=1); ΔL = paired bootstrap mean [95% CI] vs theta on test\n",
          "## Hyperparameter grid (E5)\n",
          "| omega | A | sel L | check L | test L | test ΔL vs theta [95% CI] | rel. | layers | cells |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| theta (LoRD) | - | {L0:.5f} | {-base['check'].mean():.5f} | {Lt0:.5f} | - | - | | |"]
    for r_ in grid:
        mark = " **(chosen)**" if r_ is best else ""
        md.append(f"| {r_['omega']}{mark} | ≤ {r_['A_max']:g} | {r_['sel_L']:.5f} | {r_['check_L']:.5f} | "
                  f"{r_['test_L']:.5f} | {ci(r_['test_lp'])} | {rel(r_['test_L'])} | {r_['kept']} | {r_['cells']} |")
    md += ["", f"## Chosen setting (omega {bo}, A {bA}) on test\n",
           "| Setting | test L | test ΔL vs theta [95% CI] | rel. | test avgBF | cells |",
           "|---|---|---|---|---|---|",
           f"| theta (LoRD) | {Lt0:.5f} | - | - | {np.exp(base['test']).mean():.4f} | |",
           f"| **METAMON assembly** | {best['test_L']:.5f} | {ci(best['test_lp'])} | {rel(best['test_L'])} | "
           f"{np.exp(np.asarray(best['test_lp'])).mean():.4f} | {best['cells']} |"]
    for n, v in loo.items():
        md.append(f"| LOO -{short(n)} | {v['test_L']:.5f} | {ci(v['test_lp'])} | {rel(v['test_L'])} | "
                  f"{np.exp(np.asarray(v['test_lp'])).mean():.4f} | {len(v['sel']['cells'])} |")
    md.append(f"| Random direction (same cells/scales, {a.rand_seeds} seeds) | {-rnd_test.mean():.5f} | "
              f"{ci(rnd_test)} | {rel(-rnd_test.mean())} | {np.exp(rnd_test).mean():.4f} | {best['cells']} |")
    m_r, lo_r, hi_r = paired_bootstrap(-np.asarray(best["test_lp"]), -rnd_test)
    md += ["", f"- METAMON − Random direction (test): {m_r:+.5f} [{lo_r:+.5f}, {hi_r:+.5f}]",
           f"- Random seeds test L: {[round(float(-r['test'].mean()), 5) for r in rnd]}",
           f"- layer occupancy {best['sel']['occupancy']}; median Confidence "
           f"{np.median(best['sel']['confidence']):.2e}"]
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)

    rec = {"L0_sel": L0, "omega": bo, "beta": a.beta, "alphas": bA, "names": names,
           "chosen_by": "check", "main": best["sel"],
           "loo": {n: v["sel"] for n, v in loo.items()}}
    save_text(a, "select_check.json", json.dumps(rec, indent=1, ensure_ascii=False))
    save_text(a, "report_check.json", json.dumps(
        {"grid": [{k: v for k, v in r_.items() if k != "sel"} for r_ in grid],
         "chosen": {"omega": bo, "A": bA}, "base_test_lp": base["test"].tolist(),
         "loo": {n: {k: v for k, v in x.items() if k != "sel"} for n, x in loo.items()},
         "random_test_lp": [r["test"].tolist() for r in rnd]}, indent=1, ensure_ascii=False))
    save_text(a, "report_check.md", md_text)
    print(f"저장: {a.out}/report_check.md  report_check.json  select_check.json")


if __name__ == "__main__":
    main()
