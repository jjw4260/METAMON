# -*- coding: utf-8 -*-
"""이기종 3 단계. 학습된 theta 위에서 Δz 의 기여도, 층 선택, 통합 검증, test 보고.

    python run/h3_contribution.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

  1. sel 질의(train 다음 128)의 층별 hidden state 를 캐시한다 (자기검사 포함).
  2. eq:soft_weight 용 sim_k: 각 theta_Single,k 의 질의별 eq:sim (자기 tokenizer).
  3. 칸 (l, rho) 마다, 후보 k 마다, a in A 마다
         w_theta,l,rho  <-  w_theta,l,rho + a * z~_{k,l,rho}        (eq:norm_matching)
     로 바꾸고 질의별 mean log p 를 잰다 (eq:perturbed_loss). l 층부터만 forward.
     층 하나가 끝날 때마다 Drive 에 저장한다. 끊기면 같은 명령으로 이어 간다.
     A 를 늘리면 이미 잰 a 는 저장본을 쓰고 새 a 만 잰다.
  4. eq:partial_score -> eq:layer_selection -> eq:representative_update
     -> eq:assembly_verification (omega = 1, sel).
  5. 설정마다 같은 조립을 **test** (sel, check 뒤 512) 에서 다시 잰다.
         METAMON (soft omega, A 전체)                       본 결과
         uniform omega                                     eq:soft_weight 절제
         A <= 2e-3                                         Scale 격자 절제
         LOO -k                                            {theta'_-k}
         Random direction (같은 칸, 같은 Scale, 같은 norm)  Δz 내용 대조, seed 3 개
         theta_Single,k 단독                                참고 (tokenizer 가 다름)
     theta 대비 질의별 차이의 paired bootstrap 95% CI 를 같이 낸다.
  6. report.md (표), report.json, select_soft.json (4 단계용 조립 recipe) 를 저장한다.

z 자체는 저장하지 않는다 (theta 크기라 수 GB). 칸마다 (k, a) 만 남기고, 4 단계가
Δz 파일과 eq:norm_matching 으로 같은 z 를 다시 만든다.
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
from Pipeline.metrics import EvalSet, paired_bootstrap, soft_weight, token_avg_logp
from Pipeline.modeling import setup_precision


def short(name: str) -> str:
    return name.split("/")[-1]


def copy_verified(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")


def items_for(name: str, data: str):
    tok = load_tok(name)
    cfg = Config(base=name, dataset_file=os.path.abspath(data), gold_eos=True)
    return tok, load_dataset_file(cfg.dataset_path, cfg, tok, log=lambda *_: None), cfg


def save_np(a, name: str, arr) -> None:
    tmp = os.path.join(a.local_dir, name)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    copy_verified(tmp, os.path.join(a.out, name))


def save_text(a, name: str, text: str) -> None:
    tmp = os.path.join(a.local_dir, name)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    copy_verified(tmp, os.path.join(a.out, name))


def load_np(a, name: str):
    p = os.path.join(a.out, name)
    return np.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/content/drive/MyDrive/metamon_runs/hetero/manifest.json")
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--plan", default="ot", help="h2 의 계획 (z/<plan>)")
    ap.add_argument("--out", default=None, help="기본: manifest 옆 h3/<plan>")
    ap.add_argument("--local-dir", default="/content/hetero_local/h3")
    ap.add_argument("--alphas", type=float, nargs="*", default=[5e-4, 1e-3, 2e-3, 4e-3, 8e-3],
                    help="eq:partial_score 의 A (||w_theta|| 대비 상대 크기)")
    ap.add_argument("--small-grid-max", type=float, default=2e-3,
                    help="Scale 격자 절제: 이 값 이하의 A 만 쓴 선택")
    ap.add_argument("--omega", choices=["soft", "uniform"], default="soft")
    ap.add_argument("--beta", type=float, default=0.1, help="eq:soft_weight 의 beta")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    root = os.path.dirname(a.manifest)
    a.out = a.out or os.path.join(root, "h3", a.plan)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()

    man = json.load(open(a.manifest, encoding="utf-8"))
    names = [n for n in man["locals"]
             if os.path.exists(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"))]
    if not names:
        raise SystemExit(f"{root}/z/{a.plan} 에 Δz 가 없다. h2 를 먼저 돌릴 것.")
    a.alphas = sorted(set(a.alphas))
    K, A, R = len(names), len(a.alphas), len(ROLES)
    print(f"[후보] {K}개: {[short(n) for n in names]}   A = {a.alphas}   omega {a.omega} "
          f"(beta {a.beta})")

    # ---- theta, 데이터
    tinfo = man["theta"]
    ttok, titems, _ = items_for(tinfo["name"], a.data)
    sel = titems[a.n_x:a.n_x + a.n_sel]
    t0_ = a.n_x + a.n_sel + a.n_check
    test = titems[t0_:t0_ + a.n_test]
    if len(sel) < a.n_sel or len(test) < a.n_test:
        raise SystemExit(f"질의 부족: sel {len(sel)} test {len(test)}")
    th, _ = load_trained(tinfo["name"], tinfo["delta"], device)
    L_t = th.L
    print(f"[theta] {th.describe()}   sel {len(sel)}  test {len(test)} (질의 {t0_}~)")
    with safe_open(tinfo["delta"], framework="pt", device="cpu") as f:
        rel = [float(f.get_tensor(f"{l}.{r}").float().norm() / th.weight(l, r).float().norm().cpu())
               for l in range(L_t) for r in ROLES]
    print(f"[기준] theta 자신의 ||Δw||/||w||  중앙값 {np.median(rel):.2e}  "
          f"p10 {np.percentile(rel, 10):.2e}  p90 {np.percentile(rel, 90):.2e}")

    te = TailEval(th, ttok, sel, a.bs, device)
    te.cache()
    te.check(lambda l: th.weight(l, "Down"))
    tt = TailEval(th, ttok, test, a.bs, device)
    lp0 = te.mlp()
    L0 = -float(lp0.mean())
    print(f"[theta] L(theta;1)  sel {L0:.5f}")

    # ---- sim_k (eq:sim) -> soft omega
    sims = load_np(a, "sims.npy")
    if sims is None or sims.shape != (K, a.n_sel):
        sims = np.zeros((K, a.n_sel))
        for k, n in enumerate(names):
            ltok, litems, _ = items_for(n, a.data)
            lo, _ = load_trained(n, man["locals"][n]["delta"], device)
            es = EvalSet(litems[a.n_x:a.n_x + a.n_sel], ltok, device, a.bs)
            sims[k] = np.exp(token_avg_logp(lo.model, es))
            print(f"  sim[{short(n)}] 평균 {sims[k].mean():.4f}  L {-np.log(sims[k]).mean():.5f}")
            del lo, es
            torch.cuda.empty_cache()
        save_np(a, "sims.npy", sims)
    else:
        print(f"  sim 저장본 사용 (평균 {sims.mean(1).round(4).tolist()})")

    # ---- 측정 (이전 A 의 저장본을 옮겨 담는다)
    zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                    framework="pt", device="cpu") for n in names]
    zkeys = [set(f.keys()) for f in zf]
    lp = np.full((L_t, R, K, A, a.n_sel), np.nan)
    done = np.zeros((L_t, R, A), dtype=bool)
    meta_path = os.path.join(a.out, "meta.json")
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        olp, odone = load_np(a, "lp.npy"), load_np(a, "done.npy")
        if old["names"] != names:
            raise SystemExit(f"{a.out} 의 이전 측정과 후보가 다르다: {old['names']}. --out 을 바꿀 것.")
        if olp is not None and odone is not None:
            if odone.ndim == 2:
                odone = np.repeat(odone[:, :, None], len(old["alphas"]), axis=2)
            moved = []
            for jo, al in enumerate(old["alphas"]):
                if al in a.alphas:
                    j = a.alphas.index(al)
                    lp[:, :, :, j] = olp[:, :, :, jo]
                    done[:, :, j] = odone[:, :, jo]
                    moved.append(al)
            print(f"  이전 측정에서 옮긴 a: {moved}")
    json.dump({"names": names, "alphas": a.alphas, "plan": a.plan, "n_sel": a.n_sel},
              open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    todo_total = int((~done).sum()) * K
    print(f"[측정] 칸 {L_t * R} x a {A}   남은 (칸, a) {int((~done).sum())}  "
          f"(후보 {K}개씩, l 층부터 forward)")
    t0, n_ev = time.time(), 0
    for l in range(L_t):
        if done[l].all():
            continue
        for ri, r in enumerate(ROLES):
            js = [j for j in range(A) if not done[l, ri, j]]
            if not js:
                continue
            W = th.weight(l, r)
            W0 = W.detach().clone()
            wn = float(W0.norm())
            key = f"{l}.{r}"
            for k in range(K):
                if key not in zkeys[k]:
                    continue
                dz = zf[k].get_tensor(key).to(device).float()
                f = wn / max(float(dz.norm()), 1e-30)
                for j in js:
                    with torch.no_grad():
                        W.copy_(torch.add(W0, dz, alpha=a.alphas[j] * f))
                    lp[l, ri, k, j] = te.mlp(l)
                    n_ev += 1
                del dz
            with torch.no_grad():
                W.copy_(W0)
            del W0
            done[l, ri, js] = True
        save_np(a, "lp.npy", lp)
        save_np(a, "done.npy", done)
        el = time.time() - t0
        with np.errstate(all="ignore"):
            ll = -lp[l].mean(-1)                         # (R, K, A), 균등 omega
        best = float(np.nanmax((L0 - ll) / L0)) if np.isfinite(ll).any() else float("nan")
        print(f"  층 {l + 1}/{L_t}  {el:.0f}s  남은 {el / max(n_ev, 1) * (todo_total - n_ev):.0f}s  "
              f"이 층 최대 감소율 {best:.2e}", flush=True)
    torch.cuda.empty_cache()

    # ---- 조립 도구
    base_W = {(l, r): th.weight(l, r).detach().cpu().clone()
              for l in range(L_t) for r in ROLES}
    base_norm = {kk: float(v.norm()) for kk, v in base_W.items()}

    def restore():
        with torch.no_grad():
            for (l, r), w in base_W.items():
                th.weight(l, r).copy_(w.to(device))

    def make_cells(win, ps, ai, als, keep):
        """eq:representative_update. PartialScore > 0 인 칸만."""
        return [{"l": int(l), "role": r, "k": int(win[l]), "a": float(als[int(ai[l, ri, win[l]])])}
                for l in keep for ri, r in enumerate(ROLES) if ps[l, ri, win[l]] > 0]

    def apply(cells, seed=None):
        """z~ = Δz * ||w|| / ||Δz|| (eq:norm_matching). seed 가 있으면 Δz 대신 같은 모양의
        가우시안 방향 (norm 은 같은 식으로 맞춘다)."""
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

    def verify(win, top, ps, ai, als, tag):
        """eq:assembly_verification, omega = 1, sel."""
        cells = make_cells(win, ps, ai, als, range(L_t))
        apply(cells)
        L_asm = -float(te.mlp().mean())
        print(f"  [{tag}] 통합 검증 L(theta+z;1) {L_asm:.5f} vs {L0:.5f}  "
              f"{'통과' if L_asm < L0 else '실패 - greedy 복구'}")
        keep, cur = list(range(L_t)), L_asm
        if L_asm >= L0:
            keep, cur = [], L0
            for l in sorted(range(L_t), key=lambda x: -top[x]):
                apply(make_cells(win, ps, ai, als, keep + [l]))
                v = -float(te.mlp().mean())
                if v < cur:
                    keep, cur = keep + [l], v
            print(f"  [{tag}] 유지 층 {len(keep)}/{L_t}   L {cur:.5f}")
            cells = make_cells(win, ps, ai, als, keep)
        restore()
        return sorted(keep), cells, L_asm, cur

    def selection(lp_, als, omegas, ks, tag):
        ps, ai, base = partial_scores(lp_, lp0, omegas)
        win, conf, top, LS = select_layers(ps, ks)
        keep, cells, L_asm, L_fin = verify(win, top, ps, ai, als, tag)
        return {"win": [names[int(k)] for k in win], "confidence": conf.tolist(),
                "layer_score": np.nan_to_num(LS).tolist(), "kept": keep, "cells": cells,
                "L_asm_sel": L_asm, "L_sel": L_fin, "base_omega": base.tolist(),
                "_ps": ps, "_ai": ai, "_win": win, "_conf": conf}

    def describe(s, als, tag):
        ps, ai, win, conf = s["_ps"], s["_ai"], s["_win"], s["_conf"]
        used = [k for k in range(K) if (win == k).any()]
        zero = int((ps.max(-1) <= 0).sum())
        cnt = {short(names[k]): [int(((ai[:, :, k] == j) & (ps[:, :, k] > 0)).sum())
                                 for j in range(len(als))] for k in range(K)}
        print(f"\n[{tag}]  PartialScore>0 칸 {L_t * R - zero}/{L_t * R}   "
              f"최대 PartialScore 중앙값 {np.median(ps.max(-1)):.2e}   Confidence 중앙값 "
              f"{np.median(conf):.2e}")
        print(f"  층 점유 {dict((short(names[k]), int((win == k).sum())) for k in range(K))}")
        print(f"  고른 Scale 분포 {als}: {cnt}")
        return used

    # ---- 선택들 (모두 같은 측정 lp 에서)
    om = {"uniform": None, "soft": soft_weight(sims, a.beta)}
    small = [j for j, al in enumerate(a.alphas) if al <= a.small_grid_max + 1e-15]
    small_al = [a.alphas[j] for j in small]
    allk = list(range(K))
    confs = {}
    confs["METAMON"] = selection(lp, a.alphas, om[a.omega], allk, "METAMON")
    describe(confs["METAMON"], a.alphas, f"METAMON omega={a.omega} A={a.alphas}")
    other = "uniform" if a.omega == "soft" else "soft"
    confs[f"omega={other}"] = selection(lp, a.alphas, om[other], allk, f"omega={other}")
    describe(confs[f"omega={other}"], a.alphas, f"omega={other}")
    if len(small) < A:
        confs[f"A<={a.small_grid_max:g}"] = selection(lp[:, :, :, small], small_al, om[a.omega],
                                                     allk, f"A<={a.small_grid_max:g}")
        describe(confs[f"A<={a.small_grid_max:g}"], small_al, f"A={small_al}")
    for kx in range(K):
        confs[f"LOO -{short(names[kx])}"] = selection(
            lp, a.alphas, om[a.omega], [k for k in range(K) if k != kx], f"-{short(names[kx])}")

    # ---- test
    print("\n[test] 조립을 test 에서 다시 잰다")
    tlp = {"theta": tt.mlp()}
    for nm, s in confs.items():
        apply(s["cells"])
        tlp[nm] = tt.mlp()
        restore()
        print(f"  {nm:28s} test L {-tlp[nm].mean():.5f}")
    rs_sel, rs_test = [], []
    for sd in range(a.rand_seeds):
        apply(confs["METAMON"]["cells"], seed=1000 + sd)
        rs_sel.append(te.mlp())
        rs_test.append(tt.mlp())
        restore()
        print(f"  Random direction seed {sd}        sel L {-rs_sel[-1].mean():.5f}  "
              f"test L {-rs_test[-1].mean():.5f}")
    tlp["Random direction"] = np.mean(rs_test, 0)
    single = {}
    for k, n in enumerate(names):
        ltok, litems, _ = items_for(n, a.data)
        lo, _ = load_trained(n, man["locals"][n]["delta"], device)
        es = EvalSet(litems[t0_:t0_ + a.n_test], ltok, device, a.bs)
        single[n] = token_avg_logp(lo.model, es)
        del lo, es
        torch.cuda.empty_cache()
        print(f"  theta_Single {short(n):18s} test L {-single[n].mean():.5f}  (자기 tokenizer)")

    # ---- 표
    def row(label, lp_sel_L, lpt, kept=None, ncell=None, note=""):
        Lt = -float(lpt.mean())
        if label == "theta (LoRD)":
            d = "-"
        else:
            m, lo_, hi_ = paired_bootstrap(-lpt, -tlp["theta"])
            d = f"{m:+.5f} [{lo_:+.5f}, {hi_:+.5f}]"
        return {"label": label, "sel_L": lp_sel_L, "test_L": Lt,
                "test_rel": (Lt - (-float(tlp['theta'].mean()))) / -float(tlp['theta'].mean()),
                "dL_ci": d, "test_avgBF": float(np.exp(lpt).mean()),
                "kept": kept, "cells": ncell, "note": note}

    rows = [row("theta (LoRD)", L0, tlp["theta"])]
    labels = {"METAMON": f"METAMON assembly (omega={a.omega})",
              f"omega={other}": f"  w/ omega={other}",
              f"A<={a.small_grid_max:g}": f"  w/ A={small_al}"}
    for nm, s in confs.items():
        rows.append(row(labels.get(nm, "  " + nm), s["L_sel"], tlp[nm], len(s["kept"]),
                        len(s["cells"])))
    rows.append(row("  Random direction (same cells/scales)",
                    -float(np.mean(rs_sel, 0).mean()), tlp["Random direction"],
                    len(confs["METAMON"]["kept"]), len(confs["METAMON"]["cells"]),
                    f"{a.rand_seeds} seeds, sel L {[round(-x.mean(), 5) for x in rs_sel]}, "
                    f"test L {[round(-x.mean(), 5) for x in rs_test]}"))
    for n in names:
        rows.append({"label": f"theta_Single {short(n)}", "sel_L": float(-np.log(sims[names.index(n)]).mean()),
                     "test_L": -float(single[n].mean()), "test_rel": None, "dL_ci": "-",
                     "test_avgBF": float(np.exp(single[n]).mean()), "kept": None, "cells": None,
                     "note": "own tokenizer; not directly comparable"})

    md = [f"# METAMON heterogeneous - assembly (step 3)\n",
          f"- Target-substitute theta: {tinfo['name']} (+LoRD Δw)",
          f"- Locals: {', '.join(short(n) for n in names)}   plan {a.plan}, R^2 >= 0.3",
          f"- A = {a.alphas}, omega = {a.omega} (beta {a.beta}), sel {a.n_sel} / test {a.n_test} "
          f"(items {t0_}..{t0_ + a.n_test - 1})",
          f"- L = mean over queries of -mean token log p (eq:weighted_loss, omega=1); "
          f"ΔL = paired bootstrap mean [95% CI] vs theta on test (negative = better)\n",
          "| Setting | sel L | test L | test ΔL vs theta [95% CI] | rel. | test avgBF | layers | cells |",
          "|---|---|---|---|---|---|---|---|"]
    def opt(v, fmt="{}"):
        return "" if v is None else fmt.format(v)

    for r_ in rows:
        md.append("| " + " | ".join([
            r_["label"].strip(), f"{r_['sel_L']:.5f}", f"{r_['test_L']:.5f}", r_["dL_ci"],
            opt(r_["test_rel"], "{:+.2%}"), f"{r_['test_avgBF']:.4f}",
            opt(r_["kept"]), opt(r_["cells"])]) + " |")
    md.append("")
    for r_ in rows:
        if r_["note"]:
            md.append(f"- {r_['label'].strip()}: {r_['note']}")
    m = confs["METAMON"]
    occ = {short(n): sum(1 for w in m["win"] if w == n) for n in names}
    md.append(f"- METAMON layer occupancy {occ}; median Confidence {np.median(m['confidence']):.2e}")
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)

    rec = {"L0_sel": L0, "omega": a.omega, "beta": a.beta, "alphas": a.alphas, "names": names,
           "theta_rel_dw_median": float(np.median(rel)),
           "main": {k: v for k, v in m.items() if not k.startswith("_")},
           "loo": {n: {k: v for k, v in confs[f"LOO -{short(n)}"].items() if not k.startswith("_")}
                   for n in names}}
    save_text(a, f"select_{a.omega}.json", json.dumps(rec, indent=1, ensure_ascii=False))
    save_text(a, "report.json", json.dumps(
        {"rows": rows, "test_lp": {k: v.tolist() for k, v in tlp.items()},
         "single_test_lp": {k: v.tolist() for k, v in single.items()},
         "configs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in confs.items()}}, indent=1, ensure_ascii=False))
    save_text(a, "report.md", md_text)
    print(f"저장: {a.out}/report.md  report.json  select_{a.omega}.json")


if __name__ == "__main__":
    main()
