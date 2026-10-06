# -*- coding: utf-8 -*-
"""이기종 3 단계. 학습된 theta 위에서 Δz 의 기여도, 층 선택, 통합 검증.

    python run/h3_contribution.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

  1. sel 질의(train 다음 128)의 층별 hidden state 를 캐시한다 (자기검사 포함).
  2. eq:soft_weight 용 sim_k: 각 theta_Single,k 의 질의별 eq:sim (자기 tokenizer).
  3. 칸 (l, rho) 마다, 후보 k 마다, a in A 마다
         w_theta,l,rho  <-  w_theta,l,rho + a * z~_{k,l,rho}        (eq:norm_matching)
     로 바꾸고 질의별 mean log p 를 잰다 (eq:perturbed_loss). l 층부터만 forward.
     층 하나가 끝날 때마다 Drive 에 저장한다. 끊기면 같은 명령으로 이어 간다.
  4. eq:partial_score -> eq:layer_selection -> eq:representative_update
     -> eq:assembly_verification (omega = 1).  --omega 로 고른 omega 로 선택한다.
     균등 omega 결과도 함께 출력한다 (같은 측정에서 다시 계산할 뿐이다).
  5. LOO: 후보 하나씩 뺀 선택과 통합 검증 ({theta'_-k} 용).

z 자체는 저장하지 않는다 (theta 크기라 수 GB). 칸마다 (k, a) 만 남기고, 4 단계가
Δz 파일과 eq:norm_matching 으로 같은 z 를 다시 만든다.

A 의 기준: 출력 첫머리에 theta 자신의 Δw (lord_all) 상대 크기 ||Δw||/||w|| 분포를
찍는다. A 는 그와 같은 자릿수여야 Δz 가 theta 의 학습량과 비교 가능한 섭동이 된다.
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
from Pipeline.metrics import EvalSet, soft_weight, token_avg_logp
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
    ap.add_argument("--alphas", type=float, nargs="*", default=[5e-4, 1e-3, 2e-3],
                    help="eq:partial_score 의 A (||w_theta|| 대비 상대 크기)")
    ap.add_argument("--omega", choices=["soft", "uniform"], default="soft")
    ap.add_argument("--beta", type=float, default=0.1, help="eq:soft_weight 의 beta")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
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
    K, A = len(names), len(a.alphas)
    print(f"[후보] {K}개: {[short(n) for n in names]}   A = {a.alphas}   omega {a.omega} "
          f"(beta {a.beta})")
    meta_path = os.path.join(a.out, "meta.json")
    meta = {"names": names, "alphas": a.alphas, "plan": a.plan, "n_sel": a.n_sel}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        if old["names"] != names or old["alphas"] != a.alphas:
            raise SystemExit(f"{a.out} 의 이전 측정과 후보/A 가 다르다: {old}. --out 을 바꿀 것.")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---- theta
    tinfo = man["theta"]
    ttok, titems, _ = items_for(tinfo["name"], a.data)
    sel = titems[a.n_x:a.n_x + a.n_sel]
    th, _ = load_trained(tinfo["name"], tinfo["delta"], device)
    print(f"[theta] {th.describe()}")
    with safe_open(tinfo["delta"], framework="pt", device="cpu") as f:
        rel = [float(f.get_tensor(f"{l}.{r}").float().norm() / th.weight(l, r).float().norm().cpu())
               for l in range(th.L) for r in ROLES]
    print(f"[기준] theta 자신의 ||Δw||/||w||  중앙값 {np.median(rel):.2e}  "
          f"p10 {np.percentile(rel, 10):.2e}  p90 {np.percentile(rel, 90):.2e}")

    te = TailEval(th, ttok, sel, a.bs, device)
    te.cache()
    te.check(lambda l: th.weight(l, "Down"))
    lp0 = te.mlp()
    L0 = -float(lp0.mean())
    print(f"[theta] L(theta;1) on sel = {L0:.5f}")

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

    # ---- 측정
    zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                    framework="pt", device="cpu") for n in names]
    zkeys = [set(f.keys()) for f in zf]
    lp = load_np(a, "lp.npy")
    done = load_np(a, "done.npy")
    if lp is None or done is None or lp.shape != (th.L, len(ROLES), K, A, a.n_sel):
        lp = np.full((th.L, len(ROLES), K, A, a.n_sel), np.nan)
        done = np.zeros((th.L, len(ROLES)), dtype=bool)
    print(f"[측정] 칸 {th.L * len(ROLES)}  이미 {int(done.sum())}  "
          f"칸당 {K}x{A} 번 (l 층부터 forward)")
    t0, n0 = time.time(), int(done.sum())
    for l in range(th.L):
        if done[l].all():
            continue
        for ri, r in enumerate(ROLES):
            if done[l, ri]:
                continue
            W = th.weight(l, r)
            W0 = W.detach().clone()
            wn = float(W0.norm())
            for k in range(K):
                key = f"{l}.{r}"
                if key not in zkeys[k]:
                    continue
                dz = zf[k].get_tensor(key).to(device).float()
                f = wn / max(float(dz.norm()), 1e-30)
                for j, al in enumerate(a.alphas):
                    with torch.no_grad():
                        W.copy_(torch.add(W0, dz, alpha=al * f))
                    lp[l, ri, k, j] = te.mlp(l)
                del dz
            with torch.no_grad():
                W.copy_(W0)
            del W0
            done[l, ri] = True
        save_np(a, "lp.npy", lp)
        save_np(a, "done.npy", done)
        c = int(done.sum())
        el = time.time() - t0
        with np.errstate(all="ignore"):
            ll = -lp[l].mean(-1)                         # (R, K, A), 균등 omega
        best = float(np.nanmax((L0 - ll) / L0)) if np.isfinite(ll).any() else float("nan")
        print(f"  층 {l + 1}/{th.L}  {el:.0f}s  남은 {el / max(c - n0, 1) * (done.size - c):.0f}s  "
              f"이 층 최대 감소율 {best:.2e}", flush=True)
    torch.cuda.empty_cache()

    # ---- 선택
    base_W = {(l, r): th.weight(l, r).detach().cpu().clone()
              for l in range(th.L) for r in ROLES}

    def restore():
        with torch.no_grad():
            for (l, r), w in base_W.items():
                th.weight(l, r).copy_(w.to(device))

    def apply(win, ps, ai, keep):
        restore()
        with torch.no_grad():
            for l in keep:
                k = int(win[l])
                for ri, r in enumerate(ROLES):
                    if ps[l, ri, k] <= 0:
                        continue
                    W = th.weight(l, r)
                    dz = zf[k].get_tensor(f"{l}.{r}").to(device).float()
                    f = float(base_W[(l, r)].norm()) / max(float(dz.norm()), 1e-30)
                    W.add_(dz, alpha=a.alphas[int(ai[l, ri, k])] * f)

    def verify(win, top, ps, ai, tag):
        """eq:assembly_verification, omega = 1."""
        apply(win, ps, ai, range(th.L))
        L_asm = -float(te.mlp().mean())
        print(f"  [{tag}] 통합 검증 L(theta+z;1) {L_asm:.5f} vs L(theta;1) {L0:.5f}  "
              f"{'통과' if L_asm < L0 else '실패 - greedy 복구'}")
        keep, cur = list(range(th.L)), L_asm
        if L_asm >= L0:
            keep, cur = [], L0
            for l in sorted(range(th.L), key=lambda x: -top[x]):
                apply(win, ps, ai, keep + [l])
                v = -float(te.mlp().mean())
                if v < cur:
                    keep, cur = keep + [l], v
            print(f"  [{tag}] 유지 층 {len(keep)}/{th.L}   L {cur:.5f}")
        restore()
        return sorted(keep), L_asm, cur

    om = {"uniform": None, "soft": soft_weight(sims, a.beta)}
    res = {"L0": L0, "omega": a.omega, "beta": a.beta, "alphas": a.alphas,
           "names": names, "theta_rel_dw_median": float(np.median(rel))}
    for mode in (["uniform", "soft"] if a.omega == "soft" else ["soft", "uniform"]):
        ps, ai, base = partial_scores(lp, lp0, om[mode])
        win, conf, top, LS = select_layers(ps, range(K))
        occ = [int((win == k).sum()) for k in range(K)]
        zero = int((ps.max(-1) <= 0).sum())
        cnt = np.array([[(ai[:, :, k] == j)[ps[:, :, k] > 0].sum() for j in range(A)]
                        for k in range(K)])
        print(f"\n[선택 omega={mode}]  base L(theta;omega_k) {np.round(base, 5).tolist()}")
        print(f"  PartialScore>0 칸 {ps.size // K - zero}/{ps.size // K}   "
              f"최대 PartialScore 중앙값 {np.median(ps.max(-1)):.2e}")
        print(f"  층 점유 {dict(zip([short(n) for n in names], occ))}   "
              f"Confidence 중앙값 {np.median(conf):.2e}")
        print(f"  고른 Scale 분포 (후보 x A) {cnt.tolist()}")
        print("  역할별 최대 PartialScore 중앙값  "
              + "  ".join(f"{r} {np.median(ps[:, ri].max(-1)):.1e}" for ri, r in enumerate(ROLES)))
        if mode != a.omega:
            continue
        keep, L_asm, L_fin = verify(win, top, ps, ai, "전체")
        cells = [{"l": l, "role": r, "k": int(win[l]), "a": a.alphas[int(ai[l, ri, win[l]])]}
                 for l in keep for ri, r in enumerate(ROLES) if ps[l, ri, win[l]] > 0]
        res["main"] = {"win": [names[int(k)] for k in win], "confidence": conf.tolist(),
                       "layer_score": LS.tolist(), "kept": keep, "L_asm": L_asm,
                       "L_final": L_fin, "cells": cells,
                       "partial_score": ps.tolist(), "scale_idx": ai.tolist()}
        res["loo"] = {}
        for kx in range(K):
            ks = [k for k in range(K) if k != kx]
            w2, c2, t2, LS2 = select_layers(ps, ks)
            keep2, La2, Lf2 = verify(w2, t2, ps, ai, f"-{short(names[kx])}")
            res["loo"][names[kx]] = {
                "win": [names[int(k)] for k in w2], "confidence": c2.tolist(), "kept": keep2,
                "L_asm": La2, "L_final": Lf2,
                "cells": [{"l": l, "role": r, "k": int(w2[l]),
                           "a": a.alphas[int(ai[l, ri, w2[l]])]}
                          for l in keep2 for ri, r in enumerate(ROLES) if ps[l, ri, w2[l]] > 0]}

    tmp = os.path.join(a.local_dir, f"select_{a.omega}.json")
    json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    copy_verified(tmp, os.path.join(a.out, f"select_{a.omega}.json"))
    m = res["main"]
    print(f"\n[요약] L(theta;1) {L0:.5f} -> 조립 {m['L_final']:.5f}  "
          f"(상대 {(L0 - m['L_final']) / L0:+.2%})  유지 층 {len(m['kept'])}/{th.L}  "
          f"칸 {len(m['cells'])}")
    for n, v in res["loo"].items():
        print(f"  LOO -{short(n):24s} L {v['L_final']:.5f}  유지 층 {len(v['kept'])}")
    print(f"저장: {os.path.join(a.out, f'select_{a.omega}.json')}")


if __name__ == "__main__":
    main()
