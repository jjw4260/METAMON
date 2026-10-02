# -*- coding: utf-8 -*-
"""이기종 0 단계. Mapping R^2 사전 점검. 학습은 하지 않는다.

    python run/h0_r2probe.py --data /content/runs/l32/target/dataset.json \
        --out /content/runs/hetero

LoRD 학습에 GPU 를 쓰기 전에, 학습 전(사전학습) 모델끼리 eq:input_map / eq:output_map
이 성립하는지 본다. 대부분의 쌍이 R^2 기준을 못 넘으면 eq:weight_projection 이 거의
아무것도 전달하지 못한다. 그건 학습 전에 알아야 한다.

층 대응은 아직 sensitivity 가 없으므로 Layer Ratio Mapping (같은 Pi_k 의 깊이 단조
수송)으로 잡는다. 질량이 --mass-min 이상인 (l, i) 쌍만 잰다.

    적합   train 질의 앞 --n-fit 개
    판정   sel 질의 (train 다음 --n-held 개). 학습에 쓰지 않은 질의다.

시작하면 먼저 자기검사를 한다. theta 를 theta 자신에 맞추면 R^2 가 1 이어야 하고,
모든 모델에서 역할별로 출력 == 입력 @ W^T 여야 한다. 하나라도 어긋나면 멈춘다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Pipeline.hetero import ot as OT
from Pipeline.hetero import spans as SP
from Pipeline.hetero.ridge import Held, Reg, Var, solve_group
from Pipeline.hetero.roles import IN_KIND, KINDS, ROLES, Arch, StopForward, check_roles

DEFAULT_LOCALS = ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it",
                  "microsoft/Phi-3-mini-4k-instruct",
                  "HuggingFaceTB/SmolLM2-1.7B-Instruct"]


# ------------------------------------------------------------------ 모델
def load(name: str, device: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    attn = "eager" if "gemma" in name.lower() else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
    model.config.use_cache = False
    return tok, Arch(model)


def pad(ids: List[List[int]], pad_id: int, device: str):
    T = max(len(x) for x in ids)
    inp = torch.full((len(ids), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for b, x in enumerate(ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    return inp.to(device), att.to(device)


@torch.no_grad()
def run(arch: Arch, tok, ids, ins, outs, stop, device):
    inp, att = pad(ids, tok.pad_token_id, device)
    with arch.capture(ins, outs, stop_after=stop) as st:
        try:
            arch.model(input_ids=inp, attention_mask=att)
        except StopForward:
            pass
    return dict(st)


def pool_index(wids: List[List[int]], keep: List[List[int]], device):
    """토큰 -> (keep 순서의) 단어 행 번호. 한 묶음·한 모델에 한 번 만든다."""
    rows, b_idx, t_idx, base = [], [], [], 0
    for b, (w, k) in enumerate(zip(wids, keep)):
        rank = {x: base + r for r, x in enumerate(k)}
        for t, x in enumerate(w):
            if x in rank:
                rows.append(rank[x]); b_idx.append(b); t_idx.append(t)
        base += len(k)
    r = torch.tensor(rows, dtype=torch.long, device=device)
    cnt = torch.bincount(r, minlength=base).clamp(min=1).float()[:, None]
    return (r, torch.tensor(b_idx, dtype=torch.long, device=device),
            torch.tensor(t_idx, dtype=torch.long, device=device), base, cnt)


def pool(acts: torch.Tensor, idx) -> torch.Tensor:
    """(B, T, d) -> (단어 수, d). 단어에 속한 토큰 activation 의 평균."""
    r, b, t, base, cnt = idx
    out = torch.zeros(base, acts.shape[-1], device=acts.device, dtype=torch.float32)
    out.index_add_(0, r, acts[b, t].float())
    return out / cnt


# ------------------------------------------------------------------ 한 짝
def probe_pair(th: Arch, ttok, lo: Arch, ltok, fit, held, pairs, gammas,
               chunk, bs, device, log=print, tag=""):
    """pairs = [(l, i, mass)]. 결과 행 목록."""
    rows = []
    tl_all = sorted({l for l, _, _ in pairs})
    pre_fit = [SP.text_of(x) for x in fit]
    pre_held = [SP.text_of(x) for x in held]

    def batches(pre):
        for s in range(0, len(pre), bs):
            tx = [p[0] for p in pre[s:s + bs]]
            st = [p[1] for p in pre[s:s + bs]]
            ia, wa, na = SP.encode(ttok, tx, st)
            ib, wb, nb = SP.encode(ltok, tx, st)
            keep = [SP.common(x, y, n) for x, y, n in zip(wa, wb, na)]
            yield ia, wa, ib, wb, keep

    for c0 in range(0, len(tl_all), chunk):
        t0 = time.time()
        tls = tl_all[c0:c0 + chunk]
        P = [(l, i, m) for l, i, m in pairs if l in tls]
        lis = sorted({i for _, i, _ in P})
        t_ins = [(l, k) for l in tls for k in KINDS]
        t_outs = [(l, r) for l in tls for r in ROLES]
        l_ins = [(i, k) for i in lis for k in KINDS]
        l_outs = [(i, r) for i in lis for r in ROLES]

        V: Dict[tuple, Var] = {}
        for l, k in t_ins:
            V[("t", "in", l, k)] = Var(th.in_dim(k), True, device)
        for l, r in t_outs:
            V[("t", "out", l, r)] = Var(th.out_dim(r), False, device)
        for i, k in l_ins:
            V[("l", "in", i, k)] = Var(lo.in_dim(k), False, device)
        for i, r in l_outs:
            V[("l", "out", i, r)] = Var(lo.out_dim(r), True, device)
        regs = []      # (종류, 이름, l, i, mass, x키, y키, Reg)
        for l, i, m in P:
            for k in KINDS:      # InputMap: theta 입력 -> local 입력
                xk, yk = ("t", "in", l, k), ("l", "in", i, k)
                regs.append(("in", k, l, i, m, xk, yk, Reg(V[xk], V[yk])))
            for r in ROLES:      # OutputMap: local 출력 -> theta 출력
                xk, yk = ("l", "out", i, r), ("t", "out", l, r)
                regs.append(("out", r, l, i, m, xk, yk, Reg(V[xk], V[yk])))

        def pooled(ia, wa, ib, wb, keep):
            """두 모델의 단어 단위 activation. 공통 단어가 없으면 None."""
            if sum(len(k) for k in keep) == 0:
                return None
            a = run(th, ttok, ia, t_ins, t_outs, max(tls), device)
            b = run(lo, ltok, ib, l_ins, l_outs, max(lis), device)
            xa, xb = pool_index(wa, keep, device), pool_index(wb, keep, device)
            Z = {}
            for (l, k) in t_ins:
                Z[("t", "in", l, k)] = pool(a[("in", l, k)], xa)
            for (l, r) in t_outs:
                Z[("t", "out", l, r)] = pool(a[("out", l, r)], xa)
            for (i, k) in l_ins:
                Z[("l", "in", i, k)] = pool(b[("in", i, k)], xb)
            for (i, r) in l_outs:
                Z[("l", "out", i, r)] = pool(b[("out", i, r)], xb)
            return Z

        # ---- 적합
        n_sp = 0
        for bt in batches(pre_fit):
            Z = pooled(*bt)
            if Z is None:
                continue
            Zc = {key: V[key].add(z) for key, z in Z.items()}
            n_sp += next(iter(Z.values())).shape[0]
            for *_, xk, yk, rg in regs:
                rg.add(Zc[xk], Zc[yk])
        # ---- 풀기 (같은 x 끼리 한 번 분해)
        maps = {}
        groups: Dict[tuple, list] = {}
        for row in regs:
            groups.setdefault(row[5], []).append(row[7])
        for xk, rgs in groups.items():
            maps.update(solve_group(V[xk], rgs, gammas))
        for row in regs:
            row[7].Sxy = None
        torch.cuda.empty_cache()
        # ---- 판정 (학습에 안 쓴 질의)
        H = {id(row[7]): Held(V[row[6]].d, device) for row in regs}
        for bt in batches(pre_held):
            Z = pooled(*bt)
            if Z is None:
                continue
            for *_, xk, yk, rg in regs:
                H[id(rg)].add(Z[xk], Z[yk], V[xk].mean, V[yk].mean, maps[id(rg)])
        for kind, name, l, i, m, xk, yk, rg in regs:
            rows.append({"map": kind, "name": name, "l": l, "i": i, "mass": m,
                         "r2": {str(c): v for c, v in H[id(rg)].r2().items()}})
        del maps, H, V, regs
        torch.cuda.empty_cache()
        log(f"  {tag} theta 층 {tls}  local 층 {lis}  단어 {n_sp}  "
            f"{time.time() - t0:.0f}s")
    return rows


# ------------------------------------------------------------------ 요약
def summarize(rows, gammas, name, log=print):
    names = list(KINDS) + list(ROLES)
    med = {c: {n: float(np.median([r["r2"][str(c)] for r in rows
                                   if (r["map"] == "in" and r["name"] == n and n in KINDS)
                                   or (r["map"] == "out" and r["name"] == n and n in ROLES)]))
               for n in names} for c in gammas}
    best = max(gammas, key=lambda c: np.median(list(med[c].values())))
    log(f"\n[{name}]  R^2 중앙값 (판정 질의, gamma 배수 {best} 기준)")
    log("  InputMap   " + "  ".join(f"{k}={med[best][k]:+.3f}" for k in KINDS))
    log("  OutputMap  " + "  ".join(f"{r}={med[best][r]:+.3f}" for r in ROLES))
    for c in gammas:
        log(f"    gamma 배수 {c:g}: 전체 중앙값 {np.median(list(med[c].values())):+.3f}")
    # 전달 가능한 질량: 역할 rho 의 쌍은 InputMap(그 입력 종류)과 OutputMap(rho)이
    # 모두 기준을 넘어야 전달된다 (문서).
    key = {(r["map"], r["name"], r["l"], r["i"]): r for r in rows}
    pairs = sorted({(r["l"], r["i"], r["mass"]) for r in rows})
    tot = sum(m for _, _, m in pairs)
    log("  전달 가능 질량 비율 (두 R^2 모두 기준 이상인 쌍의 질량 / 전체 질량)")
    log("    기준  " + "  ".join(f"{r:>6s}" for r in ROLES))
    out = {}
    for t in (0.3, 0.5, 0.7):
        fr = []
        for rho in ROLES:
            ok = sum(m for l, i, m in pairs
                     if key[("in", IN_KIND[rho], l, i)]["r2"][str(best)] >= t
                     and key[("out", rho, l, i)]["r2"][str(best)] >= t)
            fr.append(ok / tot if tot else 0.0)
        out[t] = dict(zip(ROLES, fr))
        log(f"    {t:.1f}   " + "  ".join(f"{x:6.1%}" for x in fr))
    return {"median": {str(c): med[c] for c in gammas}, "best_gamma": best,
            "transferable": {str(t): v for t, v in out.items()}}


# ------------------------------------------------------------------ 자기검사
def selftest(th: Arch, ttok, fit, device, bs):
    print("[자기검사 1] 역할 분해: 출력 == 입력 @ W^T")
    tx = [SP.text_of(x)[0] for x in fit[:4]]
    enc = ttok(tx, return_tensors="pt", padding=True).to(device)
    check_roles(th, enc["input_ids"], enc["attention_mask"])
    print("  통과")
    print("[자기검사 2] theta -> theta 자신. 같은 층이면 R^2 = 1 이어야 한다")
    L = th.L
    pairs = [(l, l, 1.0) for l in sorted({0, L // 2, L - 1})]
    rows = probe_pair(th, ttok, th, ttok, fit[:96], fit[:96], pairs, [1e-4],
                      chunk=3, bs=bs, device=device, tag="자기")
    worst = min(min(r["r2"].values()) for r in rows)
    print(f"  최저 R^2 {worst:.6f}")
    if worst < 0.99:
        bad = [(r["map"], r["name"], r["l"], r["r2"]) for r in rows
               if min(r["r2"].values()) < 0.99]
        raise SystemExit(f"자기검사 실패. 채집·풀링·회귀 중 하나가 틀렸다: {bad[:5]}")
    print("  통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--out", default="/content/runs/hetero")
    ap.add_argument("--theta", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--locals", nargs="*", default=DEFAULT_LOCALS)
    ap.add_argument("--n-fit", type=int, default=512)
    ap.add_argument("--n-held", type=int, default=128)
    ap.add_argument("--held-offset", type=int, default=4096,
                    help="sel 이 시작하는 위치 (= n_train)")
    ap.add_argument("--gammas", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--mass-min", type=float, default=0.05)
    ap.add_argument("--theta-stride", type=int, default=2,
                    help="theta 층을 몇 칸마다 잴지. 1 이면 전부")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)
    items = json.load(open(a.data, encoding="utf-8"))["items"]
    fit = items[:a.n_fit]
    held = items[a.held_offset:a.held_offset + a.n_held]
    if not held or len(fit) < a.n_fit:
        raise SystemExit(f"질의가 부족하다: 전체 {len(items)}")
    print(f"[데이터] 적합 {len(fit)} 질의 (train 앞)  판정 {len(held)} 질의 (sel)")

    ttok, th = load(a.theta, device)
    print(f"[theta] {a.theta}  {th.describe()}")
    selftest(th, ttok, fit, device, a.bs)

    result = {"theta": a.theta, "args": vars(a), "locals": {}}
    path = os.path.join(a.out, "h0_r2probe.json")
    for name in a.locals:
        t0 = time.time()
        try:
            ltok, lo = load(name, device)
        except Exception as e:
            print(f"\n*** {name} 를 읽지 못했다. 건너뛴다: {str(e)[:300]}")
            continue
        print(f"\n[local] {name}  {lo.describe()}")
        tx = [SP.text_of(x)[0] for x in fit[:4]]
        enc = ltok(tx, return_tensors="pt", padding=True).to(device)
        check_roles(lo, enc["input_ids"], enc["attention_mask"])
        plan = OT.ratio_plan(th.L, lo.L)
        OT.check_plan(plan)
        keep_l = set(range(0, th.L, a.theta_stride)) | {th.L - 1}
        pairs = [p for p in OT.active_pairs(plan, a.mass_min) if p[0] in keep_l]
        print(f"  Layer Ratio 쌍 {len(pairs)}  (theta 층 {len(keep_l)}개)")
        rows = probe_pair(th, ttok, lo, ltok, fit, held, pairs, a.gammas,
                          a.chunk, a.bs, device, tag=name.split('/')[-1])
        summ = summarize(rows, a.gammas, name)
        result["locals"][name] = {"arch": lo.describe(), "rows": rows, "summary": summ,
                                  "sec": time.time() - t0}
        json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        del lo
        torch.cuda.empty_cache()
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
