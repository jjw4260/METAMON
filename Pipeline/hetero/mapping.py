# -*- coding: utf-8 -*-
"""InputMap / OutputMap 을 theta 층 묶음 단위로 푼다 (eq:input_map, eq:output_map).

데이터는 셋으로 나눈다.
    fit    InputMap / OutputMap 을 적합한다              (train 앞부분)
    gsel   gamma 를 고른다. 회귀마다 R^2 가 가장 높은 값  (train 다음 부분, fit 과 겹치지 않음)
    held   R^2 판정 (문서의 "학습에 사용되지 않은 질의")   (sel)
gamma 를 판정 데이터로 고르면 판정이 낙관적으로 된다. 그래서 따로 둔다.

activation 은 두 모델에 **같은 문자열**을 넣고 단어 구간 c 로 평균한 것이다 (spans).
모델은 fp32 로 올린다. Δw 는 BASE 대비 1e-3 수준이라 bf16 이면 학습 결과가 묻힌다.

한 번에 theta 층 몇 개만 다룬다. 회귀 통계가 층마다 수 GB 라 전부 동시에 둘 수 없다.
묶음마다 결과를 돌려주고, 호출한 쪽이 Δz 를 계산한 뒤 버린다.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from . import spans as SP
from .ridge import Held, Reg, Var, solve_group
from .roles import KINDS, ROLES, Arch, StopForward


def _pad(ids: List[List[int]], pad_id: int, device: str):
    T = max(len(x) for x in ids)
    inp = torch.full((len(ids), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for b, x in enumerate(ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    return inp.to(device), att.to(device)


@torch.no_grad()
def _run(arch: Arch, tok, ids, ins, outs, stop, device):
    inp, att = _pad(ids, tok.pad_token_id, device)
    with arch.capture(ins, outs, stop_after=stop) as st:
        try:
            arch.model(input_ids=inp, attention_mask=att)
        except StopForward:
            pass
    return dict(st)


def _pool_index(wids, keep, device):
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


def _pool(acts, idx):
    r, b, t, base, cnt = idx
    out = torch.zeros(base, acts.shape[-1], device=acts.device, dtype=torch.float32)
    out.index_add_(0, r, acts[b, t].float())
    return out / cnt


def chunks(th: Arch, ttok, lo: Arch, ltok, fit, gsel, held,
           need_in: Iterable[Tuple[int, int, str]],
           need_out: Iterable[Tuple[int, int, str]],
           gammas: Sequence[float], chunk: int, bs: int, device: str, log=print):
    """need_in  = {(l, i, kind)}   InputMap: theta 입력 -> local 입력
    need_out = {(l, i, role)}   OutputMap: local 출력 -> theta 출력

    theta 층 묶음마다 (묶음 층들, 결과) 를 낸다. 결과[(종류, l, i, 이름)] =
        {"B": Map^T (d_x x d_y), "c": 고른 gamma 배수, "r2": 판정 R^2, "r2_gsel": {c: R^2}}
    """
    need_in, need_out = set(need_in), set(need_out)
    tl_all = sorted({l for l, _, _ in need_in | need_out})
    pre = {nm: [SP.text_of(x) for x in d] for nm, d in
           (("fit", fit), ("gsel", gsel), ("held", held))}

    def batches(nm):
        p = pre[nm]
        for s in range(0, len(p), bs):
            tx = [q[0] for q in p[s:s + bs]]
            st = [q[1] for q in p[s:s + bs]]
            ia, wa, na = SP.encode(ttok, tx, st)
            ib, wb, nb = SP.encode(ltok, tx, st)
            keep = [SP.common(x, y, n) for x, y, n in zip(wa, wb, na)]
            if sum(len(k) for k in keep):
                yield ia, wa, ib, wb, keep

    for c0 in range(0, len(tl_all), chunk):
        t0 = time.time()
        tls = tl_all[c0:c0 + chunk]
        Rin = sorted(x for x in need_in if x[0] in tls)
        Rout = sorted(x for x in need_out if x[0] in tls)
        lis = sorted({i for _, i, _ in Rin + Rout})
        t_ins = sorted({(l, k) for l, _, k in Rin})
        t_outs = sorted({(l, r) for l, _, r in Rout})
        l_ins = sorted({(i, k) for _, i, k in Rin})
        l_outs = sorted({(i, r) for _, i, r in Rout})

        V: Dict[tuple, Var] = {}
        for l, k in t_ins:
            V[("t", "in", l, k)] = Var(th.in_dim(k), True, device)
        for l, r in t_outs:
            V[("t", "out", l, r)] = Var(th.out_dim(r), False, device)
        for i, k in l_ins:
            V[("l", "in", i, k)] = Var(lo.in_dim(k), False, device)
        for i, r in l_outs:
            V[("l", "out", i, r)] = Var(lo.out_dim(r), True, device)
        regs = []      # (결과키, x키, y키, Reg)
        for l, i, k in Rin:
            xk, yk = ("t", "in", l, k), ("l", "in", i, k)
            regs.append((("in", l, i, k), xk, yk, Reg(V[xk], V[yk])))
        for l, i, r in Rout:
            xk, yk = ("l", "out", i, r), ("t", "out", l, r)
            regs.append((("out", l, i, r), xk, yk, Reg(V[xk], V[yk])))

        def pooled(ia, wa, ib, wb, keep):
            a = _run(th, ttok, ia, t_ins, t_outs, max(tls), device)
            b = _run(lo, ltok, ib, l_ins, l_outs, max(lis), device)
            xa, xb = _pool_index(wa, keep, device), _pool_index(wb, keep, device)
            Z = {}
            for l, k in t_ins:
                Z[("t", "in", l, k)] = _pool(a[("in", l, k)], xa)
            for l, r in t_outs:
                Z[("t", "out", l, r)] = _pool(a[("out", l, r)], xa)
            for i, k in l_ins:
                Z[("l", "in", i, k)] = _pool(b[("in", i, k)], xb)
            for i, r in l_outs:
                Z[("l", "out", i, r)] = _pool(b[("out", i, r)], xb)
            return Z

        # ---- 적합
        n_sp = 0
        for bt in batches("fit"):
            Z = pooled(*bt)
            Zc = {key: V[key].add(z) for key, z in Z.items()}
            n_sp += next(iter(Z.values())).shape[0]
            for _, xk, yk, rg in regs:
                rg.add(Zc[xk], Zc[yk])
        maps = {}
        groups: Dict[tuple, list] = {}
        for _, xk, _, rg in regs:
            groups.setdefault(xk, []).append(rg)
        for xk, rgs in groups.items():
            maps.update(solve_group(V[xk], rgs, gammas))
        for _, _, _, rg in regs:
            rg.Sxy = None
        torch.cuda.empty_cache()

        # ---- gamma 고르기 (fit 과 겹치지 않는 train 질의)
        Hg = {id(rg): Held(V[yk].d, device) for _, _, yk, rg in regs}
        for bt in batches("gsel"):
            Z = pooled(*bt)
            for _, xk, yk, rg in regs:
                Hg[id(rg)].add(Z[xk], Z[yk], V[xk].mean, V[yk].mean, maps[id(rg)])
        pick = {}
        for _, _, _, rg in regs:
            r2g = Hg[id(rg)].r2()
            c = max(r2g, key=lambda q: r2g[q])
            pick[id(rg)] = (c, r2g)
            maps[id(rg)] = {c: maps[id(rg)][c]}        # 나머지 gamma 는 버린다
        del Hg
        torch.cuda.empty_cache()

        # ---- 판정 R^2 (sel)
        Hh = {id(rg): Held(V[yk].d, device) for _, _, yk, rg in regs}
        for bt in batches("held"):
            Z = pooled(*bt)
            for _, xk, yk, rg in regs:
                Hh[id(rg)].add(Z[xk], Z[yk], V[xk].mean, V[yk].mean, maps[id(rg)])
        res = {}
        for key, xk, yk, rg in regs:
            c, r2g = pick[id(rg)]
            res[key] = {"B": maps[id(rg)][c], "c": c,
                        "r2": Hh[id(rg)].r2().get(c, float("nan")),
                        "r2_gsel": {str(q): v for q, v in r2g.items()}}
        del Hh, V, regs, maps
        log(f"    theta 층 {tls}  local 층 {len(lis)}개  회귀 {len(res)}  "
            f"단어 {n_sp}  {time.time() - t0:.0f}s")
        yield tls, res
        del res
        torch.cuda.empty_cache()
