# -*- coding: utf-8 -*-
"""InputMap / OutputMap 을 theta 층 묶음 단위로 푼다 (eq:input_map, eq:output_map).

데이터는 셋으로 나눈다.
    fit    InputMap / OutputMap 을 적합한다              (train 앞부분)
    gsel   gamma 를 고른다. 회귀마다 R^2 가 가장 높은 값  (train 다음 부분, fit 과 겹치지 않음)
    held   R^2 판정 (문서의 "학습에 사용되지 않은 질의")   (sel)
gamma 를 판정 데이터로 고르면 판정이 낙관적으로 된다. 그래서 따로 둔다.

activation 은 두 모델에 **같은 문자열**을 넣고 단어 구간 c 로 평균한 것이다 (spans).
모델은 fp32 로 올린다. Δw 는 BASE 대비 1e-3 수준이라 bf16 이면 학습 결과가 묻힌다.

한 번에 주어진 회귀 묶음만 푼다. 호출한 쪽(h2)이 메모리 추정으로 묶음을 나누고,
결과로 Δz 를 계산한 뒤 버린다.

GPU 에 동시에 두는 것은 적합 통계(교차항, x 산포)와 회귀마다 **고른 gamma 의 해 하나**
뿐이다. gamma 후보의 해를 전부 들고 있지 않는다. x 를 공유하는 회귀 묶음마다
gamma 하나씩 풀어 gsel R^2 를 바로 재고 더 나은 해만 남긴다. 묶음이 끝나면 그
교차항과 x 산포를 버린다. gsel / held activation 은 작아서(질의 128 개) 먼저 한 번
뽑아 CPU 에 두고, 잴 때만 변수 하나씩 GPU 로 올린다.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from . import spans as SP
from .ridge import Reg, Var, chol
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


def _r2(Xc: torch.Tensor, Y: torch.Tensor, mu_y: torch.Tensor, B: torch.Tensor) -> float:
    """R^2 = 1 - SSE/SST.  SSE 는 적합 평균 기준, SST 는 그 데이터 자신의 평균 기준."""
    r = (Y - mu_y) - Xc @ B
    sse = float((r.double() ** 2).sum())
    sst = float(((Y - Y.mean(0)).double() ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def solve_maps(th: Arch, ttok, lo: Arch, ltok, fit, gsel, held,
               need_in: Iterable[Tuple[int, int, str]],
               need_out: Iterable[Tuple[int, int, str]],
               gammas: Sequence[float], bs: int, device: str, log=print):
    """need_in  = {(l, i, kind)}   InputMap: theta 입력 -> local 입력
    need_out = {(l, i, role)}   OutputMap: local 출력 -> theta 출력

    반환 결과[(종류, l, i, 이름)] =
        {"B": Map^T (d_x x d_y), "c": 고른 gamma 배수, "r2": 판정 R^2, "r2_gsel": {c: R^2}}
    """
    t0 = time.time()
    need_in, need_out = sorted(set(need_in)), sorted(set(need_out))
    tls = sorted({l for l, _, _ in need_in + need_out})
    lis = sorted({i for _, i, _ in need_in + need_out})
    t_ins = sorted({(l, k) for l, _, k in need_in})
    t_outs = sorted({(l, r) for l, _, r in need_out})
    l_ins = sorted({(i, k) for _, i, k in need_in})
    l_outs = sorted({(i, r) for _, i, r in need_out})
    dims = {}
    for l, k in t_ins:
        dims[("t", "in", l, k)] = (th.in_dim(k), True)
    for l, r in t_outs:
        dims[("t", "out", l, r)] = (th.out_dim(r), False)
    for i, k in l_ins:
        dims[("l", "in", i, k)] = (lo.in_dim(k), False)
    for i, r in l_outs:
        dims[("l", "out", i, r)] = (lo.out_dim(r), True)
    regs = ([(("in", l, i, k), ("t", "in", l, k), ("l", "in", i, k)) for l, i, k in need_in]
            + [(("out", l, i, r), ("l", "out", i, r), ("t", "out", l, r)) for l, i, r in need_out])
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

    # ---- gsel / held activation (CPU)
    cache = {}
    for nm in ("gsel", "held"):
        parts = {key: [] for key in dims}
        for bt in batches(nm):
            for key, z in pooled(*bt).items():
                parts[key].append(z.cpu())
        cache[nm] = {key: torch.cat(v) for key, v in parts.items()}
        del parts

    # ---- 적합 통계
    V = {key: Var(d, sec, device) for key, (d, sec) in dims.items()}
    R = {rk: Reg(V[xk], V[yk]) for rk, xk, yk in regs}
    n_sp = 0
    for bt in batches("fit"):
        Z = pooled(*bt)
        Zc = {key: V[key].add(z) for key, z in Z.items()}
        n_sp += next(iter(Z.values())).shape[0]
        for rk, xk, yk in regs:
            R[rk].add(Zc[xk], Zc[yk])
        del Z, Zc

    # ---- x 묶음마다: gamma 하나씩 풀고 gsel R^2 로 고른다 -> held R^2
    groups: Dict[tuple, list] = {}
    for rk, xk, yk in regs:
        groups.setdefault(xk, []).append((rk, yk))
    res = {}
    for xk, members in groups.items():
        vx = V[xk]
        mu_x = vx.mean
        Xg = cache["gsel"][xk].to(device) - mu_x
        best = {rk: (None, None, -float("inf")) for rk, _ in members}
        r2g = {rk: {} for rk, _ in members}
        for c in gammas:
            L = chol(vx, c)
            for rk, yk in members:
                B = torch.cholesky_solve(R[rk].cross(), L).float()
                v = _r2(Xg, cache["gsel"][yk].to(device), V[yk].mean, B)
                r2g[rk][str(c)] = v
                if v > best[rk][2]:
                    best[rk] = (B, c, v)
                del B
            del L
        del Xg
        Xh = cache["held"][xk].to(device) - mu_x
        for rk, yk in members:
            B, c, _ = best[rk]
            res[rk] = {"B": B, "c": c, "r2_gsel": r2g[rk],
                       "r2": _r2(Xh, cache["held"][yk].to(device), V[yk].mean, B)}
            R[rk].Sxy = None
        del Xh, best
        vx.S2 = None
        torch.cuda.empty_cache()
    del cache, V, R
    torch.cuda.empty_cache()
    log(f"    theta 층 {tls}  local 층 {len(lis)}개  회귀 {len(res)}  단어 {n_sp}  "
        f"{time.time() - t0:.0f}s  GPU 최대 {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    return res
