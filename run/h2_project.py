# -*- coding: utf-8 -*-
"""이기종 2 단계. Common Weight Space Projection.

    python run/h2_project.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

Local 마다 순서대로
  1. eq:behavior_signature   s = || d mean_logp / d w ||_F^2   (theta 가 가진 질의, 각자의 tokenizer)
                             split full 이면 X 전체 4096, disjoint 이면 X_theta. theta' 의 소유자는
                             Local 의 가중치만 받으므로 sensitivity 와 Mapping 은 자기 질의로 잰다.
  2. eq:sensitivity_profile  평균 제거 (ot.alignment_cost 안에서)
  3. eq:alignment_cost       역할 rho 마다 (L_theta x L_k)
  4. eq:ot_alignment         Pi_k 위의 OT. 세 계획을 모두 저장한다
                               ot     Sinkhorn, tau = tau_rel * std(Cost_rho)
                               exact  tau = 0 (선형계획)                 E4 대조
                               ratio  Layer Ratio Mapping (깊이 단조 수송)  E4 대조
                             Δz 에는 --plan 하나를 쓴다.
                             tau_rel 0.1 이면 계획이 거의 가득 차 쌍이 역할당 600 개가 넘는다
                             (회귀가 그만큼 늘어난다). 0.02 와 질량 1e-2 미만 제외로 역할당
                             90 개 안팎, 남는 질량 99.5% 이상. 버린 질량은 meta 에 남긴다.
  5. eq:input_map / eq:output_map  ridge. fit / gamma 선택 / R^2 판정 데이터가 서로 다르다
  6. R^2 게이트              InputMap(kind(rho)) 와 OutputMap(rho) 둘 다 --r2-min 이상
  7. eq:weight_projection    Δz_{l,rho} = sum_i m * OutputMap Δw_{i,rho} InputMap

Δz 는 bf16 로 저장한다. Δw 크기가 작아 fp16 이면 아래쪽이 0 으로 떨어진다.
Colab 로컬 디스크에 먼저 쓰고 Drive 로 복사한 뒤 확인한다.
sensitivity 는 out/prof/ 에 남겨 두어 다시 돌릴 때 건너뛴다.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from Pipeline.config import Config
from Pipeline.data import load_dataset_file
from Pipeline.hetero import ot as OT
from Pipeline.hetero import sensitivity as SE
from Pipeline.hetero import split as SPL
from Pipeline.hetero.load import RoleDelta, load_tok, load_trained
from Pipeline.hetero.mapping import solve_maps
from Pipeline.hetero.roles import IN_KIND, ROLES, check_roles
from Pipeline.lord import mean_logp, token_logp
from Pipeline.modeling import setup_precision


def short(name: str) -> str:
    return name.split("/")[-1]


def copy_verified(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")
    if dst.endswith(".safetensors"):
        with safe_open(dst, framework="pt", device="cpu") as f:
            if len(list(f.keys())) == 0:
                raise SystemExit(f"Drive 사본을 읽을 수 없다: {dst}")


def items_for(name: str, data: str):
    tok = load_tok(name)
    cfg = Config(base=name, dataset_file=os.path.abspath(data), gold_eos=True)
    return tok, load_dataset_file(cfg.dataset_path, cfg, tok, log=lambda *_: None), cfg


# ------------------------------------------------------------------ 자기검사
def check_sensitivity(arch, tok, items, device, log=print) -> None:
    """ghost-norm 값이 질의 하나씩 직접 구한 gradient 의 Frobenius 제곱과 같은가.
    묶음 안 질의가 섞이지 않는지도 같이 본다 (두 질의를 한 묶음으로 넣는다)."""
    two = items[:2]
    P = SE.profiles(arch, tok, two, 2, device, log=lambda *_: None)
    worst = 0.0
    for l in (0, arch.L - 1):
        for r in ("Query", "Value", "Down"):
            mod, lo, hi = arch.out_spec(l, r)
            mod.weight.requires_grad_(True)
            for n in range(2):
                mod.weight.grad = None
                with torch.enable_grad():
                    lp, m = token_logp(arch.model, [two[n]["pid"]], [two[n]["gid"]],
                                       tok.pad_token_id, device)
                    mean_logp(lp, m).sum().backward()
                g = mod.weight.grad if lo is None else mod.weight.grad[lo:hi]
                ref = float((g.double() ** 2).sum())
                got = float(P[l, ROLES.index(r), n])
                rel = abs(got - ref) / max(ref, 1e-30)
                worst = max(worst, rel)
                if rel > 5e-3:
                    raise SystemExit(f"sensitivity 자기검사 실패: 층 {l} {r} 질의 {n}  "
                                     f"ghost {got:.6e}  직접 {ref:.6e}  (상대오차 {rel:.2e})")
            mod.weight.grad = None
            mod.weight.requires_grad_(False)
    arch.model.zero_grad(set_to_none=True)
    log(f"    sensitivity 자기검사 통과 (최대 상대오차 {worst:.1e})")


def roles_check(arch, tok, items, device) -> None:
    ids = [list(x["pid"]) + list(x["gid"]) for x in items[:4]]
    T = max(len(x) for x in ids)
    inp = torch.full((len(ids), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for b, x in enumerate(ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    check_roles(arch, inp.to(device), att.to(device))


# ------------------------------------------------------------------ sensitivity
def get_profiles(tag, arch, tok, items, a, device, log=print) -> np.ndarray:
    path = os.path.join(a.out, "prof", f"{tag}.npy")
    if os.path.exists(path):
        P = np.load(path)
        if P.shape == (arch.L, len(ROLES), len(items)):
            log(f"    sensitivity 저장본 사용: {path}")
            return P
    t0 = time.time()
    P = SE.profiles(arch, tok, items, a.sens_bs, device, log=log)
    if not np.isfinite(P).all():
        raise SystemExit(f"{tag}: sensitivity 에 nan/inf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(a.local_dir, "prof", f"{tag}.npy")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    np.save(tmp, P)
    copy_verified(tmp, path)
    log(f"    sensitivity {P.shape}  {time.time() - t0:.0f}s  -> {path}")
    return P


# ------------------------------------------------------------------ OT
def plans_for(Pt, Pk, a, log=print):
    """역할마다 cost 와 세 계획."""
    L_t, L_k = Pt.shape[0], Pk.shape[0]
    out = {}
    R = OT.ratio_plan(L_t, L_k)
    for ri, r in enumerate(ROLES):
        C = OT.alignment_cost(Pt[:, ri, :], Pk[:, ri, :])
        tau = a.tau_rel * float(C.std())
        P = {"ot": OT.sinkhorn(C, tau), "exact": OT.exact_ot(C), "ratio": R}
        for v in P.values():
            OT.check_plan(v)
        # 행동 비용이 깊이와 얼마나 맞는지: OT 계획의 평균 상대깊이 차
        dl = np.abs((np.arange(L_t)[:, None] + .5) / L_t - (np.arange(L_k)[None, :] + .5) / L_k)
        out[r] = {"cost": C, "tau": tau, "plans": P,
                  "depth_gap": {k: float((v * dl).sum() / L_t) for k, v in P.items()},
                  "cost_of": {k: float((v * C).sum() / L_t) for k, v in P.items()}}
        log(f"    {r:6s} cost [{C.min():.3f},{C.max():.3f}] std {C.std():.3f}  tau {tau:.4f}  "
            f"평균cost ot {out[r]['cost_of']['ot']:.3f} exact {out[r]['cost_of']['exact']:.3f} "
            f"ratio {out[r]['cost_of']['ratio']:.3f}  깊이차 ot {out[r]['depth_gap']['ot']:.3f}")
    return out


# ------------------------------------------------------------------ 메모리 묶음
def subsets_by_budget(th, lo, pairs, budget):
    """theta 층마다 local 층을 묶는다. 한 묶음의 GPU 추정량
        x 산포(theta 입력, local 출력) + 교차항 전부 + 가장 큰 x 묶음의 교차항(gamma 후보 해)
        + 여유 2GB
    가 budget 을 넘지 않게. 쌍 (l, i) 의 입력·출력 회귀는 같은 묶음에 들어간다.
    반환 {l: [[i, ...], ...]}"""
    by = {}
    for r, ps in pairs.items():
        for l, i, _ in ps:
            by.setdefault(l, {}).setdefault(i, set()).add(r)
    out = {}
    for l, im in sorted(by.items()):
        def est(ii):
            kinds = {IN_KIND[r] for i in ii for r in im[i]}
            v = sum(th.in_dim(k) ** 2 * 4 for k in kinds)
            grp = {}
            for i in ii:
                for k in {IN_KIND[r] for r in im[i]}:
                    x = th.in_dim(k) * lo.in_dim(k) * 4
                    v += x
                    grp[k] = grp.get(k, 0) + x
                for r in im[i]:
                    v += lo.out_dim(r) * (th.out_dim(r) + lo.out_dim(r)) * 4
            return v + max(grp.values(), default=0) + 2e9
        subs, cur = [], []
        for i in sorted(im):
            if cur and est(cur + [i]) > budget:
                subs.append(cur)
                cur = []
            cur.append(i)
        subs.append(cur)
        out[l] = subs
    return out


# ------------------------------------------------------------------ Local 하나
def project_one(name, info, th, ttok, titems, Pt, a, device, log=print):
    s = short(name)
    zdir = os.path.join(a.out, "z", a.plan)
    zpath, mpath = os.path.join(zdir, f"{s}.safetensors"), os.path.join(zdir, f"{s}.json")
    if os.path.exists(zpath) and os.path.exists(mpath):
        log(f"[{s}] 이미 있음: {zpath}")
        return json.load(open(mpath, encoding="utf-8"))["summary"]

    log(f"\n========== {name} ==========")
    ltok, litems, _ = items_for(name, a.data)
    lo, n = load_trained(name, info["delta"], device)
    log(f"[{s}] {lo.describe()}  Δw 모듈 {n}개 더함")
    roles_check(lo, ltok, litems, device)
    check_sensitivity(lo, ltok, litems, device, log)

    # 1-4
    Pk = get_profiles(s, lo, ltok, SPL.pick(litems, a.own_idx), a, device, log)
    pl = plans_for(Pt, Pk, a, log)
    np.savez(os.path.join(a.local_dir, f"plans_{s}.npz"),
             **{f"{r}.{k}": v for r in ROLES for k, v in pl[r]["plans"].items()},
             **{f"{r}.cost": pl[r]["cost"] for r in ROLES})
    copy_verified(os.path.join(a.local_dir, f"plans_{s}.npz"),
                  os.path.join(a.out, "plans", f"{s}.npz"))

    pairs, dropped = {}, {}
    for r in ROLES:
        m = pl[r]["plans"][a.plan]
        pairs[r] = OT.active_pairs(m, a.mass_min)
        dropped[r] = float(m.sum() - sum(x[2] for x in pairs[r])) / m.shape[0]
    log(f"[{s}] 계획 {a.plan}  질량>={a.mass_min} 쌍 수 "
        + " ".join(f"{r}:{len(pairs[r])}" for r in ROLES)
        + f"   버린 질량 최대 {max(dropped.values()):.2e}")

    # 5-7
    need_in = {(l, i, IN_KIND[r]) for r in ROLES for l, i, _ in pairs[r]}
    need_out = {(l, i, r) for r in ROLES for l, i, _ in pairs[r]}
    own = SPL.pick(titems, a.own_idx)          # theta 가 가진 질의 (X 또는 X_theta)
    fit = own[:a.n_fit]
    gsel = own[a.n_fit:a.n_fit + a.n_gsel]
    if len(gsel) < a.n_gsel:
        raise SystemExit(f"theta 의 질의 {len(own)} 가 fit {a.n_fit} + gsel {a.n_gsel} 보다 적다")
    held = titems[a.n_x:a.n_x + a.n_held]
    D = RoleDelta(lo, info["delta"], device)
    Z: Dict[str, torch.Tensor] = {}
    rec = []
    subs = subsets_by_budget(th, lo, pairs, a.budget_gb * 1e9)
    log(f"[{s}] 회귀 {len(need_in)} + {len(need_out)}  theta 층 {len(subs)}개  "
        f"묶음 {sum(len(v) for v in subs.values())}개 (예산 {a.budget_gb}GB)")
    t0 = time.time()
    for l, sl in subs.items():
        acc: Dict[str, torch.Tensor] = {}
        for ii in sl:
            iset = set(ii)
            torch.cuda.reset_peak_memory_stats()
            res = solve_maps(th, ttok, lo, ltok, fit, gsel, held,
                             {x for x in need_in if x[0] == l and x[1] in iset},
                             {x for x in need_out if x[0] == l and x[1] in iset},
                             a.gammas, bs=a.bs, device=device, log=log)
            for r in ROLES:
                for l2, i, mass in pairs[r]:
                    if l2 != l or i not in iset:
                        continue
                    ri = res[("in", l, i, IN_KIND[r])]
                    ro = res[("out", l, i, r)]
                    ok = ri["r2"] >= a.r2_min and ro["r2"] >= a.r2_min
                    rec.append({"l": l, "i": i, "role": r, "m": mass, "pass": bool(ok),
                                "r2_in": ri["r2"], "r2_out": ro["r2"],
                                "c_in": ri["c"], "c_out": ro["c"]})
                    if not ok:
                        continue
                    # Δz = OutputMap Δw InputMap,  OutputMap = B_out^T, InputMap = B_in^T
                    t = ro["B"].T @ D.get(i, r) @ ri["B"].T
                    if t.shape != th.weight(l, r).shape:
                        raise SystemExit(f"Δz 모양 {tuple(t.shape)} != w_theta "
                                         f"{tuple(th.weight(l, r).shape)} ({l}.{r} <- {i})")
                    if r in acc:
                        acc[r].add_(t, alpha=mass)
                    else:
                        acc[r] = t.mul_(mass)
                    del t
            del res
            torch.cuda.empty_cache()
        for r, v in acc.items():
            if not torch.isfinite(v).all():
                raise SystemExit(f"Δz {l}.{r} 에 nan/inf")
            Z[f"{l}.{r}"] = v.to(torch.bfloat16).cpu()
        del acc
        log(f"    [{s}] theta 층 {l + 1}/{th.L}  {time.time() - t0:.0f}s")

    # 요약
    summ = {"plan": a.plan, "r2_min": a.r2_min, "roles": {}}
    log(f"\n[{s}] 역할별  전달 질량비 / R^2 중앙값(in, out) / ||Δz||/||w_theta|| 중앙값")
    for r in ROLES:
        rr = [x for x in rec if x["role"] == r]
        tot = sum(x["m"] for x in rr)
        ok = sum(x["m"] for x in rr if x["pass"])
        rel = []
        for l in range(th.L):
            k = f"{l}.{r}"
            if k in Z:
                rel.append(float(Z[k].float().norm()) / float(th.weight(l, r).float().norm()))
        summ["roles"][r] = {"mass_ratio": ok / max(tot, 1e-12),
                            "r2_in_med": float(np.median([x["r2_in"] for x in rr])) if rr else None,
                            "r2_out_med": float(np.median([x["r2_out"] for x in rr])) if rr else None,
                            "layers": len(rel),
                            "rel_norm_med": float(np.median(rel)) if rel else 0.0}
        q = summ["roles"][r]
        log(f"  {r:6s} 전달 {q['mass_ratio']:6.1%}   R2 in {q['r2_in_med'] or 0:.3f} "
            f"out {q['r2_out_med'] or 0:.3f}   Δz 층 {q['layers']:2d}/{th.L}   "
            f"||Δz||/||w|| {q['rel_norm_med']:.2e}")
    summ["mass_ratio"] = (sum(x["m"] for x in rec if x["pass"])
                          / max(sum(x["m"] for x in rec), 1e-12))
    log(f"  전체 전달 질량비 {summ['mass_ratio']:.1%}")

    if not Z:
        log(f"*** [{s}] R^2 게이트를 통과한 쌍이 없다. Δz 를 저장하지 않는다.")
    else:
        tmp = os.path.join(a.local_dir, "z", a.plan, f"{s}.safetensors")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        save_file(Z, tmp)
        copy_verified(tmp, zpath)
    meta = {"name": name, "theta": a.theta_name, "plan": a.plan, "args": vars(a),
            "tau": {r: pl[r]["tau"] for r in ROLES},
            "cost_of": {r: pl[r]["cost_of"] for r in ROLES},
            "depth_gap": {r: pl[r]["depth_gap"] for r in ROLES},
            "dropped_mass": dropped, "pairs": rec, "summary": summ,
            "z": zpath if Z else None}
    os.makedirs(zdir, exist_ok=True)
    json.dump(meta, open(mpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"[{s}] 저장: {zpath if Z else '(Δz 없음)'}  {mpath}")

    del lo, D, Z, litems
    gc.collect()
    torch.cuda.empty_cache()
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/content/drive/MyDrive/metamon_runs/hetero/manifest.json")
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--out", default=None, help="기본: manifest 옆")
    ap.add_argument("--local-dir", default="/content/hetero_local/h2")
    ap.add_argument("--locals", nargs="*", default=None, help="기본: manifest 의 전부")
    ap.add_argument("--plan", choices=["ot", "exact", "ratio"], default="ot")
    ap.add_argument("--tau-rel", type=float, default=0.02, help="tau = tau_rel * std(Cost)")
    ap.add_argument("--mass-min", type=float, default=1e-2)
    ap.add_argument("--r2-min", type=float, default=0.3)
    ap.add_argument("--gammas", type=float, nargs="*", default=[0.1, 0.3, 1.0])
    ap.add_argument("--n-x", type=int, default=4096, help="X = train")
    ap.add_argument("--n-fit", type=int, default=512)
    ap.add_argument("--n-gsel", type=int, default=128)
    ap.add_argument("--n-held", type=int, default=128, help="sel (train 다음)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sens-bs", type=int, default=8)
    ap.add_argument("--budget-gb", type=float, default=30.0,
                    help="회귀 통계 GPU 예산. 두 모델(fp32, 최대 28GB)은 따로")
    a = ap.parse_args()
    a.out = a.out or os.path.dirname(a.manifest)
    os.makedirs(a.local_dir, exist_ok=True)

    device = setup_precision()
    man = json.load(open(a.manifest, encoding="utf-8"))
    tinfo = man["theta"]
    a.theta_name = tinfo["name"]
    print(f"[theta] {tinfo['name']}  Δw {tinfo['delta']}")
    ttok, titems, tcfg = items_for(tinfo["name"], a.data)
    if len(titems) < a.n_x + a.n_held:
        raise SystemExit(f"질의 부족: {len(titems)}")
    a.own_idx = SPL.theta_idx(man, a.n_x)
    print(f"[나누기] {SPL.describe(man)}   sensitivity·Mapping 질의 = theta 의 몫 {len(a.own_idx)}")
    th, n = load_trained(tinfo["name"], tinfo["delta"], device)
    print(f"[theta] {th.describe()}  Δw 모듈 {n}개 더함")
    roles_check(th, ttok, titems, device)
    check_sensitivity(th, ttok, titems, device)
    Pt = get_profiles("theta", th, ttok, SPL.pick(titems, a.own_idx), a, device)

    names = a.locals or list(man["locals"].keys())
    out = {}
    for name in names:
        if name not in man["locals"]:
            print(f"*** {name}: manifest 에 없다 (h1 미학습). 건너뜀")
            continue
        out[name] = project_one(name, man["locals"][name], th, ttok, titems, Pt, a, device)

    print(f"\n[요약] 계획 {a.plan}  R^2 >= {a.r2_min}")
    print(f"  {'Local':28s} {'전달':>6s}  " + " ".join(f"{r[:5]:>6s}" for r in ROLES))
    for name, sm in out.items():
        print(f"  {short(name):28s} {sm['mass_ratio']:6.1%}  "
              + " ".join(f"{sm['roles'][r]['mass_ratio']:6.1%}" for r in ROLES))
    print(f"\n저장: {os.path.join(a.out, 'z', a.plan)}")


if __name__ == "__main__":
    main()
