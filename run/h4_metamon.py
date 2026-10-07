# -*- coding: utf-8 -*-
"""이기종 4 단계. eq:metamon_loss 로 최종 theta' 를 학습한다.

    python run/h4_metamon.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

    L_Response(theta') = L(theta'; 1)  +  lambda * sum_{l,rho} Confidence_l * || w_theta',l,rho - (w_theta,l,rho + z_l,rho) ||_F^2

  - L(theta'; 1)  X (train 4096) 와 Y_Target 에 대한 eq:weighted_loss (omega = 1), 즉 MLE.
  - z, Confidence  h3b 가 check 로 고른 조립 recipe (select_check.json). z 가 없는 칸은 0.
  - theta' 는 theta (Llama-3.2-3B + LoRD Δw) 에서 시작한다.
  - 학습 대상은 theta 의 7 역할 선형층 전부 (LoRD 와 같은 공간), fp32, AdamW.

lambda 격자
  Confidence 와 ||z||^2 의 자릿수는 데이터마다 다르다. 그래서 시작점에서의 두 항
  R0 = sum Confidence ||z||^2 과 L0 = L(theta; 1) 로 단위 lambda_u = L0 / R0 를 잡고
  lambda = r * lambda_u, r in --lam-rel 로 둔다 (r = 0 은 w/o Weight Loss). 실제
  lambda 값은 출력과 보고서에 그대로 남긴다.

선택 (test 는 보지 않는다)
  - 시점: check L 이 가장 낮은 update 의 가중치 (--eval-every 마다 잰다).
  - lambda: 그 check L 이 가장 낮은 것.
  고른 lambda 로 LOO theta'_-k 를 학습한다 (select_check.json 의 loo recipe).

보고 (h4/<plan>/report_h4.md, report_h4.json)
  theta, theta + z (조립), theta' (lambda 별), LOO theta'_-k 의 check / test L,
  theta 대비 paired bootstrap 95% CI, avgBF, 그리고 eq:dependency_mitigation 용
  Var_k(avgBF) (theta_Single,k 대 LOO).
고른 theta' 의 Δw (= w - Llama BASE, fp16, 키 "{i}.{rho}") 를 Drive 에 저장한다.
--save-loo 면 LOO 도 저장한다 (하나에 약 6GB).

arm 하나가 끝날 때마다 결과를 Drive 에 쓴다. 끊기면 같은 명령으로 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_

from Pipeline.config import Config
from Pipeline.data import load_dataset_file
from Pipeline.hetero.contrib import TailEval
from Pipeline.hetero.load import load_tok
from Pipeline.hetero.space import HeteroSpace
from Pipeline.lord import mean_logp, token_logp
from Pipeline.metrics import paired_bootstrap
from Pipeline.modeling import setup_precision


def short(name: str) -> str:
    return name.split("/")[-1]


def copy_verified(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")


def save_json(a, rel: str, obj) -> None:
    tmp = os.path.join(a.local_dir, rel)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    copy_verified(tmp, os.path.join(a.out, rel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/content/drive/MyDrive/metamon_runs/hetero/manifest.json")
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--plan", default="ot")
    ap.add_argument("--recipe", default=None, help="기본: h3/<plan>/select_check.json")
    ap.add_argument("--out", default=None, help="기본: manifest 옆 h4/<plan>")
    ap.add_argument("--local-dir", default="/content/hetero_local/h4")
    ap.add_argument("--lam-rel", type=float, nargs="*", default=[0.0, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--lr", type=float, default=None, help="기본: cfg.sft_lr")
    ap.add_argument("--epochs", type=int, default=None, help="기본: cfg.sft_epochs")
    ap.add_argument("--eval-every", type=int, default=64, help="update 단위")
    ap.add_argument("--patience", type=int, default=4, help="check 개선 없는 평가 횟수")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32, help="평가 배치")
    ap.add_argument("--save-loo", action="store_true")
    ap.add_argument("--skip-loo", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(a.manifest)
    a.out = a.out or os.path.join(root, "h4", a.plan)
    a.recipe = a.recipe or os.path.join(root, "h3", a.plan, "select_check.json")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()

    man = json.load(open(a.manifest, encoding="utf-8"))
    rec = json.load(open(a.recipe, encoding="utf-8"))
    names = rec["names"]
    tinfo = man["theta"]
    print(f"[recipe] {a.recipe}  omega {rec['omega']}  A {rec['alphas']}  "
          f"칸 {len(rec['main']['cells'])}  고른 기준 {rec.get('chosen_by')}")

    cfg = Config(base=tinfo["name"], dataset_file=os.path.abspath(a.data), gold_eos=True,
                 out_root=a.local_dir)
    lr = a.lr if a.lr is not None else cfg.sft_lr
    epochs = a.epochs if a.epochs is not None else cfg.sft_epochs
    tok = load_tok(tinfo["name"])
    items = load_dataset_file(cfg.dataset_path, cfg, tok, log=lambda *_: None)
    c0 = a.n_x + a.n_sel
    t0_ = c0 + a.n_check
    train, chk, test = items[:a.n_x], items[c0:t0_], items[t0_:t0_ + a.n_test]
    if len(test) < a.n_test:
        raise SystemExit(f"질의 부족: {len(items)}")

    ws = HeteroSpace(cfg, device)
    keys = ws.keys
    lf = safe_open(tinfo["delta"], framework="pt", device="cpu")
    zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                    framework="pt", device="cpu") for n in names]
    ev_c = TailEval(ws.arch, tok, chk, a.bs, device)
    ev_t = TailEval(ws.arch, tok, test, a.bs, device)

    def to_theta():
        ws.apply(lambda k: lf.get_tensor(f"{k[0]}.{k[1]}"))

    def anchor(recipe):
        """w_theta + z (eq:norm_matching 로 h3 와 같은 z), 그리고 sum Confidence ||z||^2."""
        to_theta()
        A, R0 = {}, 0.0
        cells = {(c["l"], c["role"]): c for c in recipe["cells"]}
        conf = recipe["confidence"]
        with torch.no_grad():
            for k in keys:
                W = ws.lin[k].weight
                A[k] = W.detach().clone()
                c = cells.get(k)
                if c is None:
                    continue
                d = zf[c["k"]].get_tensor(f"{k[0]}.{k[1]}").to(device).float()
                d.mul_(c["a"] * float(W.norm()) / max(float(d.norm()), 1e-30))
                A[k].add_(d)
                R0 += conf[k[0]] * float((d * d).sum())
                del d
        return A, R0

    def check_L():
        ws.model.eval()
        v = ev_c.mlp()
        ws.model.train()
        return v

    # ---- 기준: theta, theta + z (h3b 와 같은 값이어야 한다)
    to_theta()
    lp_theta = {"check": ev_c.mlp(), "test": ev_t.mlp()}
    L0 = -float(lp_theta["check"].mean())
    A_main, R0_main = anchor(rec["main"])
    with torch.no_grad():
        for k in keys:
            ws.lin[k].weight.copy_(A_main[k])
    lp_asm = {"check": ev_c.mlp(), "test": ev_t.mlp()}
    del A_main
    torch.cuda.empty_cache()
    rc = os.path.join(root, "h3", a.plan, "report_check.json")
    if os.path.exists(rc):
        g = json.load(open(rc, encoding="utf-8"))
        ref = [r for r in g["grid"] if r["omega"] == g["chosen"]["omega"] and r["A"] == g["chosen"]["A"]]
        if ref:
            diff = abs(ref[0]["check_L"] + float(lp_asm["check"].mean()))
            print(f"[재현] theta+z check L {-lp_asm['check'].mean():.5f}  vs h3b {ref[0]['check_L']:.5f}  "
                  f"(차 {diff:.1e})")
            if diff > 1e-4:
                raise SystemExit("recipe 로 만든 z 가 h3b 의 조립과 다르다. 멈춘다.")
    lam_u = L0 / max(R0_main, 1e-30)
    print(f"[theta]   check L {L0:.5f}  test L {-lp_theta['test'].mean():.5f}")
    print(f"[theta+z] check L {-lp_asm['check'].mean():.5f}  test L {-lp_asm['test'].mean():.5f}")
    print(f"[lambda] R0 = sum Conf ||z||^2 = {R0_main:.4e}   lambda_u = L0/R0 = {lam_u:.4e}   "
          f"lr {lr}  epochs {epochs}  배치 {cfg.acc}  clip {cfg.grad_clip}")

    # ---- arm 하나
    def train_arm(tag, lam, recipe, save_to=None):
        path = os.path.join("arms", f"{tag}.json")
        if os.path.exists(os.path.join(a.out, path)):
            r = json.load(open(os.path.join(a.out, path), encoding="utf-8"))
            if save_to is None or os.path.exists(save_to):
                print(f"\n[{tag}] 이미 있음 (check {r['check_L']:.5f}  test {r['test_L']:.5f})")
                return r
            print(f"\n[{tag}] 결과는 있으나 Δw 저장본이 없어 다시 학습한다")
        print(f"\n========== {tag}  lambda {lam:.4e} ==========", flush=True)
        t_start = time.time()
        A, R0 = anchor(recipe) if lam > 0 else (None, 0.0)
        to_theta()
        conf = recipe["confidence"]

        def reg():
            if A is None:
                return 0.0
            with torch.no_grad():
                return sum(conf[k[0]] * float(((ws.lin[k].weight - A[k]) ** 2).sum()) for k in keys)

        ws.trainable(True)
        opt = torch.optim.AdamW(ws.params, lr=lr, foreach=True)
        best = {"L": None, "upd": 0}
        buf = {k: torch.empty(tuple(ws.lin[k].weight.shape), dtype=torch.float32)
               for k in keys}

        def evaluate(upd):
            v = check_L()
            L = -float(v.mean())
            if best["L"] is None or L < best["L"]:
                with torch.no_grad():
                    for k in keys:
                        buf[k].copy_(ws.lin[k].weight.detach())
                best.update(L=L, upd=upd, stale=0)
                mark = "최적"
            else:
                best["stale"] = best.get("stale", 0) + 1
                mark = f"정체 {best['stale']}"
            print(f"  [{tag}] u{upd:5d}  check L {L:.5f}  reg {reg():.4e}  {mark} "
                  f"(최적 u{best['upd']} {best['L']:.5f})  {time.time() - t_start:.0f}s", flush=True)

        ws.model.train()
        evaluate(0)
        upd, stop, run_nll = 0, False, []
        try:
            for e in range(epochs):
                order = list(range(len(train)))
                random.Random(cfg.seed + e).shuffle(order)
                for s in range(0, len(order) - cfg.acc + 1, cfg.acc):
                    sl = order[s:s + cfg.acc]
                    lp, m = token_logp(ws.model, [train[j]["pid"] for j in sl],
                                       [train[j]["gid"] for j in sl], tok.pad_token_id, device)
                    L = -mean_logp(lp, m).mean()
                    if not torch.isfinite(L):
                        raise SystemExit(f"{tag}: 비정상 손실")
                    opt.zero_grad(set_to_none=True)
                    L.backward()
                    if A is not None:           # d/dW  lambda * C_l * ||W - A||^2
                        with torch.no_grad():
                            for k in keys:
                                p = ws.lin[k].weight
                                p.grad.add_(p.detach() - A[k], alpha=2.0 * lam * conf[k[0]])
                    clip_grad_norm_(ws.params, cfg.grad_clip)
                    opt.step()
                    upd += 1
                    run_nll.append(float(L.detach()))
                    if upd % a.eval_every == 0:
                        print(f"    u{upd} train nll {np.mean(run_nll):.4f}", flush=True)
                        run_nll = []
                        evaluate(upd)
                        if best.get("stale", 0) >= a.patience:
                            stop = True
                            break
                if stop:
                    print(f"  [{tag}] check 가 {a.patience} 번 개선되지 않아 멈춘다")
                    break
            if not stop and upd % a.eval_every:
                evaluate(upd)
        finally:
            ws.model.eval()
            ws.trainable(False)
            del opt
            torch.cuda.empty_cache()

        with torch.no_grad():
            for k in keys:
                ws.lin[k].weight.copy_(buf[k].to(device))
        lpc, lpt = ev_c.mlp(), ev_t.mlp()
        dist = reg()
        out = {"tag": tag, "lambda": lam, "lam_rel": lam / lam_u if lam_u else 0.0,
               "best_update": best["upd"], "updates_run": upd,
               "check_L": -float(lpc.mean()), "test_L": -float(lpt.mean()),
               "reg_final": dist, "reg_start": R0,
               "check_lp": lpc.tolist(), "test_lp": lpt.tolist(),
               "sec": time.time() - t_start}
        print(f"  [{tag}] 저장 시점 u{best['upd']}  check {out['check_L']:.5f}  "
              f"test {out['test_L']:.5f}  sum Conf||w-(w+z)||^2 {R0:.3e} -> {dist:.3e}")
        # Δw (= w - Llama BASE) 를 Colab 로컬에 둔다. 고른 arm 만 Drive 로 옮긴다.
        tmp = os.path.join(a.local_dir, "delta", f"{tag}.safetensors")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        sd = {f"{k[0]}.{k[1]}": (buf[k] - ws.base[k]).to(torch.float16) for k in keys}
        save_file(sd, tmp)
        del sd
        out["delta_local"] = tmp
        if save_to is not None:
            copy_verified(tmp, save_to)
            out["delta"] = save_to
            print(f"  [{tag}] Δw 저장: {save_to}")
        save_json(a, path, out)
        del buf, A
        torch.cuda.empty_cache()
        return out

    # ---- lambda 격자
    arms = {}
    for r in a.lam_rel:
        tag = f"lam_{r:g}"
        arms[tag] = train_arm(tag, r * lam_u, rec["main"])
    best_tag = min(arms, key=lambda t: arms[t]["check_L"])
    lam_star = arms[best_tag]["lambda"]
    print(f"\n[선택] check L 최소: {best_tag}  lambda {lam_star:.4e}  check {arms[best_tag]['check_L']:.5f}")
    main_path = os.path.join(a.out, "theta_prime.safetensors")
    if not os.path.exists(main_path):
        lt = arms[best_tag].get("delta_local")
        if lt and os.path.exists(lt):
            copy_verified(lt, main_path)
            print(f"  Δw 저장: {main_path}")
        else:      # 런타임이 바뀌어 로컬 사본이 없다
            os.remove(os.path.join(a.out, "arms", f"{best_tag}.json"))
            arms[best_tag] = train_arm(best_tag, lam_star, rec["main"], save_to=main_path)
    arms[best_tag]["delta"] = main_path
    save_json(a, os.path.join("arms", f"{best_tag}.json"), arms[best_tag])

    # ---- LOO
    loo = {}
    if not a.skip_loo:
        for n in names:
            tag = f"loo_{short(n)}"
            sv = os.path.join(a.out, f"theta_prime_loo_{short(n)}.safetensors") if a.save_loo else None
            loo[n] = train_arm(tag, lam_star, rec["loo"][n], save_to=sv)

    # ---- 표
    base_t = np.asarray(lp_theta["test"])
    Lt0 = -float(base_t.mean())

    def row(label, check_L, lpt, extra=""):
        lpt = np.asarray(lpt)
        Lt = -float(lpt.mean())
        if label.startswith("theta (LoRD)"):
            d = "-"
        else:
            m, lo_, hi_ = paired_bootstrap(-lpt, -base_t)
            d = f"{m:+.5f} [{lo_:+.5f}, {hi_:+.5f}]"
        rel = "-" if d == "-" else f"{(Lt - Lt0) / Lt0:+.2%}"
        return (f"| {label} | {check_L:.5f} | {Lt:.5f} | {d} | {rel} | "
                f"{np.exp(lpt).mean():.4f} | {extra} |")

    md = ["# METAMON heterogeneous - final theta' (eq:metamon_loss)\n",
          f"- theta: {tinfo['name']} (+LoRD Δw); Locals: {', '.join(short(n) for n in names)}; plan {a.plan}",
          f"- recipe: {os.path.basename(a.recipe)} (omega {rec['omega']}, A {rec['alphas']}, chosen by {rec.get('chosen_by')})",
          f"- L(theta';1) on X ({a.n_x}) with Y_Target; AdamW lr {lr}, batch {cfg.acc}, clip {cfg.grad_clip}, "
          f"<= {epochs} epochs; checkpoint and lambda chosen by check L ({a.n_check}); reported on test ({a.n_test})",
          f"- lambda = r * L0/R0, L0 = {L0:.5f}, R0 = sum Conf ||z||^2 = {R0_main:.4e}\n",
          "| Setting | check L | test L | test ΔL vs theta [95% CI] | rel. | test avgBF | lambda (r) / stop |",
          "|---|---|---|---|---|---|---|",
          row("theta (LoRD)", L0, base_t),
          row("theta + z (assembly, no training)", -float(np.mean(lp_asm["check"])), lp_asm["test"])]
    for t, r_ in arms.items():
        lab = "theta' lambda=0 (w/o Weight Loss)" if r_["lambda"] == 0 else f"theta' (METAMON)"
        if t == best_tag:
            lab = f"**{lab} (chosen)**"
        md.append(row(lab, r_["check_L"], r_["test_lp"],
                      f"{r_['lambda']:.3e} ({r_['lam_rel']:g}) / u{r_['best_update']}"))
    for n, r_ in loo.items():
        md.append(row(f"theta'_-{short(n)} (LOO)", r_["check_L"], r_["test_lp"],
                      f"{r_['lambda']:.3e} / u{r_['best_update']}"))

    # eq:dependency_mitigation
    h3r = os.path.join(root, "h3", a.plan, "report.json")
    if loo:
        bf_loo = [float(np.exp(np.asarray(r_["test_lp"])).mean()) for r_ in loo.values()]
        md += ["", f"- Dependency (Var of test avgBF) over LOO theta'_-k: {np.var(bf_loo):.3e}  "
                   f"(avgBF {[round(x, 4) for x in bf_loo]})"]
        if os.path.exists(h3r):
            sg = json.load(open(h3r, encoding="utf-8")).get("single_test_lp", {})
            if sg:
                bf_s = [float(np.exp(np.asarray(v)).mean()) for v in sg.values()]
                md.append(f"- Dependency over theta_Single,k: {np.var(bf_s):.3e}  "
                          f"(avgBF {[round(x, 4) for x in bf_s]}; own tokenizers)")
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)
    tmp = os.path.join(a.local_dir, "report_h4.md")
    open(tmp, "w", encoding="utf-8").write(md_text)
    copy_verified(tmp, os.path.join(a.out, "report_h4.md"))
    save_json(a, "report_h4.json", {
        "L0_check": L0, "R0": R0_main, "lambda_u": lam_u, "lr": lr, "epochs": epochs,
        "chosen": best_tag, "lambda_star": lam_star,
        "theta": {k: np.asarray(v).tolist() for k, v in lp_theta.items()},
        "assembly": {k: np.asarray(v).tolist() for k, v in lp_asm.items()},
        "arms": arms, "loo": loo})
    print(f"저장: {a.out}/report_h4.md  report_h4.json  theta_prime.safetensors")


if __name__ == "__main__":
    main()
