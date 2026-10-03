# -*- coding: utf-8 -*-
"""이기종 1 단계. theta 와 이기종 Local 을 X 전체로 LoRD 학습한다.

    python run/h1_train.py --out /content/drive/MyDrive/metamon_runs/hetero

Method 대로:
  - 각 theta_Local,k 는 **X 와 Y_Target 전체**로 학습한다 (조각 없음). 결과가 theta_Single,k.
  - theta 도 같은 X, Y_Target 으로 먼저 학습한다. Llama-3.2-3B 를 같은 설정
    (lord_lr, gold_eos, sel 최적 시점 저장)으로 X 전체에 학습한 것이 이미 있다
    (동종 run 의 lord_all). --theta-delta 로 그것을 그대로 쓴다. 없거나 설정이
    다르면 theta 도 여기서 학습한다.
  - 저장하는 것은 eq:local_update 의 Δw = w_Single - w_Local 이다.

모델마다 Colab 로컬 디스크에 먼저 저장하고, Drive 로 복사한 뒤 크기와 읽기를
확인한다. Drive 에서 바로 이름을 바꾸다 파일이 사라진 일이 있었다 (local_6).
이미 Drive 에 있는 모델은 건너뛴다. 끊겨도 같은 명령으로 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open

from Pipeline.config import Config, load as load_cfg
from Pipeline.data import load_dataset_file
from Pipeline.deltastore import DeltaStore, st_path
from Pipeline.hetero.space import HeteroSpace
from Pipeline.lord import train_lord
from Pipeline.metrics import EvalSet, loss
from Pipeline.modeling import load_tokenizer, setup_precision

DEFAULT_LOCALS = ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it",
                  "microsoft/Phi-3-mini-4k-instruct",
                  "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
NAME = "single"


def short(name: str) -> str:
    return name.split("/")[-1]


def copy_verified(src: str, dst: str) -> None:
    """Drive 에서는 이름 바꾸기를 하지 않는다. 바로 쓰고 크기·읽기를 확인한다."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")
    with safe_open(dst, framework="pt", device="cpu") as f:
        if len(list(f.keys())) == 0:
            raise SystemExit(f"Drive 사본을 읽을 수 없다: {dst}")


def theta_ok(path: str) -> str:
    """재사용할 theta Δw 가 쓸 만한가. 문제가 있으면 이유, 없으면 ''."""
    if not path or not os.path.exists(path):
        return "파일이 없다"
    cj = os.path.join(os.path.dirname(path), "config.json")
    if not os.path.exists(cj):
        return "옆에 config.json 이 없어 설정을 확인할 수 없다"
    c = load_cfg(cj)
    if not c.gold_eos:
        return "EOS 수정 전 학습이다 (gold_eos=False)"
    return ""


def train_one(name: str, a, device: str, log=print) -> dict:
    s = short(name)
    cfg = Config(base=name, out_root=os.path.join(a.local_dir, s),
                 dataset_file=os.path.abspath(a.data), fleet_method="lord")
    if a.lord_lr is not None:
        cfg.lord_lr = a.lord_lr
    cfg.makedirs()
    tok = load_tokenizer(cfg)
    items = load_dataset_file(cfg.dataset_path, cfg, tok)
    train = items[:cfg.n_train]
    sel_items = items[cfg.n_train:cfg.n_train + cfg.n_sel]
    if len(train) < cfg.n_train or len(sel_items) < cfg.n_sel:
        raise SystemExit(f"질의 부족: {len(items)}")

    ws = HeteroSpace(cfg, device)
    log(f"[{s}] {ws.arch.describe()}  학습 키 {len(ws.keys)}개 "
        f"({sorted({k[1] for k in ws.keys})})")
    sel = EvalSet(sel_items, tok, device, cfg.eval_bs)
    ws.reset()
    base_L = loss(ws.model, sel)
    log(f"[{s}] base L(theta;1) = {base_L:.5f}   lord_lr {cfg.lord_lr}  "
        f"질의 {len(train)} (X 전체)")

    t0 = time.time()
    train_lord(ws, tok, cfg, NAME, train, cfg.seed,
               health=lambda: loss(ws.model, sel), log=log)

    st = DeltaStore(ws, cfg, [NAME], log=lambda *_: None)
    vals = {}
    for sc in (1.0, 0.5, 0.25):
        ws.apply(lambda k: st.raw(NAME, k), sc)
        vals[sc] = loss(ws.model, sel)
    ws.reset()
    st.close()
    ok = vals[1.0] < base_L
    log(f"[{s}] 판정 @1.0 {vals[1.0]:.5f} vs base {base_L:.5f}  "
        f"{'통과' if ok else '실패'}   (@0.5 {vals[0.5]:.5f}  @0.25 {vals[0.25]:.5f})")

    dst = os.path.join(a.out, s, f"{NAME}.safetensors")
    copy_verified(st_path(cfg, NAME), dst)
    json.dump({"cfg": cfg.to_dict()}, open(os.path.join(a.out, s, "config.json"), "w",
                                           encoding="utf-8"), indent=2, ensure_ascii=False)
    info = {"name": name, "arch": ws.arch.describe(), "L": ws.arch.L,
            "keys": "module", "delta": dst, "base_L": base_L,
            "L_at": {str(k): v for k, v in vals.items()}, "pass": ok,
            "sec": time.time() - t0}
    del ws, sel
    torch.cuda.empty_cache()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--out", default="/content/drive/MyDrive/metamon_runs/hetero")
    ap.add_argument("--local-dir", default="/content/hetero_local")
    ap.add_argument("--theta", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--theta-delta", default="/content/runs/l32/ckpt/lord_all.safetensors",
                    help="같은 설정으로 X 전체에 학습한 theta 의 Δw. 없으면 여기서 학습")
    ap.add_argument("--locals", nargs="*", default=DEFAULT_LOCALS)
    ap.add_argument("--lord-lr", type=float, default=None)
    a = ap.parse_args()

    device = setup_precision()
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    man = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) \
        else {"theta": None, "locals": {}}

    def save():
        json.dump(man, open(mpath, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- theta
    why = theta_ok(a.theta_delta)
    if man.get("theta") and os.path.exists(man["theta"]["delta"]):
        print(f"[theta] 이미 있음: {man['theta']['delta']}")
    elif not why:
        c = load_cfg(os.path.join(os.path.dirname(a.theta_delta), "config.json"))
        man["theta"] = {"name": c.base, "keys": "role", "delta": a.theta_delta,
                        "reused": True, "lord_lr": c.lord_lr, "gold_eos": c.gold_eos}
        print(f"[theta] 동종 run 의 lord_all 을 쓴다: {a.theta_delta}  "
              f"({c.base}, lr {c.lord_lr}, gold_eos {c.gold_eos})")
        save()
    else:
        print(f"[theta] 재사용 불가 ({why}). 여기서 학습한다.")
        man["theta"] = train_one(a.theta, a, device)
        man["theta"]["keys"] = "role"
        save()

    # ---- Local
    for name in a.locals:
        s = short(name)
        dst = os.path.join(a.out, s, f"{NAME}.safetensors")
        if name in man["locals"] and os.path.exists(dst):
            print(f"\n[{s}] 이미 있음. 건너뜀")
            continue
        print(f"\n========== {name} ==========")
        try:
            man["locals"][name] = train_one(name, a, device)
        except (SystemExit, Exception) as e:
            msg = str(e)
            if "게이트" in msg or "gated" in msg.lower() or "403 Client" in msg:
                print(f"*** {name}: 게이트된 저장소라 건너뛴다. "
                      f"https://huggingface.co/{name} 에서 라이선스에 동의한 뒤 다시 돌릴 것.")
                torch.cuda.empty_cache()
                continue
            raise
        save()

    print("\n[요약]  sel L(theta;1)")
    print(f"  {'모델':32s} {'base':>8s} {'@1.0':>8s} {'@0.5':>8s} {'@0.25':>8s}  판정")
    for name, x in man["locals"].items():
        la = x["L_at"]
        print(f"  {short(name):32s} {x['base_L']:8.5f} {la['1.0']:8.5f} "
              f"{la['0.5']:8.5f} {la['0.25']:8.5f}  {'통과' if x['pass'] else '실패'}")
    print(f"\n저장: {mpath}")


if __name__ == "__main__":
    main()
