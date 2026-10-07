# -*- coding: utf-8 -*-
"""이기종 5 단계. 생성 지표 (E1 / E1-b / E4 의 텍스트 열).

    python run/h5_generate.py --manifest /content/drive/MyDrive/metamon_runs/hetero/manifest.json

test 의 앞 --text-n 질의 (기본 256, 동종 run 과 같은 질의) 에 대해 greedy 로 생성하고
LoRD §5 와 같은 지표를 낸다.

    vs Target   생성 vs Target 응답 (gold).    추출 충실도 (E1-b)
    vs ref      생성 vs 데이터셋 정답 (ref).   LoRD Table 1 축 (E1)
    F           ROUGE-L(arm, ref) / ROUGE-L(Target, ref)   (LoRD eq:12)

arm (있는 것만)
    Basic theta          Llama-3.2-3B-Instruct (학습 없음)
    theta (LoRD)         + lord_all
    theta + z            + lord_all + 조립 z (h3b recipe, 학습 없음)
    theta' w/o WL        + h4 의 lambda=0 arm (theta_prime_wo.safetensors)
    theta' (METAMON)     + h4 가 고른 arm   (theta_prime.safetensors)
    Local-k + LoRD       각 theta_Single,k (자기 tokenizer 로 생성. 텍스트 지표는 tokenizer 와 무관)

생성 결과는 arm 마다 Drive 에 저장한다 (끊기면 이어 간다).
METAMON 과 다른 arm 의 차이는 paired bootstrap (ROUGE-L 문장별, BLEU-4 재표집) 으로 낸다.
저장 (h5/<plan>/): hyps.json, report_gen.md, report_gen.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from Pipeline.config import Config
from Pipeline.data import load_dataset_file
from Pipeline.hetero.load import load_tok, load_trained
from Pipeline.hetero.space import attn_impl
from Pipeline.modeling import setup_precision
from Pipeline.textgen import generate, score_text, text_compare, victim_scores


def short(name: str) -> str:
    return name.split("/")[-1]


def save_json(a, name, obj):
    tmp = os.path.join(a.local_dir, name)
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    dst = os.path.join(a.out, name)
    shutil.copyfile(tmp, dst)
    if os.path.getsize(dst) != os.path.getsize(tmp):
        raise SystemExit(f"Drive 복사 크기 불일치: {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/content/drive/MyDrive/metamon_runs/hetero/manifest.json")
    ap.add_argument("--data", default="/content/runs/l32/target/dataset.json")
    ap.add_argument("--plan", default="ot")
    ap.add_argument("--out", default=None, help="기본: manifest 옆 h5/<plan>")
    ap.add_argument("--local-dir", default="/content/hetero_local/h5")
    ap.add_argument("--n-x", type=int, default=4096)
    ap.add_argument("--n-sel", type=int, default=128)
    ap.add_argument("--n-check", type=int, default=256)
    ap.add_argument("--text-n", type=int, default=256)
    a = ap.parse_args()
    root = os.path.dirname(a.manifest)
    a.out = a.out or os.path.join(root, "h5", a.plan)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.local_dir, exist_ok=True)
    device = setup_precision()

    man = json.load(open(a.manifest, encoding="utf-8"))
    tinfo = man["theta"]
    tname = tinfo["name"]
    cfg = Config(base=tname, dataset_file=os.path.abspath(a.data), gold_eos=True)
    ttok = load_tok(tname)
    items = load_dataset_file(cfg.dataset_path, cfg, ttok, log=lambda *_: None)
    t0_ = a.n_x + a.n_sel + a.n_check
    pool = items[t0_:t0_ + a.text_n]
    gold = [x["gold"].strip() for x in pool]
    ref = [x["ref"] for x in pool]
    print(f"[test] 질의 {t0_}..{t0_ + len(pool) - 1} ({len(pool)}개)  greedy  "
          f"max_new {cfg.gen_max_new}  BERTScore {cfg.bert_score_model}")

    h4 = os.path.join(root, "h4", a.plan)
    rec_path = os.path.join(root, "h3", a.plan, "select_check.json")
    arms = [("Basic theta", tname, None, "base"),
            ("theta (LoRD)", tname, tinfo["delta"], "trained"),
            ("theta + z (assembly)", tname, tinfo["delta"], "assembly"),
            ("theta' w/o Weight Loss", tname, os.path.join(h4, "theta_prime_wo.safetensors"), "trained"),
            ("theta' (METAMON)", tname, os.path.join(h4, "theta_prime.safetensors"), "trained")]
    for n, info in man["locals"].items():
        arms.append((f"Local {short(n)} + LoRD", n, info["delta"], "trained"))

    hyp_path = os.path.join(a.out, "hyps.json")
    hyps = json.load(open(hyp_path, encoding="utf-8")) if os.path.exists(hyp_path) else {}
    if hyps.get("_pool") != [t0_, len(pool)]:
        hyps = {"_pool": [t0_, len(pool)]}

    for label, name, delta, kind in arms:
        if label in hyps:
            print(f"  {label:32s} 생성 저장본 사용")
            continue
        if delta is not None and not os.path.exists(delta):
            print(f"  {label:32s} 건너뜀 (없음: {delta})")
            continue
        if kind == "base":
            model = AutoModelForCausalLM.from_pretrained(
                name, dtype=torch.float32, attn_implementation=attn_impl(name)).to(device).eval()
        else:
            arch, _ = load_trained(name, delta, device)
            model = arch.model
            if kind == "assembly":
                if not os.path.exists(rec_path):
                    print(f"  {label:32s} 건너뜀 (recipe 없음)")
                    del arch, model
                    continue
                rec = json.load(open(rec_path, encoding="utf-8"))
                zf = [safe_open(os.path.join(root, "z", a.plan, f"{short(n)}.safetensors"),
                                framework="pt", device="cpu") for n in rec["names"]]
                with torch.no_grad():
                    for c in rec["main"]["cells"]:
                        W = arch.weight(c["l"], c["role"])
                        d = zf[c["k"]].get_tensor(f"{c['l']}.{c['role']}").to(device).float()
                        W.add_(d, alpha=c["a"] * float(W.norm()) / max(float(d.norm()), 1e-30))
        model.config.use_cache = True          # load_trained 은 학습용으로 꺼 둔다
        model.generation_config.max_length = None
        tok = ttok if name == tname else load_tok(name)
        print(f"  {label:32s} 생성 중 ...", flush=True)
        hyps[label] = generate(model, tok, pool, cfg, device)
        save_json(a, "hyps.json", hyps)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[지표]")
    victim = victim_scores(gold, ref, cfg)
    labels = [l for l, *_ in arms if l in hyps]
    text = {lab: score_text(hyps[lab], gold, ref, cfg, victim=victim, tag=lab[:15]) for lab in labels}

    def f(d, m):
        v = d.get(m)
        return "-" if v is None else f"{v:.4f}"

    md = ["# METAMON heterogeneous - generation metrics\n",
          f"- test items {t0_}..{t0_ + len(pool) - 1} ({len(pool)}), greedy, max_new_tokens {cfg.gen_max_new}",
          "- vs ref = dataset reference (E1, LoRD Table 1 axis); vs Target = Target (gpt-3.5) response (E1-b); "
          "F = ROUGE-L(arm, ref) / ROUGE-L(Target, ref)\n",
          "## E1 - vs ref\n",
          "| Method | BLEU-4 | ROUGE-L | BERTScore-F1 | Fidelity F |", "|---|---|---|---|---|",
          f"| Target Model | {f(victim, 'BLEU-4')} | {f(victim, 'ROUGE-L')} | {f(victim, 'BERT-F1')} | 1.000 |"]
    for lab in labels:
        vr, F = text[lab]["vs_ref"], text[lab].get("F", {})
        md.append(f"| {lab} | {f(vr, 'BLEU-4')} | {f(vr, 'ROUGE-L')} | {f(vr, 'BERT-F1')} | "
                  f"{F.get('ROUGE-L', float('nan')):.3f} |")
    md += ["", "## E1-b - vs Target response\n",
           "| Method | BLEU-4 | ROUGE-L | BERTScore-F1 | avg len (words) |", "|---|---|---|---|---|"]
    for lab in labels:
        vt = text[lab]["vs_target"]
        md.append(f"| {lab} | {f(vt, 'BLEU-4')} | {f(vt, 'ROUGE-L')} | {f(vt, 'BERT-F1')} | {vt['len']:.1f} |")

    main = "theta' (METAMON)"
    cmp_out = {}
    if main in hyps:
        md += ["", "## METAMON minus other (vs Target, paired bootstrap 95% CI)\n",
               "| Other | ROUGE-L diff | BLEU-4 diff |", "|---|---|---|"]
        for lab in labels:
            if lab == main:
                continue
            c = text_compare(hyps[main], hyps[lab], gold, tag=f"METAMON - {lab}"[:38])
            cmp_out[lab] = c
            rl, b4 = c["ROUGE-L"], c["BLEU-4"]
            md.append(f"| {lab} | {rl[0]:+.4f} [{rl[1]:+.4f}, {rl[2]:+.4f}] | "
                      f"{b4[0]:+.4f} [{b4[1]:+.4f}, {b4[2]:+.4f}] |")
    md += ["", "- Sample (first test item):",
           f"  - prompt: {pool[0]['prompt'][-200:]!r}",
           f"  - Target: {gold[0]!r}"]
    for lab in labels:
        md.append(f"  - {lab}: {hyps[lab][0]!r}")
    md_text = "\n".join(md) + "\n"
    print("\n" + md_text)
    tmp = os.path.join(a.local_dir, "report_gen.md")
    open(tmp, "w", encoding="utf-8").write(md_text)
    shutil.copyfile(tmp, os.path.join(a.out, "report_gen.md"))
    save_json(a, "report_gen.json", {"victim": victim, "text": text,
                                     "compare": {k: {m: list(v) for m, v in d.items()}
                                                 for k, d in cmp_out.items()}})
    print(f"저장: {a.out}/report_gen.md  report_gen.json  hyps.json")


if __name__ == "__main__":
    main()
