# -*- coding: utf-8 -*-
"""텍스트 수준 충실도.

eq:sim 은 Target 응답에 모델이 부여하는 확률이다. 확률이 높다는 것과 실제로
비슷한 문장을 생성한다는 것은 다르다. "작은 모델이 큰 모델과 비슷한 답변을
내놓는다" 를 주장하려면 모델이 **생성한** 문장을 Target 응답과 비교해야 한다.

비교 대상은 데이터셋 정답(ref)이 아니라 **Target 응답(gold)** 이다. ref 와
비교하면 번역 품질을 재는 것이고, 추출 충실도가 아니다.

지표는 LoRD 논문 §5 와 같은 것을 쓴다.
    BLEU-1 / BLEU-4   n-gram 정밀도 + 길이 벌점
    ROUGE-L           LCS 기반 F1
    BERTScore         bert_score 가 설치돼 있고 cfg.bert_score_model 이
                      비어 있지 않을 때만 잰다

생성은 왼쪽 padding 이다. decoder-only 모델에서 오른쪽 padding 으로 배치
생성을 하면 짧은 프롬프트가 padding 위치에서 생성을 시작한다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch

from .config import Config

# sacrebleu 의 13a 토크나이저를 줄인 것. 구두점을 떼고 공백으로 나눈다.
_PUNCT = re.compile(r"([\.,!?\"';:\(\)\[\]<>/\\])")


def tokenize(s: str) -> List[str]:
    s = _PUNCT.sub(r" \1 ", s.strip())
    return s.lower().split()


# ---------------------------------------------------------------- BLEU
def _ngrams(toks: Sequence[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    """말뭉치 BLEU. 참조는 Target 응답 하나다."""
    num = [0] * max_n
    den = [0] * max_n
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = tokenize(h), tokenize(r)
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            hc, rc = _ngrams(ht, n), _ngrams(rt, n)
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            den[n - 1] += max(sum(hc.values()), 0)
    if min(den) == 0 or min(num) == 0:
        return 0.0
    logp = sum(math.log(num[i] / den[i]) for i in range(max_n)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * math.exp(logp)


# ---------------------------------------------------------------- ROUGE-L
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyps: Sequence[str], refs: Sequence[str], beta: float = 1.2) -> float:
    """문장별 LCS F1 의 평균. 원 논문 정의대로 beta 로 recall 에 가중한다."""
    tot = 0.0
    for h, r in zip(hyps, refs):
        ht, rt = tokenize(h), tokenize(r)
        if not ht or not rt:
            continue
        l = _lcs(ht, rt)
        if l == 0:
            continue
        p, q = l / len(ht), l / len(rt)
        tot += ((1 + beta ** 2) * p * q) / (q + beta ** 2 * p)
    return 100.0 * tot / max(len(hyps), 1)


# ---------------------------------------------------------------- BERTScore
def bert_score(hyps: Sequence[str], refs: Sequence[str], model: str,
               log=print) -> Optional[Dict[str, float]]:
    if not model:
        return None
    try:
        from bert_score import score as _score
    except ImportError:
        log("  BERTScore 건너뜀 (pip install bert-score)")
        return None
    p, r, f = _score(list(hyps), list(refs), model_type=model,
                     verbose=False, batch_size=32)
    return {"P": float(p.mean()) * 100, "R": float(r.mean()) * 100,
            "F1": float(f.mean()) * 100}


# ---------------------------------------------------------------- 생성
@torch.no_grad()
def generate(model, tok, items: Sequence[dict], cfg: Config, device: str,
             log=print) -> List[str]:
    """프롬프트만 주고 생성한다. 왼쪽 padding 을 쓴다."""
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        order = sorted(range(len(items)), key=lambda i: len(items[i]["pid"]))
        out: List[Optional[str]] = [None] * len(items)
        for s in range(0, len(order), cfg.gen_bs):
            sl = order[s:s + cfg.gen_bs]
            enc = tok([items[i]["prompt"] for i in sl], return_tensors="pt",
                      padding=True, add_special_tokens=True).to(device)
            greedy = cfg.text_temp <= 0.0
            gen = model.generate(
                **enc, do_sample=not greedy,
                temperature=None if greedy else cfg.text_temp,
                top_p=None if greedy else cfg.gen_top_p,
                max_new_tokens=cfg.gen_max_new,
                pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(sl):
                out[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            del enc, gen, new
        return out  # type: ignore
    finally:
        tok.padding_side = side


def score_text(hyps: Sequence[str], golds: Sequence[str], cfg: Config,
               tag: str = "", log=print) -> Dict[str, float]:
    """생성 문장 vs Target 응답. ref 가 아니라 gold 와 비교한다."""
    r = {"BLEU-1": bleu(hyps, golds, 1), "BLEU-4": bleu(hyps, golds, 4),
         "ROUGE-L": rouge_l(hyps, golds),
         "len": sum(len(tokenize(h)) for h in hyps) / max(len(hyps), 1)}
    bs = bert_score(hyps, golds, cfg.bert_score_model, log)
    if bs:
        r.update({f"BERT-{k}": v for k, v in bs.items()})
    if tag:
        log(f"  {tag:15s} " + "  ".join(f"{k} {v:6.2f}" for k, v in r.items()))
    return r


# ---------------------------------------------------------------- 비용
def cost_table(model, cfg: Config, n_query: int, log=print) -> Dict[str, object]:
    """파라미터 수와 추론 배수. 병합이 1 배, 앙상블이 K 배라는 것이 논지의 절반
    이므로 수치로 남긴다."""
    n_param = sum(p.numel() for p in model.parameters())
    row = {
        "surrogate": cfg.base,
        "surrogate_params": n_param,
        "target": f"{cfg.target_provider}/{cfg.target_model}",
        "queries_total": n_query,
        "inference_merged": 1,
        "inference_ensemble": cfg.k,
        "models_kept_merged": 1,
        "models_kept_ensemble": cfg.k,
    }
    log("[비용]")
    log(f"  surrogate     {cfg.base}  {n_param/1e9:.2f}B")
    log(f"  Target        {row['target']}")
    log(f"  질의          {n_query}")
    log(f"  추론 비용      병합 1배 (모델 1개)  vs  앙상블 {cfg.k}배 (모델 {cfg.k}개)")
    return row
