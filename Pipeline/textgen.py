# -*- coding: utf-8 -*-
"""텍스트 수준 충실도.

**두 기준선을 모두 낸다.** 둘은 다른 질문에 답한다.

    vs Target   모델 생성 문장 vs Target 응답(gold).
                "작은 모델이 Target 과 비슷한 답을 내는가". 추출 충실도다.

    vs ref      모델 생성 문장 vs 데이터셋 정답 문장(ref).
                LoRD 논문 Table 1 이 쓰는 기준이다. 그 표와 나란히 놓으려면
                이쪽이 있어야 한다. Victim 자신도 이 기준으로 잰다.

    Fidelity F  eq:12 (LoRD).  F = M(y_Nt, y) / M(y_vic, y)
                분모는 Victim 을 ref 에 대고 잰 값이다. Victim 이 1.000 이고,
                추출 모델이 Victim 의 몇 %까지 왔는지를 말한다.
                모델 크기가 달라도 비교할 수 있는 유일한 축이다.

지표는 LoRD §5 와 같다. BLEU-1 / BLEU-4 / ROUGE-L / BERTScore-F1.

생성은 왼쪽 padding 이다. decoder-only 모델에서 오른쪽 padding 으로 배치
생성을 하면 짧은 프롬프트가 padding 위치에서 생성을 시작한다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import METRICS, Config

# sacrebleu 의 13a 토크나이저를 줄인 것. 구두점을 떼고 공백으로 나눈다.
_PUNCT = re.compile(r"([\.,!?\"';:\(\)\[\]<>/\\])")


def tokenize(s: str) -> List[str]:
    s = _PUNCT.sub(r" \1 ", s.strip())
    return s.lower().split()


# ---------------------------------------------------------------- BLEU
def _ngrams(toks: Sequence[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    """말뭉치 BLEU. 참조는 하나다."""
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
    return bp * math.exp(logp)


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
    return tot / max(len(hyps), 1)


# ---------------------------------------------------------------- BERTScore
_BERT: Dict[str, object] = {}
_BERT_OFF = [False]


def _scorer(model: str, log=print):
    """BERTScorer 를 한 번만 만들어 재사용한다. score() 를 부를 때마다
    roberta-large 를 다시 올리면 arm 마다 두 번씩 1.4GB 를 다시 읽는다."""
    if model in _BERT:
        return _BERT[model]
    from bert_score import BERTScorer
    sc = BERTScorer(model_type=model, batch_size=32, rescale_with_baseline=False,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    _BERT[model] = sc
    log(f"  BERTScore 준비 {model}")
    return sc


def bert_f1(hyps: Sequence[str], refs: Sequence[str], model: str,
            log=print) -> Optional[float]:
    """빈 문자열은 bert-score 의 sent_encode 를 깨뜨린다(최신 transformers 에서
    build_inputs_with_special_tokens 가 없다). 빈 가설은 F1 0 으로 세고, 참조가
    비면 그 쌍을 빼고 잰다. 어떤 이유로든 실패하면 None 을 돌려주고 나머지
    지표는 그대로 낸다 --- 지표 하나 때문에 실행 전체가 죽으면 안 된다."""
    if not model or _BERT_OFF[0]:
        return None
    pairs = [(h.strip(), r.strip()) for h, r in zip(hyps, refs)]
    usable = [(h, r) for h, r in pairs if r]              # 참조가 없으면 못 잰다
    if not usable:
        return None
    ok = [(h, r) for h, r in usable if h]                 # 빈 가설은 0 점
    if not ok:
        return 0.0
    try:
        sc = _scorer(model, log)
        _, _, f = sc.score([h for h, _ in ok], [r for _, r in ok])
    except ImportError:
        log("  BERTScore 건너뜀 (pip install bert-score)")
        _BERT_OFF[0] = True
        return None
    except Exception as e:                                 # 버전 충돌 등
        log(f"  BERTScore 실패로 이후 건너뜀: {type(e).__name__}: {e}")
        _BERT_OFF[0] = True
        return None
    return float(f.sum()) / len(usable)                    # 빈 가설 = 0 으로 평균


def score_pair(hyps: Sequence[str], refs: Sequence[str], cfg: Config,
               log=print) -> Dict[str, float]:
    """한 쌍에 대한 네 지표. 0~1 범위로 둔다(LoRD Table 1 과 같은 축)."""
    r: Dict[str, float] = {
        "BLEU-1": bleu(hyps, refs, 1),
        "BLEU-4": bleu(hyps, refs, 4),
        "ROUGE-L": rouge_l(hyps, refs),
        "len": sum(len(tokenize(h)) for h in hyps) / max(len(hyps), 1),
    }
    b = bert_f1(hyps, refs, cfg.bert_score_model, log)
    if b is not None:
        r["BERT-F1"] = b
    return r


def victim_scores(gold: Sequence[str], ref: Sequence[str], cfg: Config,
                  log=print) -> Dict[str, float]:
    """Victim 자신을 ref 에 대고 잰 값. Fidelity F 의 분모다."""
    v = score_pair(gold, ref, cfg, log)
    log("  " + f"{'Target Model (vs ref)':22s}" +
        "  ".join(f"{m} {v.get(m, float('nan')):.4f}" for m in METRICS
                  if m in v))
    return v


def score_text(hyps: Sequence[str], gold: Sequence[str], ref: Sequence[str],
               cfg: Config, victim: Optional[Dict[str, float]] = None,
               tag: str = "", log=print) -> Dict[str, object]:
    """생성 문장을 두 기준선에 대고 재고, Fidelity F 를 붙인다."""
    out: Dict[str, object] = {
        "vs_target": score_pair(hyps, gold, cfg, log),
        "vs_ref": score_pair(hyps, ref, cfg, log),
    }
    if victim:
        vr = out["vs_ref"]                                  # type: ignore
        out["F"] = {m: (vr[m] / victim[m]) for m in METRICS
                    if m in vr and victim.get(m, 0.0) > 0}
    if tag:
        vt, vr = out["vs_target"], out["vs_ref"]            # type: ignore
        f = out.get("F") or {}
        log(f"  {tag:15s} "
            f"vsTgt B4 {vt['BLEU-4']:.4f} RL {vt['ROUGE-L']:.4f} | "
            f"vsRef B4 {vr['BLEU-4']:.4f} RL {vr['ROUGE-L']:.4f} | "
            f"F(RL) {f.get('ROUGE-L', float('nan')):.3f}")
    return out


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


# ---------------------------------------------------------------- 생성 비교
def _rouge_each(hyps: Sequence[str], refs: Sequence[str],
                beta: float = 1.2) -> List[float]:
    """질의별 ROUGE-L. 부트스트랩에 쓴다."""
    out = []
    for h, r in zip(hyps, refs):
        ht, rt = tokenize(h), tokenize(r)
        if not ht or not rt:
            out.append(0.0)
            continue
        l = _lcs(ht, rt)
        if l == 0:
            out.append(0.0)
            continue
        p, q = l / len(ht), l / len(rt)
        out.append(((1 + beta ** 2) * p * q) / (q + beta ** 2 * p))
    return out


def text_compare(hyp_a: Sequence[str], hyp_b: Sequence[str],
                 refs: Sequence[str], tag: str = "", n: int = 2000,
                 seed: int = 1, log=print) -> Dict[str, tuple]:
    """같은 질의를 함께 재표집하는 paired bootstrap (Koehn 2004).

    BLEU 는 말뭉치 지표라 질의별로 쪼갤 수 없다. 표본을 다시 뽑아 그 표본에서
    말뭉치 BLEU 를 다시 계산한다. ROUGE-L 은 문장별 값이 있으므로 그대로 쓴다.
    지금까지 생성 지표에 구간이 없어서 0.4896 과 0.4145 의 차이가 유의한지
    말할 수 없었다.
    """
    N = len(refs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, (n, N))

    ra = np.asarray(_rouge_each(hyp_a, refs))
    rb = np.asarray(_rouge_each(hyp_b, refs))
    d_r = ra - rb
    s_r = d_r[idx].mean(1)

    d_b = bleu(hyp_a, refs, 4) - bleu(hyp_b, refs, 4)
    s_b = np.empty(n)
    for t in range(n if n <= 400 else 400):          # BLEU 재계산은 비싸다
        j = idx[t]
        s_b[t] = (bleu([hyp_a[i] for i in j], [refs[i] for i in j], 4)
                  - bleu([hyp_b[i] for i in j], [refs[i] for i in j], 4))
    m = min(n, 400)
    out = {
        "ROUGE-L": (float(d_r.mean()), float(np.percentile(s_r, 2.5)),
                    float(np.percentile(s_r, 97.5))),
        "BLEU-4": (float(d_b), float(np.percentile(s_b[:m], 2.5)),
                   float(np.percentile(s_b[:m], 97.5))),
    }
    if tag:
        for k, (d, lo, hi) in out.items():
            v = "양수" if lo > 0 else ("음수" if hi < 0 else "불확실")
            log(f"  {tag:38s} {k:8s} {d:+.5f}  [{lo:+.5f}, {hi:+.5f}]  {v}")
    return out
