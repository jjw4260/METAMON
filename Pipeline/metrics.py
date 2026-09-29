# -*- coding: utf-8 -*-
"""지표. Method 의 수식과 1:1 로 대응시킨다.

    eq:mean_log_probability   token_avg_logp   토큰 평균 log-probability
    eq:sim                    sim = exp(위)
    eq:single_fidelity        avg_bf = mean(sim)
    eq:weighted_loss          loss   = -sum(w * logp) / sum(w)
    eq:soft_weight            soft_weight(sim, beta)
    eq:single_dependency      dependency = var(avg_bf)

평가 배치는 한 번 만들어 재사용한다. 길이로 정렬해 padding 을 줄인다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class EvalSet:
    """토크나이즈와 padding 을 한 번만 수행해 두는 평가 집합."""

    def __init__(self, items: Sequence[dict], tok, device: str, bs: int):
        self.bs = bs
        pad = tok.pad_token_id
        order = sorted(range(len(items)),
                       key=lambda i: len(items[i]["pid"]) + len(items[i]["gid"]))
        self.n = len(items)
        self.batches = []
        for s in range(0, len(order), bs):
            sl = [items[i] for i in order[s:s + bs]]
            mx = max(len(x["pid"]) + len(x["gid"]) for x in sl)
            inp = torch.full((len(sl), mx), pad, dtype=torch.long)
            att = torch.zeros((len(sl), mx), dtype=torch.long)
            msk = torch.zeros((len(sl), mx), dtype=torch.float)
            for j, x in enumerate(sl):
                n = len(x["pid"]) + len(x["gid"])
                inp[j, :n] = torch.tensor(x["pid"] + x["gid"])
                att[j, :n] = 1
                msk[j, len(x["pid"]):n] = 1
            self.batches.append((
                inp.to(device), att.to(device), msk.to(device),
                torch.tensor([x["ntok"] for x in sl],
                             dtype=torch.float, device=device),
                np.asarray(order[s:s + bs]),
            ))


# 한 번에 GPU 에 띄울 logits 의 상한. Llama-3.2 는 어휘가 128256 으로
# TinyLlama(32000)의 4 배라, eval_bs=64 로 log_softmax 를 뜨면 10.5GB 가 든다.
# 학습 상태가 64GB 를 잡고 있는 3B 에서는 그 자리에서 OOM 이다.
LOGIT_BUDGET = 1.0e9          # 바이트


def _chunk(es: "EvalSet", vocab: int, tlen: int) -> int:
    """logits 가 LOGIT_BUDGET 을 넘지 않는 행 수."""
    per = max(tlen * vocab * 4, 1)
    return max(1, min(es.bs, int(LOGIT_BUDGET // per)))


@torch.inference_mode()
def _gold_logp(model, inp, att, msk, ch: int):
    """gold 토큰의 log-probability. **log_softmax 를 만들지 않는다.**

    logp = z[target] - logsumexp(z) 로 두면 (B, T, V) 짜리 출력 텐서가 한 번
    더 뜨지 않는다. 그리고 행을 ch 개씩 끊어 forward 해서 logits 자체의 상주
    량도 묶는다.
    """
    rows = []
    for s0 in range(0, inp.shape[0], ch):
        s1 = min(s0 + ch, inp.shape[0])
        z = model(input_ids=inp[s0:s1],
                  attention_mask=att[s0:s1]).logits[:, :-1]
        g = z.gather(-1, inp[s0:s1, 1:].unsqueeze(-1)).squeeze(-1)
        rows.append((g - torch.logsumexp(z, -1)).float())
        del z, g
    return torch.cat(rows, 0)


@torch.inference_mode()
def token_avg_logp(model, es: EvalSet) -> np.ndarray:
    """eq:mean_log_probability. 질의별 토큰 평균 log-probability."""
    out = np.empty(es.n)
    ch = None
    for inp, att, msk, ntk, idx in es.batches:
        if ch is None:
            v = int(model(input_ids=inp[:1, :2],
                          attention_mask=att[:1, :2]).logits.shape[-1])
            ch = _chunk(es, v, inp.shape[1])
        lp = _gold_logp(model, inp, att, msk, ch)
        out[idx] = ((lp * msk[:, 1:]).sum(1) / ntk).cpu().numpy()
        del lp
    return out


def sim(model, es: EvalSet) -> np.ndarray:
    """eq:sim. 질의별 (0,1] 행동적 유사성."""
    return np.exp(token_avg_logp(model, es))


def avg_bf(model, es: EvalSet) -> float:
    """eq:single_fidelity."""
    return float(np.mean(sim(model, es)))


def loss(model, es: EvalSet, w: Optional[np.ndarray] = None) -> float:
    """eq:weighted_loss. w 가 None 이면 omega = 1."""
    lp = token_avg_logp(model, es)
    if w is None:
        return -float(np.mean(lp))
    w = np.asarray(w, dtype=np.float64)
    return -float(np.sum(w * lp) / np.sum(w))


def soft_weight(sims: np.ndarray, beta: float) -> np.ndarray:
    """eq:soft_weight. sims: (K, N) -> (K, N), 질의마다 K 에 대해 정규화."""
    z = sims / max(beta, 1e-12)
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def dependency(values: Sequence[float]) -> float:
    """eq:single_dependency. 주어진 모델 집합의 avgBF 분산."""
    return float(np.var(np.asarray(values, dtype=np.float64)))


def paired_bootstrap(x: Sequence[float], y: Sequence[float],
                     n: int = 4000, seed: int = 1) -> Tuple[float, float, float]:
    """같은 질의를 함께 재표집한다. 고정 checkpoint 의 표본 불확실성."""
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return (float(d.mean()), float(np.percentile(s, 2.5)),
            float(np.percentile(s, 97.5)))


def verdict(lo: float, hi: float) -> str:
    return "양수" if lo > 0 else ("음수" if hi < 0 else "불확실")


@torch.inference_mode()
def token_probs(model, es: EvalSet) -> List[np.ndarray]:
    """질의별 gold 토큰 확률 벡터. 출력 앙상블 baseline 용."""
    out: List[Optional[np.ndarray]] = [None] * es.n
    ch = None
    for inp, att, msk, ntk, idx in es.batches:
        if ch is None:
            v = int(model(input_ids=inp[:1, :2],
                          attention_mask=att[:1, :2]).logits.shape[-1])
            ch = _chunk(es, v, inp.shape[1])
        lp = _gold_logp(model, inp, att, msk, ch)
        m = msk[:, 1:] > 0
        for j, i in enumerate(idx):
            out[i] = lp[j][m[j]].exp().cpu().numpy()
        del lp
    return out  # type: ignore


def ensemble_sim(prob_sets: List[List[np.ndarray]]) -> np.ndarray:
    """K 개 모델의 토큰 확률을 평균한 출력 앙상블의 eq:sim."""
    n = len(prob_sets[0])
    out = np.empty(n)
    for i in range(n):
        p = np.mean([ps[i] for ps in prob_sets], axis=0).clip(1e-12)
        out[i] = float(np.exp(np.log(p).mean()))
    return out
