# -*- coding: utf-8 -*-
"""LoRD-VI (Liang et al., "Yes, My LoRD"). 논문이 보고하는 방법이다.

`lord_train.py:1109` 에서 `LoRD-VI -> from train_pod2 import train` 이다.
Table 1 의 수치는 이 경로에서 나온다.

손실 (Eq.10 / Eq.11, black-box)
    L_obj = log P(y-|x) - log P(y+|x)
    L_reg = clip( log P(y-|x) - log P(y_vic|x) )
    L     = sigma( 2 * [ (1-lambda1) * L_obj + lambda1 * L_reg ] )

    lambda1 = 0.5 이면 Eq.11 과 같다. 세 항 모두 **Target 응답 y_vic 이
    목적함수에 직접 들어간다**. 이것이 추출 알고리즘인 이유다.

    log P(y|x) 는 **응답 토큰만**의 평균 log-probability 로 쓴다.
    합을 쓰면 세 후보의 길이가 달라 손실이 길이에 지배된다. tau1 이
    비교하는 p = exp(평균 logp) 와도 같은 양이 된다.

표집과 제어 (train_pod2.py:576-630)
    p  = exp( sum(logp * mask) / sum(mask) )        # (0,1] 정규화 확률
    if p12 > p11: swap                              # 11 이 양성이 되도록
    if max(p11,p12) < tau1 and delta11 < tau_delta: # cold start
        y+ <- y_vic
    if min(p11,p12) < tau2: period_break

공개 구현과 논문이 갈리는 곳 (lord_variant 로 고른다)
    "code"   train_pod2.py:963 의 `loss = los2 + loss11 + 2*loss12` 그대로.
             L_reg 는 있고 clip 만 없다. 공개 구현은 clip 항(term3)을 :941
             에서 계산만 하고 손실에 넣지 않는다. Table 1 을 낸 경로이므로
             이것이 기본값이다.
    "paper"  Eq.10 을 글자 그대로. clip 을 건다.
             clip 범위는 [-0.223, +0.182] 인데 log P(y-) - log P(y_vic) 는
             초반에 이를 크게 벗어나므로 L_reg 가 포화해 기울기가 0 이 된다.
             그 상태가 곧 Table 6 의 "w.o. L_reg -> NC" 다. 쓰려면 로그의
             clip_sat 을 보고 판단할 것.

옮기지 않은 원본의 특이점
    - 원본은 `torch.mean(logits2_cons)` 로 mask 를 쓰지 않아 프롬프트 토큰이
      평균에 섞인다(:986-988 의 print 에서만 mask 를 쓴다). 여기서는 응답
      토큰만 쓴다.
    - 원본은 직전 확률을 `sum(exp(logp)*mask)/sum(mask)` 로, 현재 확률을
      `exp(sum(logp*mask)/sum(mask))` 로 계산해 서로 다른 양을 빼서 delta 를
      만든다(:581 vs :638). 여기서는 둘 다 후자로 통일한다.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .config import Config
from .modeling import WeightSpace, ckpt_path, save_atomic


# ---------------------------------------------------------------- 기본 연산
def log_clip(t: torch.Tensor, eps: float) -> torch.Tensor:
    """rlhf_train.log_clip. 토큰 단위 비율 clip 의 로그 형태."""
    lo = math.log(1.0 - eps)
    hi = math.log(1.0 + eps)
    return t.clamp(lo, hi)


def pack(pids: Sequence[Sequence[int]], gids: Sequence[Sequence[int]],
         pad: int, device: str):
    mx = max(len(p) + len(g) for p, g in zip(pids, gids))
    inp = torch.full((len(pids), mx), pad, dtype=torch.long)
    att = torch.zeros((len(pids), mx), dtype=torch.long)
    msk = torch.zeros((len(pids), mx), dtype=torch.float)
    for j, (p, g) in enumerate(zip(pids, gids)):
        n = len(p) + len(g)
        inp[j, :n] = torch.tensor(list(p) + list(g))
        att[j, :n] = 1
        msk[j, len(p):n] = 1
    return inp.to(device), att.to(device), msk.to(device)


def token_logp(model, pids, gids, pad: int, device: str
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """토큰별 log-probability 와 mask. (bs, L-1) 두 개."""
    inp, att, msk = pack(pids, gids, pad, device)
    lg = model(input_ids=inp, attention_mask=att).logits.float()
    lp = F.log_softmax(lg[:, :-1], -1).gather(
        -1, inp[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp, msk[:, 1:]


def norm_prob(lp: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """train_pod2 의 p = exp( sum(logp*mask) / sum(mask) )."""
    return torch.exp((lp * m).sum(1) / m.sum(1).clamp(min=1))


def truncate(ids: List[int], eos: int, pad: int) -> List[int]:
    """첫 EOS 를 포함해 자른다. 뒤 padding 만 버린다."""
    if eos in ids:
        return ids[: ids.index(eos) + 1]
    while ids and ids[-1] == pad:
        ids.pop()
    return ids


@torch.inference_mode()
def sample(ws: WeightSpace, tok, items: Sequence[dict], cfg: Config,
           seed: int) -> List[List[int]]:
    """decoder-only 배치 생성은 왼쪽 padding 을 쓴다."""
    torch.manual_seed(seed)
    pad, eos = tok.pad_token_id, tok.eos_token_id
    ws.model.config.use_cache = True
    order = sorted(range(len(items)), key=lambda i: len(items[i]["pid"]))
    out: List[Optional[List[int]]] = [None] * len(items)
    try:
        for s in range(0, len(order), cfg.gen_bs):
            sl = order[s:s + cfg.gen_bs]
            mx = max(len(items[i]["pid"]) for i in sl)
            inp = torch.full((len(sl), mx), pad, dtype=torch.long)
            att = torch.zeros((len(sl), mx), dtype=torch.long)
            for j, i in enumerate(sl):                    # left padding
                p = items[i]["pid"]
                inp[j, mx - len(p):] = torch.tensor(p)
                att[j, mx - len(p):] = 1
            gen = ws.model.generate(
                input_ids=inp.to(ws.device), attention_mask=att.to(ws.device),
                do_sample=True, temperature=cfg.gen_temp, top_p=cfg.gen_top_p,
                max_new_tokens=cfg.gen_max_new, pad_token_id=pad)
            for j, i in enumerate(sl):
                t = truncate(gen[j, mx:].tolist(), eos, pad)
                out[i] = t if t else [eos]
    finally:
        ws.model.config.use_cache = False
    return out  # type: ignore


# ---------------------------------------------------------------- 손실
@dataclass
class LordStats:
    swap: int = 0
    cold: int = 0
    clip_sat: float = 0.0
    obj: float = 0.0
    reg: float = 0.0
    p_pos: float = 0.0
    p_neg: float = 0.0
    same: float = 0.0

    def as_dict(self) -> dict:
        return {"swap": self.swap, "cold": self.cold,
                "clip_sat": round(self.clip_sat, 4),
                "obj": round(self.obj, 4), "reg": round(self.reg, 4),
                "p_pos": round(self.p_pos, 4), "p_neg": round(self.p_neg, 4),
                "same": round(self.same, 4)}


def mean_logp(lp: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """응답 토큰만의 평균 log-probability. log p = log P(y|x) 의 길이 정규화 형."""
    return (lp * m).sum(1) / m.sum(1).clamp(min=1)


def lord_loss(lp_pos, m_pos, lp_neg, m_neg, lp_vic, m_vic,
              cfg: Config) -> Tuple[torch.Tensor, LordStats]:
    """LoRD-VI. Eq.10 / Eq.11.

        L_obj = log P(y-) - log P(y+)
        L_reg = clip( log P(y-) - log P(y_vic) )        ("paper")
                     log P(y-) - log P(y_vic)           ("code")
        L     = sigma( 2 * [ (1-l1) * L_obj + l1 * L_reg ] )
    """
    g_pos = mean_logp(lp_pos, m_pos)
    g_neg = mean_logp(lp_neg, m_neg)
    g_vic = mean_logp(lp_vic, m_vic)

    obj = g_neg - g_pos
    reg_raw = g_neg - g_vic
    reg = log_clip(reg_raw, cfg.log_clip_eps) if cfg.lord_variant == "paper" \
        else reg_raw

    l1 = cfg.lambda1
    inner = 2.0 * ((1.0 - l1) * obj + l1 * reg)
    out = torch.sigmoid(inner) if cfg.use_sigmoid else inner

    with torch.no_grad():
        lo, hi = math.log(1 - cfg.log_clip_eps), math.log(1 + cfg.log_clip_eps)
        sat = float(((reg_raw <= lo) | (reg_raw >= hi)).float().mean())
        st = LordStats(clip_sat=sat, obj=float(obj.mean()), reg=float(reg.mean()))
    return out, st


# ---------------------------------------------------------------- 학습
def train_lord(ws: WeightSpace, tok, cfg: Config, name: str,
               data: Sequence[dict], seed: int,
               health: Optional[Callable[[], float]] = None,
               log=print) -> None:
    """period 마다 재표집하고, 붕괴하면 즉시 멈춘다."""
    pad = tok.pad_token_id
    ws.reset()
    ws.trainable(True)
    opt = torch.optim.AdamW(ws.params, lr=cfg.lord_lr, foreach=True)
    jl = open(os.path.join(cfg.log_dir, f"{name}.jsonl"), "a", encoding="utf-8")
    upd = 0
    prev_p: Dict[int, Tuple[float, float]] = {}

    # periods 는 arm 의 질의 수에 맞춘다. 고정값이면 질의가 많은 arm 이
    # 자기 데이터를 다 보지 못한 채 끝나 union_g 와 fleet_g 의 비교가 깨진다.
    periods = cfg.periods or max(
        1, -(-len(data) // cfg.period_chunk) * cfg.lord_epochs)
    log(f"  [{name}] 질의 {len(data)}  period {periods} x {cfg.period_chunk}  "
        f"변형 {cfg.lord_variant}  lambda1 {cfg.lambda1}  "
        f"sigmoid {cfg.use_sigmoid}")
    sat_warned = False

    try:
        for t in range(periods):
            idx = [(t * cfg.period_chunk + i) % len(data)
                   for i in range(min(cfg.period_chunk, len(data)))]
            chunk = [data[i] for i in idx]
            pids = [x["pid"] for x in chunk]

            ws.model.eval()
            neg_c = sample(ws, tok, chunk, cfg, seed * 1000 + t * 10 + 1)
            # 첫 period 의 양성 후보는 Target(victim) 응답이다.
            pos_c = ([x["gid"] for x in chunk] if t == 0
                     else sample(ws, tok, chunk, cfg, seed * 1000 + t * 10 + 2))

            with torch.no_grad():
                lp_p, m_p = token_logp(ws.model, pids, pos_c, pad, ws.device)
                lp_n, m_n = token_logp(ws.model, pids, neg_c, pad, ws.device)
                p_pos = norm_prob(lp_p, m_p)
                p_neg = norm_prob(lp_n, m_n)

            # --- swap: 11 이 양성이 되도록 (train_pod2: p12 > p11 이면 교환)
            swap = (p_neg > p_pos)
            n_swap = int(swap.sum())
            for j in range(len(chunk)):
                if bool(swap[j]):
                    pos_c[j], neg_c[j] = neg_c[j], pos_c[j]
            if n_swap:
                with torch.no_grad():
                    lp_p, m_p = token_logp(ws.model, pids, pos_c, pad, ws.device)
                    lp_n, m_n = token_logp(ws.model, pids, neg_c, pad, ws.device)
                    p_pos = norm_prob(lp_p, m_p)
                    p_neg = norm_prob(lp_n, m_n)

            # --- cold start: max(p) < tau1 이고 증가량이 작으면 victim 응답 사용
            n_cold = 0
            for j, i in enumerate(idx):
                prev = prev_p.get(i, (0.0, 0.0))
                delta = float(p_pos[j]) - prev[0]
                if max(float(p_pos[j]), float(p_neg[j])) < cfg.tau1 \
                        and delta < cfg.tau_delta:
                    pos_c[j] = list(chunk[j]["gid"])
                    n_cold += 1
            if n_cold:
                with torch.no_grad():
                    lp_p, m_p = token_logp(ws.model, pids, pos_c, pad, ws.device)
                    p_pos = norm_prob(lp_p, m_p)

            same = float(np.mean([1.0 if a == b else 0.0
                                  for a, b in zip(pos_c, neg_c)]))
            log(f"  [{name}] p{t+1:3d}  p+ {float(p_pos.median()):.4f}  "
                f"p- {float(p_neg.median()):.4f}  swap {n_swap}  cold {n_cold}  "
                f"동일 {same:.2f}")

            if float(p_pos.median()) > 0.999 and same > 0.9:
                log(f"  [{name}] 붕괴 감지 (p+ ~ 1, 두 후보 동일). 학습 중단.")
                break

            ws.model.train()
            order = list(range(len(chunk)))
            random.Random(seed + t).shuffle(order)
            broke = False
            for s in range(0, len(order) - cfg.acc + 1, cfg.acc):
                sl = order[s:s + cfg.acc]
                bp = [pids[j] for j in sl]
                lp_p2, m_p2 = token_logp(ws.model, bp, [pos_c[j] for j in sl],
                                         pad, ws.device)
                lp_n2, m_n2 = token_logp(ws.model, bp, [neg_c[j] for j in sl],
                                         pad, ws.device)
                # y_vic. Eq.10 의 L_reg 는 Target 응답을 기준점으로 쓴다.
                lp_v2, m_v2 = token_logp(ws.model, bp,
                                         [chunk[j]["gid"] for j in sl],
                                         pad, ws.device)
                L, st = lord_loss(lp_p2, m_p2, lp_n2, m_n2, lp_v2, m_v2, cfg)
                Lm = L.mean()
                if not torch.isfinite(Lm):
                    raise SystemExit(f"{name}: 비정상 손실 {float(Lm)}")
                opt.zero_grad(set_to_none=True)
                Lm.backward()
                gn = float(clip_grad_norm_(ws.params, cfg.grad_clip))
                opt.step()
                upd += 1

                with torch.no_grad():
                    cp = float(norm_prob(lp_p2.detach(), m_p2).median())
                    cn = float(norm_prob(lp_n2.detach(), m_n2).median())
                    cv = float(norm_prob(lp_v2.detach(), m_v2).median())
                jl.write(json.dumps({"period": t, "update": upd,
                                     "loss": float(Lm.detach()), "gnorm": gn,
                                     "p_pos": cp, "p_neg": cn, "p_vic": cv,
                                     "obj": st.obj, "reg": st.reg,
                                     "clip_sat": st.clip_sat}) + "\n")
                if upd % 16 == 0:
                    jl.flush()
                    log(f"    u{upd:4d} loss {float(Lm.detach()):8.4f} "
                        f"gnorm {gn:7.2f} obj {st.obj:+7.3f} reg {st.reg:+7.3f} "
                        f"clip_sat {st.clip_sat:.2f} "
                        f"p+ {cp:.3f} p- {cn:.3f} p_vic {cv:.3f}")
                if (cfg.lord_variant == "paper" and not sat_warned
                        and upd >= 16 and st.clip_sat > 0.9):
                    sat_warned = True
                    log(f"  [{name}] *** clip_sat {st.clip_sat:.2f}. L_reg 가 "
                        f"포화해 기울기가 없다. 사실상 w.o. L_reg 상태다. "
                        f"lord_variant='code' 로 돌릴 것.")
                if min(cp, cn) < cfg.tau2:          # period break
                    broke = True
                    break

            for j, i in enumerate(idx):
                prev_p[i] = (float(p_pos[j]), float(p_neg[j]))
            if broke:
                log(f"  [{name}] p{t+1} period break (min p < tau2)")
            if health is not None and (t + 1) % 4 == 0:
                ws.model.eval()
                log(f"  [{name}] p{t+1} 상태 L(theta;1) = {health():.5f}")
                ws.model.train()
    finally:
        jl.close()
        ws.model.eval()
        ws.trainable(False)
        del opt
        torch.cuda.empty_cache()

    save_atomic(ws.snapshot(), ckpt_path(cfg, name))
