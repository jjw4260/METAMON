# -*- coding: utf-8 -*-
"""LoRD (Liang et al., "Yes, My LoRD").

원본 구현(LoRD-MEA)을 그대로 따른다.

    rlhf_train.log_clip(t, eps=0.2) -> clamp(t, log(1-eps), log(1+eps))
        토큰 단위로 적용한 뒤 mask 로 합산한다.  (이 순서가 중요하다.
        시퀀스 합에 clip 을 걸면 항상 포화해 L_reg 가 죽고, 논문 Table 6 의
        "w.o. L_reg -> NC(not converged)" 와 같은 상태가 된다.)

    train_pod2.py
        p  = exp( sum(logp * mask) / sum(mask) )        # (0,1] 정규화 확률
        delta = p_now - p_prev
        if p12 > p11: swap                              # 11 이 양성이 되도록
        if max(p11,p12) < tau1 and delta11 < tau_delta: # cold start
            y+ <- y_vic
        if min(p11,p12) < tau2: period_break

    lord_train.train_one_period  (black-box: use_vic_logits = 0)
        loss_vic    = sum(log_clip(logp_pos - old_logp_pos) * mask_pos)
        loss_reward = sum(logp_pos * mask_pos) - sum(logp_neg * mask_neg)
        loss        = -(loss_vic + loss_reward)

논문 Eq.11 의 sigmoid 는 옵션이다(ablation 상 필수 아님).
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
    p_pos: float = 0.0
    p_neg: float = 0.0
    same: float = 0.0

    def as_dict(self) -> dict:
        return {"swap": self.swap, "cold": self.cold,
                "clip_sat": round(self.clip_sat, 4),
                "p_pos": round(self.p_pos, 4), "p_neg": round(self.p_neg, 4),
                "same": round(self.same, 4)}


def lord_loss(lp_pos, m_pos, lp_neg, m_neg, old_pos, old_neg,
              cfg: Config) -> Tuple[torch.Tensor, LordStats]:
    """lord_train.train_one_period (black-box 경로)."""
    d_pos = log_clip(lp_pos - old_pos, cfg.log_clip_eps)
    d_neg = log_clip(lp_neg - old_neg, cfg.log_clip_eps)

    loss_vic = (d_pos * m_pos).sum(1)
    loss_reward = (lp_pos * m_pos).sum(1) - (lp_neg * m_neg).sum(1)
    raw = -(loss_vic + loss_reward)
    out = torch.sigmoid(raw) if cfg.use_sigmoid else raw

    with torch.no_grad():
        lo, hi = math.log(1 - cfg.log_clip_eps), math.log(1 + cfg.log_clip_eps)
        r = (lp_pos - old_pos)
        sat = (((r <= lo) | (r >= hi)).float() * m_pos).sum() / m_pos.sum().clamp(min=1)
    return out, LordStats(clip_sat=float(sat))


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

    try:
        for t in range(cfg.periods):
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

            old_pos, old_neg = lp_p.detach().clone(), lp_n.detach().clone()
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
                L, st = lord_loss(lp_p2, m_p2, lp_n2, m_n2,
                                  old_pos[sl][:, :lp_p2.shape[1]],
                                  old_neg[sl][:, :lp_n2.shape[1]], cfg)
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
                jl.write(json.dumps({"period": t, "update": upd,
                                     "loss": float(Lm.detach()), "gnorm": gn,
                                     "p_pos": cp, "p_neg": cn,
                                     "clip_sat": st.clip_sat}) + "\n")
                if upd % 16 == 0:
                    jl.flush()
                    log(f"    u{upd:4d} loss {float(Lm.detach()):9.3f} "
                        f"gnorm {gn:8.2f} clip_sat {st.clip_sat:.2f} "
                        f"p+ {cp:.3f} p- {cn:.3f}")
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
