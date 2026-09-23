# -*- coding: utf-8 -*-
"""질의 구성과 분할, 그리고 Target 응답 붙이기.

  1. build_queries   질의만 만든다. 프롬프트 길이로만 거른다. 비용 0.
  2. attach          Target 응답을 붙인다. 응답이 길면 자른다(버리지 않는다).
                     버리면 그 질의에 쓴 돈이 날아가고 분할 크기도 흔들린다.
  3. Splits          train / sel / check / test 와 Local 조각.

프롬프트 형식은 LoRD-MEA 와 같다.
    local  : "Instruction: {pp} User: {src} Assistant: "
    target : system="Instruction: {pp}", user="{src}"
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset

from .config import Config
from .target import TASK_PROMPT

Item = Dict[str, object]


def instruction(cfg: Config) -> str:
    return cfg.instruction or TASK_PROMPT.get(cfg.subset, "")


def local_prompt(cfg: Config, src: str) -> str:
    return f"Instruction: {instruction(cfg)} User: {src} Assistant: "


def build_queries(cfg: Config, tok) -> List[Item]:
    """Target 에 보낼 질의 목록. 아직 응답은 없다."""
    ds = load_dataset(cfg.dataset, cfg.subset, split="train")
    n = min(cfg.pool, len(ds))
    rows = [ds[i]["translation"] for i in range(n)]

    srcs = [r[cfg.src_key] for r in rows]
    refs = [r[cfg.tgt_key] for r in rows]
    prompts = [local_prompt(cfg, s) for s in srcs]
    enc = tok(prompts, add_special_tokens=True)["input_ids"]

    seen, items = set(), []
    for s, r, p, ip in zip(srcs, refs, prompts, enc):
        if p in seen or len(ip) > cfg.max_prompt_tok:
            continue
        seen.add(p)
        items.append({"src": s, "ref": r, "prompt": p, "pid": ip})

    order = np.random.RandomState(cfg.seed).permutation(len(items))
    items = [items[i] for i in order]
    need = cfg.n_train + cfg.n_sel + cfg.n_check + cfg.n_test
    if len(items) < need:
        raise SystemExit(f"질의 부족 {len(items)}/{need}. pool 을 늘릴 것.")
    return items[:need]


def attach(items: Sequence[Item], responses: Sequence[str], cfg: Config, tok,
           log=print) -> List[Item]:
    """Target 응답을 gold 로 붙이고 토크나이즈한다."""
    out, cut, empty = [], 0, 0
    for it, resp in zip(items, responses):
        g = " " + resp.strip()
        gid = tok(g, add_special_tokens=False)["input_ids"]
        room = cfg.max_tok - len(it["pid"])
        if room < 1:
            raise SystemExit("max_tok 이 프롬프트보다 짧다. 설정을 확인할 것.")
        if len(gid) > room:
            gid = gid[:room]
            cut += 1
        if len(gid) == 0:
            gid = [tok.eos_token_id]
            empty += 1
        d = dict(it)
        d.update({"gold": g, "gid": gid, "ntok": len(gid)})
        out.append(d)
    log(f"[응답] {len(out)}개 부착.  길이 초과로 자름 {cut}  빈 응답 {empty}")
    return out


def save_dataset(path: str, items: Sequence[Item], meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta,
                   "items": [{"src": x["src"], "ref": x["ref"],
                              "prompt": x["prompt"], "gold": x["gold"]}
                             for x in items]}, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_dataset_file(path: str, cfg: Config, tok, log=print) -> List[Item]:
    raw = json.load(open(path, encoding="utf-8"))
    items = []
    for x in raw["items"]:
        items.append({"src": x["src"], "ref": x["ref"], "prompt": x["prompt"],
                      "pid": tok(x["prompt"], add_special_tokens=True)["input_ids"]})
    return attach(items, [x["gold"] for x in raw["items"]], cfg, tok, log)


class Splits:
    """train / sel / check / test 와 Local 조각."""

    def __init__(self, cfg: Config, items: List[Item]):
        a = cfg.n_train
        b = a + cfg.n_sel
        c = b + cfg.n_check
        self.train = items[:a]
        self.sel = items[a:b]          # 기여도 측정
        self.check = items[b:c]        # 배율 선택
        self.test = items[c:]          # 최종 보고
        self.all = items

        per = cfg.n_train // cfg.k
        if cfg.shard == "disjoint":
            self.shard = {cfg.local_names[i]: list(range(i * per, (i + 1) * per))
                          for i in range(cfg.k)}
        else:
            self.shard = {
                cfg.local_names[i]: np.random.RandomState(cfg.seed + 1 + i)
                .permutation(cfg.n_train)[: cfg.n_train // 2].tolist()
                for i in range(cfg.k)
            }

    def query_hash(self) -> str:
        h = hashlib.sha256()
        for it in self.all:
            h.update(it["prompt"].encode())
            h.update(it["gold"].encode())
        return h.hexdigest()[:16]

    def shard_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.shard, sort_keys=True).encode()).hexdigest()[:16]

    def overlap(self) -> Dict[str, int]:
        key = {(x["prompt"], x["gold"]) for x in self.train}
        return {
            name: sum(1 for x in part if (x["prompt"], x["gold"]) in key)
            for name, part in (("sel", self.sel), ("check", self.check),
                               ("test", self.test))
        }

    def assert_clean(self) -> None:
        ov = self.overlap()
        if sum(ov.values()) != 0:
            raise SystemExit(f"학습 질의가 평가에 섞였다: {ov}")

    def summary(self) -> str:
        return (f"train {len(self.train)} (조각 {len(next(iter(self.shard.values())))}"
                f" x {len(self.shard)}) / sel {len(self.sel)} / "
                f"check {len(self.check)} / test {len(self.test)}   "
                f"중복 {self.overlap()}  질의해시 {self.query_hash()}")
