# -*- coding: utf-8 -*-
"""입력 구간 c. tokenizer 가 서로 달라도 같은 단어끼리 activation 을 짝짓는다.

두 모델에는 **같은 문자열**을 넣는다 (프롬프트 + Target 응답). 그래서 문자 offset 으로
정확히 맞출 수 있다. 구간 c 는 공백으로 나눈 단어 하나이고, 각 모델은 그 단어에
속한 토큰들의 activation 을 평균한다.

토큰을 단어에 배정하는 규칙: 토큰의 **마지막 문자**가 들어 있는 단어. 토큰 offset 이
앞 공백을 포함하든 안 하든 같은 단어로 간다. 마지막 문자가 공백이거나 특수 토큰
(offset 길이 0)이면 어느 단어에도 넣지 않는다.

지시문 앞부분("Instruction: ... User:")은 모든 질의에서 글자가 같아 activation 도
같다. 넣으면 같은 점이 수천 번 들어가 R^2 를 부풀린다. 그래서 "User:" 뒤부터 쓴다.
"""
from __future__ import annotations

import bisect
import re
from typing import List, Sequence, Tuple

WORD = re.compile(r"\S+")
MARK = "User:"


def text_of(item: dict) -> Tuple[str, int]:
    """(전체 문자열, 단어를 세기 시작할 문자 위치)."""
    p, g = item["prompt"], item["gold"]
    k = p.find(MARK)
    start = k + len(MARK) if k >= 0 else 0
    return p + g, start


def words(text: str, start: int) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD.finditer(text) if m.start() >= start]


def assign(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]]
           ) -> List[int]:
    """토큰마다 단어 번호. 없으면 -1."""
    starts = [s for s, _ in spans]
    out = []
    for s, e in offsets:
        if e <= s:
            out.append(-1)
            continue
        c = e - 1
        j = bisect.bisect_right(starts, c) - 1
        out.append(j if j >= 0 and spans[j][0] <= c < spans[j][1] else -1)
    return out


def encode(tok, texts: Sequence[str], starts: Sequence[int], max_len: int = 320):
    """tokenizer 하나로 묶음을 인코딩하고 토큰별 단어 번호를 붙인다.

    반환: input_ids (list of list), word_ids (list of list), n_words (list)
    """
    enc = tok(list(texts), add_special_tokens=True, return_offsets_mapping=True,
              truncation=True, max_length=max_len)
    ids, wids, nw = [], [], []
    bad = tot = 0
    for t, st, ii, off in zip(texts, starts, enc["input_ids"], enc["offset_mapping"]):
        sp = words(t, st)
        w = assign(off, sp)
        ids.append(list(ii))
        wids.append(w)
        nw.append(len(sp))
        tot += len(off)
        bad += sum(1 for s, e in off if e <= s)
    if tot and bad / tot > 0.5:
        raise SystemExit(
            f"{getattr(tok, 'name_or_path', tok)}: offset 길이가 0 인 토큰이 "
            f"{bad}/{tot}. 이 tokenizer 의 offset_mapping 이 고장났다. "
            f"transformers / tokenizers 를 올릴 것.")
    return ids, wids, nw


def common(wa: Sequence[int], wb: Sequence[int], n: int) -> List[int]:
    """두 모델 모두에서 토큰이 하나 이상 배정된 단어 번호(오름차순)."""
    ha = set(x for x in wa if x >= 0)
    hb = set(x for x in wb if x >= 0)
    return sorted(ha & hb & set(range(n)))
