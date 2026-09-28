# -*- coding: utf-8 -*-
"""조각 만들기. 조각이 IID 면 병합할 것이 없다.

지난 실행이 실패한 이유가 여기다. `disjoint` 는 무작위 순열을 잘라 쓰므로
조각 16 개가 **같은 분포**에서 나온다. 같은 분포의 조각으로 학습한 모델들은
잡음으로 다르고 능력으로 다르지 않다. 그래서 `local_0` 이 154 칸 중 77 칸을
점유하고 생성에서도 모든 병합을 이겼다. 지배하는 구성원이 있으면 어떤 결합
규칙도 그것을 못 이긴다. 기전의 실패가 아니라 전제의 부재다.

"Local 1 은 문제 1 을, Local 2 는 문제 2 를 맞춘다" 를 시험하려면 조각이
실제로 서로 다른 영역이어야 한다. 방법은 셋이다.

    iid       무작위. 상보성 0. 대조군으로만 쓴다 (= 지난 실행)
    length    원문 길이로 층화. 조각마다 문장 길이 대역이 다르다
    cluster   원문의 문자 n-gram 으로 군집. 조각마다 어휘/주제가 다르다

세 가지 모두 **조각 크기를 똑같이** 유지한다. 크기가 달라지면 `union_g` 의
데이터량이 흔들려 종속성 대조가 깨진다.

평가 집합(sel/check/test)은 건드리지 않는다. 전체 혼합 분포 그대로다. 그래서
어느 Local 도 평가 전체를 덮지 못하고, 덮으려면 합쳐야 한다. 이 비대칭이
병합에 이길 여지를 주는 유일한 구조다.

해시는 `zlib.crc32` 를 쓴다. `hash()` 는 프로세스마다 소금이 달라 같은 설정을
두 번 돌리면 조각이 바뀌고 `shard_hash` 관문이 헛돌게 된다.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Sequence

import numpy as np

NGRAM = 4
DIM = 1 << 14


def _feat(texts: Sequence[str], dim: int = DIM, n: int = NGRAM) -> np.ndarray:
    """문자 n-gram 해싱 + idf 가중 + L2 정규화. sklearn 없이 TF-IDF 대용."""
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = " " + str(t).lower() + " "
        for j in range(max(1, len(s) - n + 1)):
            X[i, zlib.crc32(s[j:j + n].encode("utf-8")) % dim] += 1.0
    df = (X > 0).sum(0)
    idf = np.log(len(texts) / np.maximum(df, 1.0)).astype(np.float32)
    X *= idf
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-8)


def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 60) -> np.ndarray:
    """구면 k-means. 중심은 L2 정규화되어 내적이 코사인이 된다."""
    rs = np.random.RandomState(seed)
    C = X[rs.choice(len(X), k, replace=False)].copy()
    prev = None
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        if prev is not None and np.array_equal(a, prev):
            break
        prev = a
        for j in range(k):
            m = X[a == j]
            v = m.sum(0) if len(m) else X[rs.randint(len(X))]
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                C[j] = v / nv
    return C


def _balanced(S: np.ndarray, per: int) -> np.ndarray:
    """용량 제한 배정. 확신(1 등과 2 등의 차) 큰 것부터 원하는 군집에 넣는다.

    k-means 군집은 크기가 들쭉날쭉하다. 그대로 쓰면 조각 크기가 달라져
    `union_g` 비교가 깨지므로 군집마다 정확히 `per` 개씩 채운다.
    """
    n, k = S.shape
    top2 = np.partition(S, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    order = np.argsort(-margin, kind="stable")
    pref = np.argsort(-S, axis=1)
    cap = np.full(k, per, dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    for i in order:
        for j in pref[i]:
            if cap[j] > 0:
                cap[j] -= 1
                out[i] = j
                break
    return out


def make_shards(mode: str, train: Sequence[dict], k: int, seed: int,
                log=print) -> Dict[int, List[int]]:
    """조각 k 개. 각 조각은 `len(train)//k` 개로 크기가 같다."""
    n = len(train)
    per = n // k
    use = per * k
    if mode == "iid" or mode == "disjoint":
        # train 은 이미 무작위 순열이므로 앞에서부터 자르면 IID 다.
        g = {j: list(range(j * per, (j + 1) * per)) for j in range(k)}
    elif mode == "bootstrap":
        g = {j: np.random.RandomState(seed + 1 + j).permutation(n)[:per].tolist()
             for j in range(k)}
    elif mode == "length":
        key = np.array([len(x["pid"]) for x in train], dtype=np.int64)
        order = np.argsort(key, kind="stable")[:use]
        g = {j: sorted(order[j * per:(j + 1) * per].tolist()) for j in range(k)}
    elif mode == "cluster":
        X = _feat([x["src"] for x in train])
        C = _kmeans(X, k, seed)
        a = _balanced(X @ C.T, per)
        g = {j: sorted(np.flatnonzero(a == j).tolist()) for j in range(k)}
    else:
        raise SystemExit(f"shard 를 모른다: {mode}. "
                         f"iid | bootstrap | length | cluster")
    sizes = sorted({len(v) for v in g.values()})
    if mode != "bootstrap" and sizes != [per]:
        raise SystemExit(f"조각 크기가 다르다 {sizes}. per={per}")
    log(f"[조각] {mode}   {k} x {per}")
    return g


# --------------------------------------------------------------- 진단
def skew(train: Sequence[dict], shard: Dict[str, List[int]]) -> Dict[str, float]:
    """조각이 실제로 얼마나 치우쳤나. IID 면 셋 다 0 근처다.

      len_eta2   원문 길이의 조각간 분산 / 전체 분산. 0=IID, 1=완전 층화
      vocab_jac  조각별 상위 어휘 200 개의 평균 쌍별 Jaccard. IID 면 높다
      feat_cos   조각 중심 벡터의 평균 쌍별 코사인. IID 면 1 에 가깝다

    `len_eta2` 가 0.05 미만이고 `feat_cos` 가 0.9 를 넘으면 조각이 사실상
    IID 다. 그 상태에서 병합이 최고 단일을 이기는 일은 없다.
    """
    names = list(shard)
    ln = np.array([len(x["pid"]) for x in train], dtype=np.float64)
    parts = [ln[shard[nm]] for nm in names]
    gm = ln.mean()
    sst = float(((ln - gm) ** 2).sum())
    ssb = float(sum(len(p) * (p.mean() - gm) ** 2 for p in parts))
    eta2 = ssb / sst if sst > 0 else 0.0

    tops = []
    for nm in names:
        c: Dict[str, int] = {}
        for i in shard[nm]:
            for w in str(train[i]["src"]).lower().split():
                c[w] = c.get(w, 0) + 1
        tops.append({w for w, _ in sorted(c.items(), key=lambda kv: -kv[1])[:200]})
    jac, cnt = 0.0, 0
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            u = len(tops[i] | tops[j])
            jac += len(tops[i] & tops[j]) / u if u else 0.0
            cnt += 1
    X = _feat([x["src"] for x in train])
    Cs = []
    for nm in names:
        v = X[shard[nm]].sum(0)
        Cs.append(v / max(float(np.linalg.norm(v)), 1e-8))
    M = np.stack(Cs) @ np.stack(Cs).T
    iu = np.triu_indices(len(Cs), 1)
    return {"len_eta2": eta2,
            "vocab_jaccard": jac / cnt if cnt else 0.0,
            "feat_cosine": float(M[iu].mean()),
            "len_mean": [float(p.mean()) for p in parts]}


def report(sk: Dict[str, float], log=print) -> bool:
    """조각이 병합을 시험할 만한가. 거짓이면 그대로 돌려도 결과가 뻔하다.

    축을 나눠서 말한다. `length` 는 길이만 갈라 `len_eta2` 만 오르고 어휘는
    그대로이므로 `feat_cosine` 이 높게 남는다. `cluster` 는 그 반대다. 어느
    축도 갈리지 않았을 때만 거짓이다.
    """
    by_len = sk["len_eta2"] >= 0.05
    by_voc = sk["feat_cosine"] <= 0.90
    log(f"  조각 치우침     len_eta2 {sk['len_eta2']:.3f}   "
        f"vocab_jaccard {sk['vocab_jaccard']:.3f}   "
        f"feat_cosine {sk['feat_cosine']:.3f}")
    log(f"  조각별 평균 길이 " + " ".join(f"{v:.0f}" for v in sk["len_mean"]))
    log(f"  갈린 축         "
        + (" ".join(x for x, b in (("길이", by_len), ("어휘", by_voc)) if b)
           or "없음"))
    if not (by_len or by_voc):
        log("  *** 조각이 사실상 IID 다. 어느 Local 하나가 지배하고 병합은")
        log("      그것을 못 이긴다. --shard cluster 또는 length 를 쓸 것.")
    return by_len or by_voc
