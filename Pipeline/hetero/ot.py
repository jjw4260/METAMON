# -*- coding: utf-8 -*-
"""Behavioral OT. 층 대응 m_{k,l,i,rho}.

eq:alignment_cost
    Cost[l, i] = 1 - <s~_theta,l , s~_k,i> / (||s~_theta,l|| ||s~_k,i|| + eps)

eq:ot_alignment
    m = argmin_{m in Pi_k}  sum m*Cost + tau * sum m log m
    Pi_k = { m >= 0 :  행 합 = 1 (L_theta 개),  열 합 = L_theta / L_k (L_k 개) }

tau > 0 은 log 영역 Sinkhorn, tau = 0 은 같은 Pi_k 의 선형계획(정확한 OT)이다.

Layer Ratio Mapping (w/o Behavioral OT) 은 **같은 Pi_k** 안에서 비용만 깊이 비율로
바꾼 단조 수송이다. 깊이 [0,1] 을 theta 는 L_theta 칸, local 은 L_k 칸으로 나눴을 때
두 칸이 겹치는 길이에 L_theta 를 곱한 값이 질량이다. 이 계획은 |l/L_theta - i/L_k|
같은 볼록 비용에 대한 최적 수송이고, 행 합 1 · 열 합 L_theta/L_k 를 정확히 만족한다.
OT 와 대조군의 차이는 "비용을 행동으로 쟀느냐 깊이로 쟀느냐" 하나뿐이다.

이 파일은 numpy / scipy 만 쓴다. 행렬이 28 x 36 정도라 GPU 가 필요 없다.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp


def alignment_cost(prof_theta: np.ndarray, prof_local: np.ndarray,
                   eps: float = 1e-12) -> np.ndarray:
    """eq:alignment_cost.

    prof_theta  (L_theta, |X|)  평균을 뺀 sensitivity profile (eq:sensitivity_profile)
    prof_local  (L_k, |X|)
    반환        (L_theta, L_k)
    """
    a = np.asarray(prof_theta, dtype=np.float64)
    b = np.asarray(prof_local, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)      # 이미 빼 왔어도 해가 없다
    b = b - b.mean(axis=1, keepdims=True)
    num = a @ b.T
    den = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    return 1.0 - num / (den + eps)


def marginals(L_theta: int, L_k: int):
    return np.ones(L_theta), np.full(L_k, L_theta / L_k)


def ratio_plan(L_theta: int, L_k: int) -> np.ndarray:
    """Layer Ratio Mapping. 같은 Pi_k 안의 깊이 단조 수송."""
    m = np.zeros((L_theta, L_k))
    for l in range(L_theta):
        a0, a1 = l / L_theta, (l + 1) / L_theta
        for i in range(L_k):
            b0, b1 = i / L_k, (i + 1) / L_k
            ov = min(a1, b1) - max(a0, b0)
            if ov > 0:
                m[l, i] = ov * L_theta
    return m


def sinkhorn(cost: np.ndarray, tau: float, iters: int = 5000,
             tol: float = 1e-10) -> np.ndarray:
    """eq:ot_alignment, tau > 0. log 영역이라 tau 가 작아도 넘치지 않는다."""
    L_t, L_k = cost.shape
    a, b = marginals(L_t, L_k)
    la, lb = np.log(a), np.log(b)
    K = -cost / tau
    f = np.zeros(L_t)
    g = np.zeros(L_k)
    for _ in range(iters):
        f = la - logsumexp(K + g[None, :], axis=1)
        g_new = lb - logsumexp(K + f[:, None], axis=0)
        if np.max(np.abs(g_new - g)) < tol:
            g = g_new
            break
        g = g_new
    f = la - logsumexp(K + g[None, :], axis=1)
    return np.exp(K + f[:, None] + g[None, :])


def exact_ot(cost: np.ndarray) -> np.ndarray:
    """eq:ot_alignment, tau = 0. 같은 Pi_k 의 선형계획."""
    L_t, L_k = cost.shape
    a, b = marginals(L_t, L_k)
    A = []
    for l in range(L_t):                        # 행 합
        r = np.zeros((L_t, L_k)); r[l, :] = 1; A.append(r.ravel())
    for i in range(L_k - 1):                    # 열 합 (하나는 중복이라 뺀다)
        r = np.zeros((L_t, L_k)); r[:, i] = 1; A.append(r.ravel())
    rhs = np.concatenate([a, b[:-1]])
    res = linprog(cost.ravel(), A_eq=np.array(A), b_eq=rhs,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"OT 선형계획 실패: {res.message}")
    return res.x.reshape(L_t, L_k)


def ot_alignment(cost: np.ndarray, tau: float) -> np.ndarray:
    return exact_ot(cost) if tau <= 0 else sinkhorn(cost, tau)


def check_plan(m: np.ndarray, tol: float = 1e-6) -> None:
    L_t, L_k = m.shape
    a, b = marginals(L_t, L_k)
    if (m < -tol).any():
        raise AssertionError("음수 질량")
    if not np.allclose(m.sum(1), a, atol=1e-5):
        raise AssertionError(f"행 합이 1 이 아니다: {m.sum(1)}")
    if not np.allclose(m.sum(0), b, atol=1e-5):
        raise AssertionError(f"열 합이 L_theta/L_k 가 아니다: {m.sum(0)}")


def active_pairs(m: np.ndarray, mass_min: float):
    """질량이 mass_min 이상인 (l, i). eq:weight_projection 의 합에서 남길 항."""
    return [(int(l), int(i), float(m[l, i]))
            for l, i in zip(*np.nonzero(m >= mass_min))]
