# -*- coding: utf-8 -*-
"""eq:input_map, eq:output_map 과 R^2.

    Map = argmin  sum_{n,c} || y - Map x ||^2  +  gamma ||Map||_F^2

x, y 는 X 전체 평균을 뺀 activation 이다 (문서). activation 은 저장하지 않고
흘려 누적한다. 평균을 나중에 빼면 큰 평균 성분에서 자릿수가 날아가므로, 첫 묶음의
평균을 기준점 r 로 잡고 (x - r) 의 합과 곱을 누적한다.

    중심화 산포   C_xx = S2 - S1 S1^T / n
    교차          C_xy = sum (x-r_x)(y-r_y)^T - S1_x S1_y^T / n
    해            Map^T = (C_xx + gamma I)^{-1} C_xy       (Cholesky, fp64)

gamma 는 척도에 맞춰 gamma = c * tr(C_xx) / d_x 로 둔다. c 는 Experimental Setup 값.

R^2 는 학습에 쓰지 않은 질의에서 잰다 (문서).
    SSE = sum || (y - mu_y) - (x - mu_x) Map^T ||^2      mu 는 적합 데이터의 평균
    SST = sum || y - ybar_held ||^2
    R^2 = 1 - SSE / SST                                   (차원 전체를 모은다)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


class Var:
    """한 변수의 흐르는 1·2차 모멘트."""

    def __init__(self, d: int, second: bool, device):
        self.d, self.second, self.device = d, second, device
        self.n = 0
        self.ref: Optional[torch.Tensor] = None
        self.S1 = torch.zeros(d, device=device, dtype=torch.float32)
        self.S2 = torch.zeros(d, d, device=device, dtype=torch.float32) if second else None
        self.sq = 0.0

    def center(self, Z: torch.Tensor) -> torch.Tensor:
        if self.ref is None:
            self.ref = Z.float().mean(0)
        return Z.float() - self.ref

    def add(self, Z: torch.Tensor) -> torch.Tensor:
        Zc = self.center(Z)
        self.n += Zc.shape[0]
        self.S1 += Zc.sum(0)
        self.sq += float((Zc * Zc).sum())
        if self.second:
            self.S2.addmm_(Zc.T, Zc)
        return Zc

    @property
    def mean(self) -> torch.Tensor:
        return self.ref + self.S1 / max(self.n, 1)

    def scatter(self) -> torch.Tensor:
        S1 = self.S1.double()
        return self.S2.double() - torch.outer(S1, S1) / self.n

    def sst(self) -> float:
        return self.sq - float((self.S1.double() ** 2).sum()) / max(self.n, 1)


class Reg:
    """회귀 y ~ x 하나의 교차항."""

    def __init__(self, vx: Var, vy: Var):
        self.vx, self.vy = vx, vy
        self.Sxy = torch.zeros(vx.d, vy.d, device=vx.device, dtype=torch.float32)

    def add(self, Xc: torch.Tensor, Yc: torch.Tensor) -> None:
        """Xc, Yc 는 각각 vx.add / vy.add 가 돌려준 (기준점을 뺀) 값."""
        self.Sxy.addmm_(Xc.T, Yc)

    def cross(self) -> torch.Tensor:
        return (self.Sxy.double()
                - torch.outer(self.vx.S1.double(), self.vy.S1.double()) / self.vx.n)


def chol(vx: Var, c: float) -> torch.Tensor:
    """(C_xx + gamma I) 의 Cholesky 인자, fp64. gamma = c * tr(C_xx) / d_x."""
    C = vx.scatter()
    g = c * float(torch.diagonal(C).sum()) / C.shape[0]
    C.diagonal().add_(g)
    L, info = torch.linalg.cholesky_ex(C)
    j = g + 1e-12
    for _ in range(8):                     # fp32 누적 오차로 양정치가 깨지면 조금씩 민다
        if int(info) == 0:
            return L
        C.diagonal().add_(j * 9)
        j *= 10
        L, info = torch.linalg.cholesky_ex(C)
    if int(info) == 0:
        return L
    raise RuntimeError("Cholesky 실패: 산포 행렬이 양정치가 아니다")


def solve_group(vx: Var, regs: Sequence[Reg], cs: Sequence[float]):
    """같은 x 를 쓰는 회귀를 한 번의 분해로 푼다. {id(reg): {c: Map^T}}.

    Map^T 는 (d_x x d_y) 이고 예측은 (x - mu_x) @ Map^T 다. 분해는 c 하나씩 만들고
    바로 버린다 (d_x = 11008 이면 fp64 로 1GB).
    """
    out = {id(r): {} for r in regs}
    for c in cs:
        L = chol(vx, c)
        for r in regs:
            out[id(r)][c] = torch.cholesky_solve(r.cross(), L).float()
        del L
    return out


class Held:
    """학습에 쓰지 않은 질의에서 R^2."""

    def __init__(self, d_y: int, device):
        self.vy = Var(d_y, False, device)
        self.sse: Dict[float, float] = {}

    def add(self, X: torch.Tensor, Y: torch.Tensor, mu_x: torch.Tensor,
            mu_y: torch.Tensor, maps: Dict[float, torch.Tensor]) -> None:
        self.vy.add(Y)
        Xc = X.float() - mu_x
        Yc = Y.float() - mu_y
        for c, B in maps.items():
            r = Yc - Xc @ B
            self.sse[c] = self.sse.get(c, 0.0) + float((r * r).sum())

    def r2(self) -> Dict[float, float]:
        sst = self.vy.sst()
        return {c: 1.0 - s / sst for c, s in self.sse.items()} if sst > 0 else {}
