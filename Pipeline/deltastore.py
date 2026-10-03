# -*- coding: utf-8 -*-
"""Δw 저장소. 칸 단위로 디스크에서 읽는다.

지금까지는 `load_deltas` 가 arm 전부의 Δw 를 CPU 에 통째로 올렸다.
TinyLlama-1.1B K=16 에서 21 arm x 0.97B x 4B = 81GB 로 Colab 고RAM(83GB)에
간신히 들어갔고, 3B 로 올리면 237GB 라 시작조차 못 한다.

pipeline 의 접근은 전부 **칸 바깥 고리**다. `apply` 도 `measure` 도
`pairwise_cosine` 도 key 하나를 잡고 arm 을 돈다. 그래서 캐시는 한 칸 분이면
충분하다. Llama-3.2-3B 의 가장 큰 칸이 3072x8192 = 100MB 이고 21 arm 이면
2.1GB 다.

저장 형식
    <ckpt>/<arm>.safetensors     Δw = ckpt - BASE.  키는 "{layer}.{role}"
    <ckpt>/norms.json            칸별 Frobenius norm. 매번 다시 훑지 않는다

**절대 가중치가 아니라 Δw 를 저장한다.** 복원은 BASE + Δw 이고 BASE 는 이미
`WeightSpace.base` 에 GPU 로 있다. Δw 는 BASE 대비 1e-2 수준이라 fp16 으로
충분하고(fp16 최소 정규수 6.1e-5, 절대오차는 값의 5e-4), 디스크가 절반이 된다.
정밀도가 걱정되면 `delta_dtype="float32"` 로 둔다. 어느 쪽이든 실행 2 의
`verify_restore` 가 질의별 log-probability 로 1e-4 관문을 건다.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import Config
from .modeling import Key, WeightSpace, ckpt_path

DT = {"float16": torch.float16, "bfloat16": torch.bfloat16,
      "float32": torch.float32}


def skey(key: Key) -> str:
    return f"{key[0]}.{key[1]}"


def st_path(cfg: Config, name: str) -> str:
    return os.path.join(cfg.ckpt_dir, f"{name}.safetensors")


def norms_path(cfg: Config) -> str:
    return os.path.join(cfg.ckpt_dir, "norms.json")


# ------------------------------------------------------------------ 저장
def save_delta(ws: WeightSpace, cfg: Config, name: str,
               state: Optional[Dict[Key, torch.Tensor]] = None,
               delta: Optional[Dict[Key, torch.Tensor]] = None
               ) -> Dict[str, float]:
    """현재 모델(또는 주어진 state / delta)에서 Δw 를 뽑아 저장한다.

    학습 직후에 부르면 텐서가 이미 메모리에 있으므로 norm 계산이 공짜다.
    `delta` 는 이미 Δw 인 dict 다. 학습 중 최적 시점을 CPU 에 떠 두었다가
    그것을 저장할 때 쓴다.
    """
    dt = DT[cfg.delta_dtype]
    out, nrm = {}, {}
    for k in ws.keys:
        if delta is not None:
            d = delta[k].float().contiguous()
        else:
            w = (state[k].to(ws.device).float() if state is not None
                 else ws.lin[k].weight.detach().float())
            d = (w - ws.base[k].to(w.device)).contiguous()
        nrm[skey(k)] = float(d.norm())
        out[skey(k)] = d.to(dt).cpu()
    tmp = st_path(cfg, name) + ".tmp"
    save_file(out, tmp)
    os.replace(tmp, st_path(cfg, name))
    _merge_norms(cfg, {name: nrm})
    return nrm


def _merge_norms(cfg: Config, add: Dict[str, Dict[str, float]]) -> None:
    p = norms_path(cfg)
    cur = {}
    if os.path.exists(p):
        try:
            cur = json.load(open(p, encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(add)
    tmp = p + ".tmp"
    json.dump(cur, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, p)


def convert_pt(ws: WeightSpace, cfg: Config, name: str, log=print) -> bool:
    """예전 `.pt`(절대 가중치)를 `.safetensors`(Δw)로 한 번 바꾼다.

    1.1B 로 이미 학습해 둔 checkpoint 를 버리지 않기 위한 다리다.
    """
    old = ckpt_path(cfg, name)
    if os.path.exists(st_path(cfg, name)) or not os.path.exists(old):
        return False
    st = torch.load(old, map_location="cpu")
    save_delta(ws, cfg, name, state=st)
    del st
    log(f"  {name} .pt -> .safetensors 변환")
    return True


# ------------------------------------------------------------------ 읽기
class DeltaStore:
    """arm 별 Δw 를 디스크에 두고 칸 단위로 GPU 에 올린다.

    `raw(name, key)` 는 **GPU 의 fp32 텐서**를 돌려준다. 같은 key 를 계속
    물으면 캐시에서 나가고, key 가 바뀌면 이전 칸을 통째로 버린다.
    """

    def __init__(self, ws: WeightSpace, cfg: Config, names: Sequence[str],
                 log=print):
        self.ws, self.cfg = ws, cfg
        self.names = list(names)
        self.keys = list(ws.keys)
        self.device = ws.device
        for n in self.names:
            convert_pt(ws, cfg, n, log=log)
            if not os.path.exists(st_path(cfg, n)):
                raise SystemExit(f"{n}: checkpoint 가 없다 ({st_path(cfg, n)})")
        self._h = {n: safe_open(st_path(cfg, n), framework="pt", device="cpu")
                   for n in self.names}
        self.norm = self._load_norms(log)
        self._ck: Optional[Key] = None
        self._buf: Dict[str, torch.Tensor] = {}
        big = max(int(np.prod(ws.base[k].shape)) for k in self.keys)
        log(f"[Δw] {len(self.names)} arm  디스크 상주  "
            f"칸 캐시 최대 {big*4*len(self.names)/1e9:.2f}GB  "
            f"저장 {cfg.delta_dtype}")

    def _load_norms(self, log) -> Dict[str, Dict[Key, float]]:
        p = norms_path(self.cfg)
        raw = {}
        if os.path.exists(p):
            try:
                raw = json.load(open(p, encoding="utf-8"))
            except Exception:
                raw = {}
        need = [n for n in self.names
                if n not in raw or len(raw[n]) != len(self.keys)]
        if need:
            log(f"[Δw] norm 계산 {len(need)} arm (한 번만 한다)")
            for n in need:
                raw[n] = {skey(k): float(self._h[n].get_tensor(skey(k)).float().norm())
                          for k in self.keys}
            _merge_norms(self.cfg, {n: raw[n] for n in need})
        return {n: {k: raw[n][skey(k)] for k in self.keys} for n in self.names}

    # ---- 칸 하나 분 캐시. key 가 바뀌면 통째로 버린다.
    def raw(self, name: str, key: Key) -> torch.Tensor:
        if self._ck != key:
            self._ck, self._buf = key, {}
        t = self._buf.get(name)
        if t is None:
            t = self._h[name].get_tensor(skey(key)).to(
                self.device, non_blocking=True).float()
            self._buf[name] = t
        return t

    def state(self, name: str, key: Key) -> torch.Tensor:
        """복원용 절대 가중치 = BASE + Δw."""
        r = self.raw(name, key)
        return self.ws.base[key].to(r.device) + r

    def rel_delta(self, name: str) -> float:
        """‖Δw‖ / ‖w_BASE‖. 학습이 얼마나 움직였는지."""
        num = math.sqrt(sum(self.norm[name][k] ** 2 for k in self.keys))
        den = math.sqrt(sum(float(self.ws.base[k].norm()) ** 2
                            for k in self.keys))
        return num / max(den, 1e-30)

    def finite(self, name: str) -> bool:
        return all(torch.isfinite(self.raw(name, k)).all().item()
                   for k in self.keys)

    def release(self) -> None:
        self._ck, self._buf = None, {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.release()
        self._h = {}

    def __contains__(self, name: str) -> bool:
        return name in self.norm

    def __getitem__(self, name: str) -> "_Arm":
        return _Arm(self, name)


class _Arm:
    """`store[name][key]` 를 쓰던 예전 코드를 위한 얇은 뷰."""

    __slots__ = ("s", "n")

    def __init__(self, store: DeltaStore, name: str):
        self.s, self.n = store, name

    def __getitem__(self, key: Key) -> torch.Tensor:
        return self.s.raw(self.n, key)
