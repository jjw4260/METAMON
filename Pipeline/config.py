# -*- coding: utf-8 -*-
"""설정. 모든 하이퍼파라미터를 한 곳에 두고 해시로 고정한다.

LoRD 상수는 LoRD-MEA 원본(train_pod2.py, lord_train.py, rlhf_train.py)과
논문 §5.1 을 따른다.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class Config:
    # ---------------- 모델 / 데이터 ----------------
    base: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dataset: str = "wmt/wmt16"
    subset: str = "cs-en"
    src_key: str = "cs"
    tgt_key: str = "en"
    # 빈 문자열이면 target.TASK_PROMPT[subset] 을 쓴다 (LoRD-MEA 와 동일)
    instruction: str = ""
    pool: int = 8192
    max_prompt_tok: int = 96       # 프롬프트만의 상한 (질의 필터)
    max_tok: int = 160             # 프롬프트 + Target 응답 상한
    seed: int = 17

    # ---------------- Target Model (victim) ----------------
    # LoRD 논문의 victim 이 gpt-3.5-turbo 다. 기본값을 거기에 맞춘다.
    target_provider: str = "openai"      # "openai" | "hf" | "reference"
    target_model: str = "gpt-3.5-turbo-1106"
    target_temperature: float = 1.0
    target_max_tokens: int = 128
    target_base_url: str = ""            # OpenAI 호환 다른 제공자
    query_budget: int = 4096             # 실제로 보내는 신규 질의 상한
    dataset_file: str = "target/dataset.json"
    cache_file: str = "target/cache.jsonl"

    # n_train 은 Local 하나가 보는 조각(n_train / k)을 결정한다. 256 에서
    # 학습이 붙는 것을 확인했으므로 건드리지 않는다.
    # n_sel 은 실행 2 의 비용을 그대로 곱한다. 실행 2 는
    #   154칸 x K후보 x |alphas| 번 sel 전체를 다시 잰다.
    n_train: int = 2048
    n_sel: int = 128
    n_check: int = 256
    n_test: int = 512

    # ---------------- fleet ----------------
    k: int = 8                     # Local Model 수
    shard: str = "disjoint"        # "disjoint" | "bootstrap"
    fleet_method: str = "lord"     # "lord" | "sft"
    # 독립 fleet 대조. k 개를 fleet_size 개씩 겹치지 않게 묶는다.
    #   Dependency(single)  개별 Local. 조각 하나 분량
    #   Dependency(union)   조각 fleet_size 개를 합쳐 학습한 단일 모델. 병합 없음
    #   Dependency(merged)  같은 조각들을 학습 후 병합. 병합 있음
    # union 과 merged 는 본 데이터가 같다. 둘의 차이가 병합 자체의 효과다.
    fleet_size: int = 2

    # ---------------- LoRD (LoRD-VI = 논문이 보고하는 방법) ----------------
    # lord_train.py:1109  "LoRD-VI" -> from train_pod2 import train
    #
    #   Eq.10  L = sum_x  log[P(y-)/P(y+)] + clip(log[P(y-)/P(y_vic)])
    #   Eq.11  L = sum_x  sigma( 위 )
    #   Eq.12  lambda1 로 두 항을 볼록결합. lambda1 = 0.5 가 Eq.11 이다
    #
    # lord_variant
    #   "code"   train_pod2.py:963 그대로. L_reg 는 있고 clip 만 없다.
    #            논문 Table 1 의 수치를 낸 것이 이 경로다. 기본값.
    #   "paper"  Eq.10 을 글자 그대로. L_reg 에 clip 을 건다.
    #
    # clip 범위는 [log0.8, log1.2] = [-0.223, +0.182] 인데
    # log P(y-) - log P(y_vic) 는 학습 초반에 이 범위를 크게 벗어난다.
    # 그러면 L_reg 가 포화해 기울기가 0 이 되고 손실이 L_obj 만 남는다 ---
    # 논문 Table 6 의 "w.o. L_reg -> NC(not converged)" 와 같은 상태다.
    # "paper" 를 쓸 때는 로그의 clip_sat 을 반드시 볼 것.
    lord_variant: str = "code"     # "code" | "paper"
    lambda1: float = 0.5           # 논문 §5.1
    tau1: float = 0.8              # 논문 §5.1. 확률 임계
    tau_delta: float = -0.1        # 논문 §5.1 의 τ2. 증가량 임계
    tau2: float = 0.4              # period break 임계 (train_pod2.py)
    log_clip_eps: float = 0.2      # rlhf_train.log_clip. 토큰 단위 clip
    use_sigmoid: bool = True       # Eq.11 의 σ(·). 코드 기본 분기도 sigmoid 다
    lord_lr: float = 3e-5          # 논문 §5.1
    # periods = 0 이면 arm 의 질의 수에 맞춰 자동으로 잡는다.
    #   periods = ceil(len(data) / period_chunk) * lord_epochs
    # 고정값을 쓰면 질의를 많이 가진 arm(union_g, <fleet>_all)이 자기 데이터를
    # 다 보지 못한 채로 끝나 비교가 깨진다.
    periods: int = 0
    lord_epochs: int = 2
    period_chunk: int = 32         # period 당 질의 수. 자주 재표집한다

    # ---------------- SFT (대조군 fleet) ----------------
    sft_lr: float = 1e-5
    sft_epochs: int = 3

    # ---------------- 공통 학습 ----------------
    acc: int = 8                   # 실효 배치
    grad_clip: float = 1.0

    # ---------------- 생성 ----------------
    gen_max_new: int = 64
    gen_temp: float = 1.0
    gen_top_p: float = 0.95
    gen_bs: int = 64

    # ---------------- 텍스트 수준 충실도 ----------------
    # eq:sim 은 Target 응답에 모델이 부여하는 확률이다. "비슷한 답을 내놓는다"
    # 를 주장하려면 모델이 실제로 생성한 문장을 Target 응답과 비교해야 한다.
    # 비교 대상은 데이터셋 정답(ref)이 아니라 Target 응답(gold)이다.
    # LoRD Table 1 은 **데이터셋 정답 문장(ref)** 에 대고 잰 값이다. 그 표와
    # 나란히 놓으려면 ref 대비도 내야 하고, Victim 자신을 ref 에 대고 잰 값이
    # 있어야 Fidelity F = M(local, ref) / M(victim, ref) 가 나온다.
    # 둘 다 낸다. gold 대비는 추출 충실도, ref 대비는 LoRD 비교용이다.
    text_n: int = 256              # test 중 생성에 쓸 질의 수. 0 이면 전부
    text_temp: float = 0.0         # 0 이면 greedy. 보고용은 greedy 가 낫다
    # LoRD §5 가 쓰는 BERTScore. pip install bert-score 가 필요하다.
    # 비우면 건너뛴다.
    bert_score_model: str = "roberta-large"

    # ---------------- 병합 / 선택 ----------------
    # alphas = eq:partial_score 의 Scale 격자. 실행 2 비용이 여기에 비례한다.
    #   154 x K x |alphas| 번 sel 을 다시 잰다. K=8, |alphas|=3 이면 3696 번이다.
    # 논문 축이 종속성으로 바뀌면서 기여도 기반 선택은 주 결과가 아니라
    # ablation 이 되었다. 주 결과(soup / fleet_g / union_g)는 실행 2 를 쓰지
    # 않으므로 격자를 1 점으로 둔다. 되돌리려면 --alphas 로 준다.
    alphas: List[float] = field(default_factory=lambda: [0.25])
    scales: List[float] = field(default_factory=lambda: [
        0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0])
    n_random: int = 8
    cos_max: float = 0.90          # 칸별 코사인 중앙값 관문
    beta: float = 0.1              # eq:soft_weight 의 β

    # ---------------- 평가 ----------------
    boot: int = 4000
    eval_bs: int = 64

    # ---------------- 경로 ----------------
    out_root: str = "runs/default"

    # ---------------- 파생 ----------------
    @property
    def roles(self) -> List[str]:
        return ["Query", "Key", "Value", "Output", "Gate", "Up", "Down"]

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.out_root, "ckpt")

    @property
    def log_dir(self) -> str:
        return os.path.join(self.out_root, "logs")

    @property
    def dataset_path(self) -> str:
        return os.path.join(self.out_root, self.dataset_file)

    @property
    def cache_path(self) -> str:
        return os.path.join(self.out_root, self.cache_file)

    @property
    def local_names(self) -> List[str]:
        return [f"local_{i}" for i in range(self.k)]

    @property
    def all_name(self) -> str:
        return f"{self.fleet_method}_all"

    @property
    def n_fleet(self) -> int:
        return self.k // max(self.fleet_size, 1)

    @property
    def fleets(self) -> List[List[int]]:
        """겹치지 않는 Local 묶음. 독립 fleet 종속성 대조에 쓴다."""
        s = self.fleet_size
        return [list(range(g * s, (g + 1) * s)) for g in range(self.n_fleet)]

    @property
    def union_names(self) -> List[str]:
        """fleet g 의 조각을 합쳐 학습한 단일 모델. 데이터량 대조군."""
        return [f"union_{g}" for g in range(self.n_fleet)]

    @property
    def arm_names(self) -> List[str]:
        return self.local_names + self.union_names + [self.all_name]

    def makedirs(self) -> None:
        for d in (self.ckpt_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]

    # 기존 checkpoint 와 설정이 다르면 섞이지 않도록 막는다.
    def guard(self, extra: dict | None = None) -> None:
        self.makedirs()
        payload = {"cfg": self.to_dict(), "extra": extra or {}}
        path = os.path.join(self.ckpt_dir, "config.json")
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            if old != payload:
                diff = [k for k in set(old.get("cfg", {})) | set(self.to_dict())
                        if old.get("cfg", {}).get(k) != self.to_dict().get(k)]
                raise SystemExit(
                    f"기존 checkpoint 와 설정이 다르다: {diff}\n"
                    f"  {self.ckpt_dir} 를 비우거나 설정을 되돌릴 것."
                )
        else:
            json.dump(payload, open(path, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)


def load(path: str) -> Config:
    raw = json.load(open(path, encoding="utf-8"))
    return Config(**raw.get("cfg", raw))


def load_extra(path: str) -> dict:
    """guard 가 함께 남긴 질의 해시 등."""
    raw = json.load(open(path, encoding="utf-8"))
    return raw.get("extra", {})


def assert_same_data(path: str, query_hash: str, shard_hash: str) -> None:
    """checkpoint 를 만든 질의·조각과 지금 재구성한 것이 같은지 확인한다.

    설정이 같아도 datasets 버전이나 tokenizer 가 바뀌면 질의가 달라질 수 있다.
    그 상태로 측정하면 학습에 쓰인 질의가 평가에 섞일 수 있다.
    """
    ex = load_extra(path)
    bad = []
    if ex.get("query_hash") not in (None, query_hash):
        bad.append(f"query_hash {ex['query_hash']} != {query_hash}")
    if ex.get("shard_hash") not in (None, shard_hash):
        bad.append(f"shard_hash {ex['shard_hash']} != {shard_hash}")
    if bad:
        raise SystemExit("checkpoint 와 질의가 다르다:\n  " + "\n  ".join(bad))
