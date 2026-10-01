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
from typing import List, Tuple

# 보고 지표. LoRD §5 와 같다.
METRICS: Tuple[str, ...] = ("BLEU-1", "BLEU-4", "ROUGE-L", "BERT-F1")


@dataclass
class Config:
    # ---------------- 모델 / 데이터 ----------------
    # base 가 주축이다. LoRD 는 Llama-3-8B 로 F(ROUGE-L) 0.891 을 냈고
    # 부록에서 "2.7B 면 충분", "Phi-3(3.8B)가 Llama-3(8B)와 맞먹는다" 고 했다.
    # 그 하한 바로 위를 잡아 "3B 병합 = 8B 단일" 을 시험한다.
    #   TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T   1.1B  (가장 작은 점)
    #   meta-llama/Llama-3.2-3B-Instruct                      3.2B  (본 실험)
    #   Qwen/Qwen2.5-3B-Instruct                              3.1B  (차선)
    base: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dataset: str = "wmt/wmt16"
    subset: str = "cs-en"
    src_key: str = "cs"
    tgt_key: str = "en"
    # 빈 문자열이면 target.TASK_PROMPT[subset] 을 쓴다 (LoRD-MEA 와 동일)
    instruction: str = ""
    pool: int = 8192
    # 질의 필터는 **base 와 무관**해야 한다. surrogate 토크나이저로 거르면
    # base 를 바꿀 때마다 질의 집합이 달라져 캐시가 안 맞고 크기 비교가 깨진다.
    # 고정 토크나이저 하나로만 재고, 그 기준을 설정 해시에 박는다.
    # **이미 산 캐시를 살리려면 이 값이 첫 실행 때와 같아야 한다.** 필터는
    # "고정" 이기만 하면 되고 어느 토크나이저인지는 상관없다. 첫 4992 질의를
    # TinyLlama 토크나이저로 걸러서 샀으므로 그것을 기준으로 못박는다.
    # 바꾸면 질의 집합이 달라져 캐시가 전부 빗나가고 돈을 다시 낸다.
    # (게이트된 저장소를 기준으로 삼지 말 것. HF 토큰이 없으면 못 읽는다.)
    filter_tokenizer: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    max_prompt_tok: int = 96       # filter_tokenizer 기준 프롬프트 상한
    max_tok: int = 160             # 프롬프트 + Target 응답 상한
    seed: int = 17

    # ---------------- Target Model (victim) ----------------
    # LoRD 논문의 victim 이 gpt-3.5-turbo 다. 기본값을 거기에 맞춘다.
    target_provider: str = "openai"      # "openai" | "hf" | "reference"
    target_model: str = "gpt-3.5-turbo-1106"
    target_temperature: float = 1.0
    target_max_tokens: int = 128
    target_base_url: str = ""            # OpenAI 호환 다른 제공자
    query_budget: int = 4992             # 실제로 보내는 신규 질의 상한
    dataset_file: str = "target/dataset.json"
    cache_file: str = "target/cache.jsonl"

    # n_train 은 Local 하나가 보는 조각(n_train / k)을 결정한다. 256 에서
    # 학습이 붙는 것을 확인했으므로 건드리지 않는다.
    # n_sel 은 실행 2 의 비용을 그대로 곱한다. 실행 2 는
    #   154칸 x K후보 x |alphas| 번 sel 전체를 다시 잰다.
    n_train: int = 4096
    n_sel: int = 128
    n_check: int = 256
    n_test: int = 512

    # ---------------- fleet ----------------
    # K 를 정하는 것은 디스크다. arm 하나가 Δw 를 통째로 갖는다.
    #   1.1B fp16  0.97B x 2B = 1.9GB/arm    K=16 (21 arm) ->  41GB
    #   3.2B fp16  2.82B x 2B = 5.6GB/arm    K=16 (21 arm) -> 118GB  못 담는다
    #                                        K=8  (11 arm) ->  62GB  가능
    k: int = 8                     # Local Model 수
    # Δw 저장 정밀도. Δw 는 BASE 대비 1e-2 수준이라 fp16 으로 충분하다.
    # verify_restore 가 질의별 log-probability 로 1e-4 관문을 건다.
    delta_dtype: str = "float16"   # "float16" | "bfloat16" | "float32"
    # 조각을 어떻게 나누는가. **이것이 실험의 전제다.**
    #   iid       무작위. 조각 16 개가 같은 분포다. 상보성 0
    #   length    원문 길이로 층화. 조각마다 길이 대역이 다르다
    #   cluster   원문 문자 n-gram 군집. 조각마다 어휘/주제가 다르다
    #
    # 처음 실행을 iid 로 했고, 그래서 local_0 이 154 칸 중 77 칸을 점유하고
    # 생성에서 모든 병합을 이겼다. 같은 분포의 조각으로 학습한 모델들은 잡음
    # 으로 다르고 능력으로 다르지 않다. 지배하는 구성원이 있으면 어떤 결합
    # 규칙도 그것을 못 이긴다. 기전의 실패가 아니라 전제의 부재였다.
    # 평가 집합은 전체 혼합 그대로 두므로 어느 Local 도 평가를 덮지 못한다.
    # 그 비대칭이 병합에 이길 여지를 주는 유일한 구조다.
    shard: str = "cluster"         # "cluster" | "length" | "iid" | "bootstrap"
    fleet_method: str = "lord"     # "lord" | "sft"
    # 독립 fleet 대조. k 개를 fleet_size 개씩 겹치지 않게 묶는다.
    #   Dependency(single)  개별 Local. 조각 하나 분량
    #   Dependency(union)   조각 fleet_size 개를 합쳐 학습한 단일 모델. 병합 없음
    #   Dependency(merged)  같은 조각들을 학습 후 병합. 병합 있음
    # union 과 merged 는 본 데이터가 같다. 둘의 차이가 병합 자체의 효과다.
    #
    # fleet_size = 2 로 재 보니 merged 의 분산이 오히려 커졌다. 좋은 Local 과
    # 나쁜 Local 을 1:1 로 섞으면 희석이 안 되기 때문이다(구성원 둘보다 낮은
    # 묶음이 나왔다). 반면 7 개를 합친 loo 는 분산이 1/92 였다. 그래서 단일
    # 비교 대신 **병합 크기별 곡선**으로 본다. curve_sizes 의 각 m 에 대해
    # k 개를 m 개씩 겹치지 않게 묶어 분산을 잰다.
    # E2 는 fleet_g 와 union_g 의 분산을 비교한다. n_fleet 이 2 면 분산이
    # 표본 2개라 뜻이 없다. fleet_size=2 로 두어 묶음 4개를 만든다.
    # E1 의 주 병합(soup / greedy)은 K 전체를 쓰므로 여기 영향을 안 받는다.
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
    # 논문 §5.1 은 3e-5 다. 그 값은 1.1B(base L 1.92958)에서는 맞았지만
    # Llama-3.2-3B(base L 0.78472)에서는 과도했다. 13 arm 전부를 배율별로 다시
    # 재 보니 **같은 모양**이 나왔다 (run/01b_audit.py):
    #   arm        L@1.0    L@0.5   L@0.25      BASE = 0.78472
    #   local_0   0.92378  0.64219  0.57212
    #   union_0   1.03053  0.67843  0.57978
    #   lord_all  1.24758  0.71960  0.57945
    # 방향은 맞고 **크기만 4 배쯤 크다**. 13 번의 고장이 아니라 한 개의 상수다.
    # 저장 시점을 최적으로 바꾸는 것만으로는 부족하다. union_0 의 궤적에서 측정된
    # 최저값이 0.88812 로 이미 BASE 보다 나빴다. 보폭 자체를 줄여야 한다.
    lord_lr: float = 7.5e-6        # = 3e-5 / 4. 위 측정에서 나온 배수다
    # periods = 0 이면 arm 의 질의 수에 맞춰 자동으로 잡는다.
    #   periods = ceil(len(data) / period_chunk) * lord_epochs
    # 고정값을 쓰면 질의를 많이 가진 arm(union_g, <fleet>_all)이 자기 데이터를
    # 다 보지 못한 채로 끝나 비교가 깨진다.
    periods: int = 0
    lord_epochs: int = 2
    # period 당 질의 수. acc 와 같게 두는 것이 중요하다.
    #   chunk > acc 이면 period break 가 첫 update 에서 걸릴 때 나머지 질의의
    #   생성 결과가 통째로 버려진다. chunk 32, acc 8 로 돌렸더니 계획한 64
    #   update 중 16 만 돌았고 질의 절반은 학습에 한 번도 안 들어갔다.
    #   chunk = acc 면 방문 수(periods x chunk)가 같은 채로 update 가 4 배다.
    #   LoRD 원본도 sub_set_num=1 로 질의 하나마다 재표집한다.
    period_chunk: int = 8

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
    n_random: int = 3               # E4 의 w/o Selection—Random
    n_shuffle: int = 1              # 선택 자체의 값어치 대조
    # greedy 채택 판정에 쓸 check 질의 수. 생성으로 채점하므로 비싸다.
    greedy_n: int = 128
    greedy_cells: int = 40          # metamon_greedy 가 시도할 상위 칸 수
    greedy_scales: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0])
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
        """겹치지 않는 Local 묶음. union_g 와 데이터량이 같은 짝을 만든다."""
        m = max(1, min(self.fleet_size, self.k))
        return [list(range(g * m, (g + 1) * m)) for g in range(self.k // m)]

    @property
    def union_names(self) -> List[str]:
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
