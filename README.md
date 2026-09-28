# METAMON

Model Extraction via Target-driven Aggregation of Multiple Open-source
Neural-surrogates.

같은 BASE 를 공유하는 여러 Local Model 을 Target 응답으로 학습시킨 뒤 하나로
합친다. 합친 모델은 Target 보다 작고, 추론 비용은 모델 하나 분량이다.

## 전제 — 조각이 IID 면 아무것도 시험되지 않는다

첫 실행이 여기서 깨졌다. `shard = iid` 로 돌렸고, 그러면 조각 16 개가 **같은
분포**에서 나온다. 같은 분포의 조각으로 학습한 모델들은 잡음으로 다르고 능력으로
다르지 않다. 결과는 이랬다.

| | |
|---|---|
| 기여도 점유 | `[77,3,6,4,6,27,0,31]` — 154 칸 중 77 칸이 `local_0` |
| 생성 ROUGE-L | `local_0 0.4896` > `metamon_layer 0.4347` > `lord_all 0.4177` > `soup 0.4145` |

지배하는 구성원이 있으면 **어떤 결합 규칙도 그것을 이길 수 없다.** 기전의
실패가 아니라 전제의 부재다. "Local 1 은 문제 1 을, Local 2 는 문제 2 를
맞춘다" 를 시험하려면 조각이 실제로 서로 다른 영역이어야 한다.

그래서 조각을 비-IID 로 만든다. `Pipeline/shard.py` 가 셋을 준다.

| `shard` | 무엇으로 나누나 | 갈리는 축 |
|---|---|---|
| `cluster` | 원문의 문자 4-gram 을 해싱해 구면 k-means. 기본값 | 어휘 / 주제 |
| `length` | 원문 토큰 길이로 층화 | 문장 길이 |
| `iid` | 무작위. 첫 실행이 이것이었다 | 없음 (대조군) |

세 가지 모두 조각 크기를 `n_train / k` 로 **똑같이** 맞춘다. 크기가 달라지면
`union_g` 의 데이터량이 흔들려 종속성 대조가 깨진다. `cluster` 의 군집은 크기가
들쭉날쭉하므로 확신(1 등과 2 등 유사도의 차) 큰 것부터 용량 제한 배정을 한다.

**평가 집합(`sel` / `check` / `test`)은 건드리지 않는다.** 전체 혼합 분포
그대로다. 그래서 어느 Local 도 평가 전체를 덮지 못하고, 덮으려면 합쳐야 한다.
이 비대칭이 병합에 이길 여지를 주는 유일한 구조다.

해시는 `zlib.crc32` 다. `hash()` 는 프로세스마다 소금이 달라 같은 설정을 두 번
돌리면 조각이 바뀌고 `shard_hash` 관문이 헛돈다.

### 두 개의 진단이 관문이다

**1. 조각 치우침** (`Pipeline/shard.py`, 실행 1 이 학습 **전에** 찍는다)

| 값 | IID 면 | 갈렸으면 |
|---|---|---|
| `len_eta2` | 0 | 길이로 갈렸을 때 1 에 가깝다 |
| `feat_cosine` | 1 에 가깝다 | 어휘로 갈렸을 때 0 에 가깝다 |

두 축 어느 쪽도 갈리지 않으면 실행 1 이 **멈춘다.** 21 개를 6 시간 학습한 뒤에
결과가 뻔한 것을 확인하는 일을 막는다. IID 대조군이 필요할 때만 `--allow-iid`
로 뚫는다.

**2. 상보성 / headroom** (`Pipeline/oracle.py`, 실행 3)

```
oracle(x) = max_k m_k(x)                         질의마다 최고 Local 을 골랐다면
headroom  = mean(oracle) - max_k mean(m_k)
recovery  = (mean(arm) - mean(best_local)) / headroom
```

`oracle` 은 선택 기반 결합이 넘을 수 없는 상한이다. 그래서 `headroom` 이
전부를 결정한다.

- `headroom ~ 0` → 한 Local 이 거의 모든 질의에서 이긴다. 어떤 병합도 최고
  단일을 못 이긴다. **기전을 고칠 문제가 아니라 조각을 바꿀 문제다.**
- `headroom >> 0` → 상보성이 있다. 이제 `recovery` 가 기전의 성적이다.

`최상위 Local 승률` 도 같이 찍는다. `1/K` 면 완전 상보, `100%` 면 완전 지배다.
첫 실행에서 이것을 재지 않은 것이 실수였다. 기전을 여섯 군데 고치기 전에 뽑을
것이 있는지부터 확인해야 했다.

## 주장

**base 가 주축이고, 병합은 base 가 덮지 못한 영역을 덮는 보정항이다.**

LoRD 는 Llama-3-8B 를 base 로, 질의 **16 개**로 `F(ROUGE-L) = 0.891` 을 냈다
(base 0.348/0.604 = 0.576). 질의 수는 지렛대가 아니다. base 가 전부다.
그리고 부록에서 스스로 선을 그어 놨다.

> "a model with **2.7 billion** appears sufficient to steal domain-specific
> knowledge from commercial LLMs"
> "**Phi-3 (3.8B)** achieves a comparable fidelity to larger models like Llama-3 (8B)"

그 하한 바로 위를 잡아서, 검증할 등식을 하나로 줄인다.

```
F( 3B + METAMON )  >=  F( 8B + single LoRD ) = 0.891
```

**1. 충실도.** 병합이 최고 단일 surrogate 와 전체 질의 단일 모델(`lord_all`)을
이긴다. **생성에서** 본다. LoRD Table 1 이 생성이고, 확률(`eq:sim`)과 생성의
순위는 병합 arm 에서 Spearman −0.821 로 거의 뒤집혀 있다. 확률만 높은 것은
주장이 아니다.

**2. 기전 — 이게 본체다.** "구멍을 메운다" 는 평균이 아니라 **분포**의 주장
이다. 질의를 BASE 의 점수로 5 등분하고 버킷마다 `병합 − 최고단일` 을 낸다.

| | 뜻 |
|---|---|
| 이득이 **낮은 버킷에 몰린다** | base 가 못 하는 데서 병합이 산다. 성립 |
| 이득이 고르게 퍼진다 | 보정이 아니라 그냥 평균 효과. 기각 |

여기서 예측이 하나 따라 나온다. **base 가 커지면 메울 구멍이 줄어드니 이득이
줄어야 한다.** 크기 축의 두 점에서 기울기가 같은 방향이면 기전 주장이 선다.

**3. 종속성.** `Dependency(merged) < Dependency(union)`. `union_g` 는 fleet
`g` 의 조각을 **합쳐서** 학습한 단일 모델이고 `fleet_g` 는 같은 조각들을 따로
학습한 뒤 병합한 것이다. 둘은 본 데이터가 같으므로 분산 차이가 병합 자체의
효과다. 이 대조 없이 낸 종속성 숫자는 "데이터를 더 봤을 뿐" 으로 반박당한다.
leave-one-out 은 K 개 중 K−1 개를 공유하므로 참고로만 찍는다.

> 병합 크기 곡선(m = 1, 2, 4, 8)은 **삭제했다.** 조각이 IID 면 `σ²/m` 이라
> 산수지 발견이 아니고, 비-IID 로 만든 뒤에는 묶음의 m 개가 같은 분포에서
> 뽑힌 것이 아니어서 전제마저 깨진다. 실측도 단조가 아니었다
> (m=1 1.49e-3, m=2 2.99e-3, m=4 3.21e-3).

## 구성

| 파트 | 내용 | 대응 수식 |
|---|---|---|
| `Pipeline/config.py` | 하이퍼파라미터, 설정 해시 관문 | - |
| `Pipeline/target.py` | Target Model(victim) 질의, 캐시, 질의 예산 | `theta_Target`, `y_Target` |
| `Pipeline/data.py` | 질의 구성, Target 응답 부착, 분할, 중복 검사 | `X`, `Y_Target` |
| `Pipeline/shard.py` | 조각 나누기(cluster / length / iid), 치우침 진단 | - |
| `Pipeline/deltastore.py` | Δw 를 safetensors 로 두고 칸 단위로 읽기 | - |
| `Pipeline/buckets.py` | BASE 취약도별 이득 분해 | - |
| `Pipeline/modeling.py` | 모델 1 인스턴스, 7 종 역할 가중치 공간 | `rho` 정의 |
| `Pipeline/metrics.py` | 지표와 통계 | `eq:mean_log_probability`, `eq:sim`, `eq:single_fidelity`, `eq:weighted_loss`, `eq:soft_weight`, `eq:single_dependency` |
| `Pipeline/lord.py` | LoRD 학습 | LoRD Eq.8-11 |
| `Pipeline/sft.py` | 대조군 fleet | - |
| `Pipeline/fleet.py` | fleet 준비, 생존 관문, 복원 검증 | `eq:local_update` |
| `Pipeline/weightspace.py` | 공통 가중치 공간, 다양성 관문 | `eq:norm_matching` (수정판) |
| `Pipeline/contribution.py` | 기여도와 선택 | `eq:perturbed_loss`, `eq:partial_score`, `eq:layer_selection`, `eq:representative_update`, `eq:assembly_verification` |
| `Pipeline/aggregate.py` | 조립 방식과 대조군 | - |
| `Pipeline/evaluate.py` | 배율(check), 최종 비교(test) | - |
| `Pipeline/oracle.py` | oracle, headroom, 승자 점유, 회수율 | - |
| `Pipeline/dependency.py` | 종속성 세 집합, 앙상블 baseline | `eq:single_dependency`, `eq:dependency_mitigation` |
| `Pipeline/textgen.py` | 생성, BLEU / ROUGE-L / BERTScore, 비용표 | - |

## 실행

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=...
export OPENAI_API_KEY=...
python run/00_target.py       --out runs/gpt35 --model gpt-3.5-turbo-1106 \
                              --budget 4992

# 크기 축의 한 점. --base 만 바꿔서 같은 질의로 다시 돌린다.
python run/01_fleet.py        --out runs/l32 --fleet lord --shard cluster \
                              --base meta-llama/Llama-3.2-3B-Instruct
python run/02_contribution.py --out runs/l32
python run/03_evaluate.py     --out runs/l32 --ensemble --text
python run/04_report.py       --out runs/l32
```

분할 크기(`n_train`, `n_sel`)는 실행 1 에서 `ckpt/config.json` 에 박힌다.
실행 2, 3 은 그 파일을 읽으므로 나중에 바꿀 수 없다. 실행 0 의 질의 수와
맞춰야 한다. `alphas` 와 `scales` 는 실행 2, 3 에서 그때그때 덮어쓸 수 있다.

`--fleet sft` 로 바꾸면 병합 기전만 따로 볼 수 있다. 실행 2 의 결과가 남아
있으므로 조립 방식만 바꿔 볼 때는 실행 3 부터 다시 돌리면 된다.

기본값 `--k 8 --fleet-size 2` 면 학습할 모델이 Local 8 + union 4 + 전체 1 =
**13 개**다. `k` 를 정하는 것은 디스크다(위의 크기 축 표). `fleet_size` 를 2 로
둔 것은 기여 3 의 분산이 묶음 4 개 위에 서게 하기 위한 것이고, 기여 1 의 주
병합(`soup` / `greedy`)은 K 전체를 쓰므로 영향을 안 받는다.

### 크기 축

`--base` 만 바꿔 같은 표를 다시 채운다. **Target 응답은 surrogate 와 무관하므로
API 비용이 0 이다** — `target/cache.jsonl` 을 그대로 쓴다. 그게 성립하려면
질의 필터가 base 와 무관해야 하고, `cfg.filter_tokenizer` 가 그 고정 기준이다.
surrogate 토크나이저로 거르면 base 마다 질의 집합이 달라져 캐시도 안 맞고 크기
비교도 깨진다.

| base | 추적 파라미터 | arm 하나 (fp16) | 13 arm 합계 |
|---|---|---|---|
| TinyLlama-1.1B | 0.97B | 1.9GB | **25GB** |
| Llama-3.2-3B | 2.82B | 5.6GB | **73GB** |
| Llama-3-8B | 6.98B | 14.0GB | 182GB |

**막는 것은 GPU 가 아니라 디스크다.** arm 하나가 Δw 를 통째로 갖는다. `k` 를
정할 때 이 표를 먼저 본다. GPU 는 3B full fine-tuning 이 모델 12.8 + grad 12.8
+ Adam 25.6 = 51GB 라 80GB A100 에 들어간다.

## Target Model

공격자가 관측할 수 있는 것은 응답뿐이다. `run/00_target.py` 가 질의를 보내고
응답을 모은다. 이후 실행은 저장된 데이터셋만 읽으므로 API 를 다시 부르지 않는다.

| `--provider` | 설명 | 비용 |
|---|---|---|
| `openai` | OpenAI 호환 API. `--base-url` 로 다른 제공자에도 붙는다 | 질의당 과금 |
| `hf` | 로컬 HuggingFace 모델을 victim 으로 사용 | GPU 시간 |
| `reference` | 데이터셋 정답 문장을 응답 대신 사용. **배관 점검 전용** | 0 |

프롬프트 형식은 LoRD-MEA 와 같다.

```
Target : system = "Instruction: Translate the sentence from Czech to English Please."
         user   = "<원문>"
Local  : "Instruction: {pp} User: {원문} Assistant: "
```

### 질의 비용

`eq:sim` 은 **Target 응답**에 모델이 부여하는 확률이다. 따라서 `sel`, `check`,
`test` 도 Target 응답이 필요하다. 총 질의 수는
`n_train + n_sel + n_check + n_test` 다. 기본 설정에서 4992 회다.
줄이려면 `n_test` 가 아니라 `n_train` 을 줄인다. LoRD 의 주장이 질의 효율이므로
`n_train` 이 작을수록 오히려 논지에 맞다.

- `<out>/target/cache.jsonl` 에 질의-응답 원본이 append 로 쌓인다. 중간에
  끊겨도 이미 산 응답은 남고, 다시 돌리면 캐시에서 읽는다.
- `--budget` 은 **새로 보내는** 질의의 상한이다. 넘으면 중단한다.
- `[회계] 신규 질의 / 캐시 적중` 을 실행 로그와 `00_target.json` 에 남긴다.

### `reference` 로 돌린 결과를 보고하지 말 것

정답 문장은 Target 모델의 응답이 아니다. 이 모드로 나온 수치는 추출 충실도가
아니라 참조 문장 재현도다. 실행 0 과 1 이 그때마다 경고를 찍는다.

## LoRD 이식에서 반드시 지켜야 하는 것

원본은 `LoRD-MEA/lord_train.py`, `train_pod2.py`, `rlhf_train.py` 다.

0. **어느 변형인지부터 맞출 것.** `lord_train.py:1109` 에서
   `LoRD-VI -> from train_pod2 import train` 이다. 논문 Table 1 의 수치는 이
   경로에서 나온다. `lord_train.py` 안의 `lord` / `Complex-lord` 는 다른
   변형이고 손실이 다르다.

1. **손실에 `y_vic` 가 들어간다** (Eq.10, `train_pod2.py:959-963`).

   ```
   L_obj = log P(y-|x) - log P(y+|x)
   L_reg =     log P(y-|x) - log P(y_vic|x)          # lord_variant="code"
           clip(log P(y-|x) - log P(y_vic|x))        # lord_variant="paper"
   L     = sigma( 2 * [ (1-lambda1)*L_obj + lambda1*L_reg ] )
   ```

   `lambda1 = 0.5` 이면 Eq.11 과 같다. Target 응답이 목적함수에 직접 있어야
   추출 알고리즘이 된다. 이 항이 없으면 자기 표본끼리의 선호 최적화일 뿐이다.

2. **clip 은 `"paper"` 에서만 건다. 기본값은 `"code"` 다.**
   공개 구현은 clip 항을 `train_pod2.py:941` 에서 계산만 하고 `:963` 의 손실에
   넣지 않는다. clip 범위가 `[-0.223, +0.182]` 인데 `log P(y-) - log P(y_vic)`
   는 초반에 이를 크게 벗어나므로, clip 을 걸면 `L_reg` 가 포화해 기울기가 0 이
   된다. 그 상태가 논문 Table 6 의 `w.o. L_reg -> NC(not converged)` 다.
   `"paper"` 로 돌릴 때는 로그의 `clip_sat` 을 보고 판단한다.

3. **확률은 토큰 평균의 지수다.**
   `p = exp( sum(logp * mask) / sum(mask) )` 로 (0,1] 범위를 갖는다.
   `tau1 = 0.8` 은 이 값에 대한 임계다. 시퀀스 합을 `log(0.8)` 과 비교하면
   언제나 참이 되어 cold start 가 100% 발동한다. 손실의 `log P(y|x)` 도 같은
   길이 정규화를 쓴다. 합을 쓰면 세 후보의 길이 차이가 손실을 지배한다.

4. **swap 은 현재 확률 비교**(`p_neg > p_pos`), **cold 는**
   `max(p_pos, p_neg) < tau1` **이고 증가량이** `tau_delta` **미만일 때**
   양성 후보를 Target 응답으로 바꾼다.

5. **period break.** period 안에서 `min(p_pos, p_neg) < tau2` 가 되면 그
   period 를 끊고 다시 표집한다. 음의 항이 무한히 내려가는 것을 막는 장치다.

6. **period 수는 arm 의 질의 수에 맞춘다.** `periods = 0` 이면
   `ceil(len(data)/period_chunk) * lord_epochs` 로 잡는다. 고정값을 쓰면
   질의가 많은 arm(`union_g`, `<fleet>_all`)이 자기 데이터를 다 보지 못한 채
   끝나 `fleet_g` 와의 비교가 깨진다.

7. **생성은 왼쪽 padding, 손실은 오른쪽 padding.** decoder-only 모델의 배치
   생성에서 오른쪽 padding 을 쓰면 짧은 프롬프트가 padding 위치에서 생성을
   시작한다. 잘라낼 때는 **첫 EOS 를 포함**한다.

### 원본을 그대로 옮기지 않은 두 곳

- 원본은 `torch.mean(logits2_cons)` 로 mask 를 쓰지 않아 프롬프트 토큰이 평균에
  섞인다(`:986-988` 의 `print` 에서만 mask 를 쓴다). 여기서는 응답 토큰만 쓴다.
- 원본은 직전 확률을 `sum(exp(logp)*mask)/sum(mask)` 로, 현재 확률을
  `exp(sum(logp*mask)/sum(mask))` 로 계산해 서로 다른 양을 빼서 `delta` 를
  만든다(`:581` vs `:638`). 여기서는 둘 다 후자로 통일한다.

### 임계값 이름 대응

논문 §5.1 과 원본 코드의 이름이 다르다. `config.py` 는 원본 코드 쪽 역할로 나눈다.

| `config.py` | 뜻 | 논문 §5.1 | `train_pod2.py` 기본값 |
|---|---|---|---|
| `tau1` | 확률 임계. `max(p+, p-) < tau1` 이면 cold 후보 | `tau1 = 0.8` | `0.99` |
| `tau_delta` | 증가량 임계. cold 의 두 번째 조건 | `tau2 = -0.1` | `tau_delta` |
| `tau2` | period break. `min(p+, p-) < tau2` 면 재표집 | (없음) | `0.998` |

`tau2` 를 원본 기본값 `0.998` 로 두면 거의 매번 period 가 끊겨 사실상 1 스텝마다
재표집한다. 기본값은 `0.4` 로 낮춰 두었다. 이 값은 논문 근거가 아니라 선택이므로
바꿀 때는 로그의 `p+ / p-` 를 보고 정한다.

## 측정 규칙

- **생성 비교는 두 기준선을 다 낸다.**

  | 기준 | 뜻 | 쓰는 곳 |
  |---|---|---|
  | `gold` (Target 응답) 대비 | 추출 충실도 | 본 지표. 주장 1 |
  | `ref` (데이터셋 정답) 대비 | 번역 품질 | LoRD Table 1 과 같은 축 |
  | `F = M(local, ref) / M(victim, ref)` | Fidelity (LoRD Eq.12) | 모델 크기가 달라도 비교된다 |

  LoRD Table 1 은 전부 `ref` 대비다. `gold` 대비 수치를 그 표와 나란히 놓으면
  안 된다. Victim 자신을 `ref` 에 대고 잰 값이 `F` 의 분모이고, 실행 3 이
  `victim_text` 로 남긴다.

  `MLE` 행은 `--fleet sft` 로 따로 돌려서 채운다. `KD` 는 Target 의 logit 이
  필요해 black-box 에서는 못 한다 (LoRD §5.1 도 grey-box 에서만 잰다).
  `N/A (black-box)` 로 둔다.
- 생성은 왼쪽 padding, 기본은 greedy(`text_temp = 0`) 다. 표집으로 재면
  같은 모델도 매번 다른 BLEU 가 나와 arm 비교가 흔들린다.
- 선택 기준은 **손실**이다(`eq:weighted_loss`). `avgBF` 는 확률의 평균이라
  질의를 가로질러 평균하는 순간 순서가 달라진다. 보고에만 쓴다.
- `--omega soft` 를 쓰면 `eq:soft_weight` 의 `omega_k` 를 **후보마다 다르게**
  적용하고 기준 손실도 `L(theta; omega_k)` 로 후보별로 잡는다. K 개를 하나로
  평균하면 균등 가중치와 같아져 `eq:soft_weight` 가 의미를 잃는다.
  단, `eq:assembly_verification` 은 언제나 `omega = 1` 로 검증한다.
- 측정에 쓴 후보를 **그대로** 조립한다. 고른 뒤 크기를 바꾸면 다른 대상을
  평가한 것이 된다.
- 배율은 `check` 에서만 고르고 `test` 는 한 번만 잰다. 최고 Local 도 `check`
  에서 고른다. **기준은 셋이고 보고하는 지표와 같은 것을 쓴다.** `avgBF` 로
  고르면 생성에서 5 위인 Local 이 뽑힌다. 확률 판정은 손실 기준 최고 Local,
  생성 판정은 `check` 생성 ROUGE-L 기준 최고 Local 을 상대로 한다.
- 모든 arm 에 **같은 배율 격자**를 준다. 상한을 고른 arm 이 있으면 경고한다.
- 기여도 측정은 FP32 로 한다. bf16 은 칸별 차이(1e-4 수준)보다 오차가 커서
  승자가 30% 이상 바뀐다.
- 비교는 질의별 값에 대한 paired bootstrap 이다. 이 구간은 고정된 checkpoint
  에 대한 표본 불확실성이며, 학습 시드 반복을 대신하지 않는다.
- **생성 지표에도 구간을 낸다.** 질의쌍을 재표집하고 BLEU 는 재표집마다 코퍼스
  수준에서 다시 계산한다(Koehn 2004). ROUGE-L 은 질의별 값의 평균이다. 구간
  없이 `0.4347 > 0.4177` 만 써서 이겼다고 말하면 안 된다.

## 관문

순서대로 걸리며, 실패하면 그 자리에서 멈춘다.

1. 설정 해시 - 기존 checkpoint 와 설정이 다르면 중단
1.5 조각 치우침 - 두 축 어느 쪽도 갈리지 않으면 중단 (`--allow-iid` 로 뚫는다)
2. 학습-평가 중복 - 0 이 아니면 중단
3. arm 생존 - 비-IID local 은 **자기 조각**에서, union / all 은 전체 혼합
   `sel` 에서 본다. local 이 전체 혼합에서 못 오르는 것은 좁게 배운 설계의
   결과이므로 기록만 하고 넘어간다. 자기 조각에서도 못 오른 local 이 4 개면
   학습 자체가 고장난 것이므로 중단한다. union / all 은 한 번이라도 실패하면 중단
4. 저장/복원 - 질의별 log-probability 차이가 1e-4 이상이면 중단
5. 다양성 - 칸별 코사인 중앙값이 `cos_max` 를 넘으면 중단 (합칠 것이 없다)
5.5 상보성 - `headroom` 이 0 이면 실행 4 가 기여 1 을 '불성립' 이 아니라
   **'미시험'** 으로 찍는다. 뽑을 것이 없으면 기전을 고칠 문제가 아니다
6. 통합 검증 - `eq:assembly_verification` 실패 시 greedy 복구

## 대조군

| arm | 뜻 |
|---|---|
| `<fleet>_all` | 전체 질의로 학습한 단일 모델. 상한선 |
| `local_k` | 개별 Local. 조각 1 개 분량 |
| `union_g` | fleet `g` 의 조각을 합쳐 학습한 단일 모델. **병합 없음** |
| `fleet_g` | 묶음 `g` 의 Local 평균. **병합 있음**. `union_g` 와 데이터량이 같다 |
| `greedy` | 최고 단일에서 출발해 좋아질 때만 채택. **주 결과**. `greedy_soup` 과 `metamon_greedy` 중 check 에서 좋은 쪽 |
| `greedy_soup` | 하나씩 더해 본다 (Wortsman et al.) |
| `metamon_greedy` | PartialScore 큰 칸부터 갈아끼워 본다 |
| `metamon_layer` | `eq:layer_selection` 기반 조립 |
| `metamon_cell` | (layer, role) 마다 argmax. 집중도 대조군 |
| `weighted_t*` | `softmax(PartialScore / T)` 가중 평균. T 가 크면 soup |
| `soup` | 균등 평균 |
| `random_*` | 칸마다 균등 무작위 선택 |
| oracle | 질의마다 최고 Local. 선택 기반 결합의 상한. arm 이 아니라 진단이다 |
| `shuffle_*` | metamon 의 점유율을 유지한 채 위치만 섞음 |
| `loo_k` | k 번째 Local 을 뺀 평균. `eq:dependency_mitigation` |
| ensemble | 출력 확률 평균. 추론 비용 K 배 |

`metamon - shuffle` 은 **위치를 고른 것**의 기여를, `shuffle - random` 은
**한 모델에 몰린 것**의 효과를 분리한다. 둘을 나누지 않으면 패배 원인을
확정할 수 없다.

`fleet_g - union_g` 는 **데이터량이 같은 짝**이다. 기여 3 의 판정이 이 짝
위에 선다.

`soup` 은 구성상 최고 단일을 못 이긴다. K 개 중 하나만 좋으면 나머지 K-1 개가
끌어내린다. 그래서 주 결과 arm 은 최고 단일에서 **출발하는** `greedy` 이고
`soup` 은 `w/o Selection` ablation 행이다.

**greedy 의 채택 기준은 보고할 지표와 같아야 한다.** 지난 실행은 `check` 의
avgBF 로 채택했는데, 병합 arm 에서 avgBF 와 생성 ROUGE-L 의 Spearman 이
−0.821 이었다. 확률을 올리는 방향이 생성을 내리는 방향이어서, 최고 단일에서
출발하고도 생성에서 졌다. 기본값은 `--greedy-on gen` 이다. 그래야 생성에서
최고 단일 이상이 **구성상** 보장된다.

## 비용

| 항목 | 수 |
|---|---|
| 총 질의 | 4992 (train 4096 / sel 128 / check 256 / test 512) |
| Local 하나가 보는 조각 | 512 |
| `union_g` 가 보는 양 | 1024 |
| 학습할 모델 | 13 (local 8 + union 4 + all 1) |

| 실행 | 1.1B | 3B | 무엇에 비례하는가 |
|---|---|---|---|
| 0 Target | ~1.7h | 0 (캐시) | 질의 수. API 직렬 |
| 1 fleet | ~4h | ~11h | `n_train` x `lord_epochs` x 모델 크기 |
| 2 contribution | ~0.6h | ~2h | **칸수 x K x `len(alphas)` x `n_sel`** |
| 3 evaluate | ~1.5h | ~4h | arm 수 x `len(scales)` x `n_check` |

실행 3 에서 `--greedy-on gen` 은 채택 판정마다 `check` 문장을 생성한다.
`greedy_n`(128), `greedy_scales`(3점), `greedy_cells`(40)이 그 비용을 잡는
손잡이다.

`alphas` 를 1 점으로 둔 것은 기여도 기반 선택이 주 결과가 아니라 ablation 이기
때문이다. 주 결과인 `greedy` 는 칸 **순서**에만 PartialScore 를 쓰므로 격자
1 점으로 충분하다. 선택 자체를 본 결과로 쓸 때는 `--alphas 0.125 0.25 0.5` 로
되돌린다.
