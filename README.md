# METAMON

Model Extraction via Target-driven Aggregation of Multiple Open-source
Neural-surrogates.

같은 BASE 를 공유하는 여러 Local Model 을 Target 응답으로 학습시킨 뒤 하나로
합친다. 합친 모델은 Target 보다 작고, 추론 비용은 모델 하나 분량이다.

## 주장

둘이다. 둘 다 미리 정한 비교로만 판정한다.

**1. 충실도.** 병합 모델이 최고 단일 surrogate 보다 Target 응답을 잘 재현한다.
확률(`eq:sim`) 과 생성 문장(BLEU / ROUGE-L) 양쪽에서 본다. 확률만 높고 문장이
안 비슷하면 주장이 아니다.

**2. 종속성.** `Dependency(merged) < Dependency(union)`.
`union_g` 는 fleet `g` 의 조각을 **합쳐서** 학습한 단일 모델이고, `fleet_g` 는
같은 조각들을 따로 학습한 뒤 병합한 것이다. 둘은 본 데이터가 같다. 그래서 분산
차이가 병합 자체의 효과가 된다. 이 대조 없이 낸 종속성 숫자는 "데이터를 더 봤을
뿐" 으로 반박당한다. leave-one-out 분산은 K 개 중 K-1 개를 공유하므로 줄어드는
것이 구성상 거의 자동이다. 참고로만 찍고 판정에는 쓰지 않는다.

"작은 모델로 큰 모델을 뽑을 수 있다" 자체는 LoRD 가 이미 보인 것이다. 여기서
더하는 것은 **surrogate 를 하나 고르는 일이 불안정하고, 병합이 그 불안정을
없앤다** 는 부분이다.

## 구성

| 파트 | 내용 | 대응 수식 |
|---|---|---|
| `Pipeline/config.py` | 하이퍼파라미터, 설정 해시 관문 | - |
| `Pipeline/target.py` | Target Model(victim) 질의, 캐시, 질의 예산 | `theta_Target`, `y_Target` |
| `Pipeline/data.py` | 질의 구성, Target 응답 부착, 분할, 중복 검사 | `X`, `Y_Target` |
| `Pipeline/modeling.py` | 모델 1 인스턴스, 7 종 역할 가중치 공간 | `rho` 정의 |
| `Pipeline/metrics.py` | 지표와 통계 | `eq:mean_log_probability`, `eq:sim`, `eq:single_fidelity`, `eq:weighted_loss`, `eq:soft_weight`, `eq:single_dependency` |
| `Pipeline/lord.py` | LoRD 학습 | LoRD Eq.8-11 |
| `Pipeline/sft.py` | 대조군 fleet | - |
| `Pipeline/fleet.py` | fleet 준비, 생존 관문, 복원 검증 | `eq:local_update` |
| `Pipeline/weightspace.py` | 공통 가중치 공간, 다양성 관문 | `eq:norm_matching` (수정판) |
| `Pipeline/contribution.py` | 기여도와 선택 | `eq:perturbed_loss`, `eq:partial_score`, `eq:layer_selection`, `eq:representative_update`, `eq:assembly_verification` |
| `Pipeline/aggregate.py` | 조립 방식과 대조군 | - |
| `Pipeline/evaluate.py` | 배율(check), 최종 비교(test) | - |
| `Pipeline/dependency.py` | 종속성 세 집합, 앙상블 baseline | `eq:single_dependency`, `eq:dependency_mitigation` |
| `Pipeline/textgen.py` | 생성, BLEU / ROUGE-L / BERTScore, 비용표 | - |

## 실행

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=...
python run/00_target.py       --out runs/gpt35 --model gpt-3.5-turbo-1106 \
                              --budget 3072
python run/01_fleet.py        --out runs/gpt35 --fleet lord --k 8 --fleet-size 2
python run/02_contribution.py --out runs/gpt35
python run/03_evaluate.py     --out runs/gpt35 --ensemble --text
python run/04_report.py       --out runs/gpt35
```

`--fleet sft` 로 바꾸면 병합 기전만 따로 볼 수 있다. 실행 2 의 결과가 남아
있으므로 조립 방식만 바꿔 볼 때는 실행 3 부터 다시 돌리면 된다.

`--k 8 --fleet-size 2` 면 학습할 모델이 Local 8 + union 4 + 전체 1 = 13 개다.
`--k 4 --fleet-size 2` 로 줄이면 7 개로 끝나지만 독립 fleet 이 2 개뿐이라
종속성 분산 추정이 얇다. 그때는 실행 1 과 4 가 경고를 찍는다.

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
`n_train + n_sel + n_check + n_test` 다. 기본 설정에서 2048 회다.
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

1. **`log_clip` 은 토큰 단위다.**
   `clamp(logp_t - old_logp_t, log(1-eps), log(1+eps))` 를 토큰마다 적용한 뒤
   mask 로 합산한다. 시퀀스 합에 clip 을 걸면 항상 포화해 `L_reg` 의 기울기가
   0 이 되고, 논문 Table 6 의 `w.o. L_reg -> NC(not converged)` 와 같은 상태가
   된다. 실제로 그렇게 구현하면 수십 update 안에 모델이 한 문장에 확률 1 을
   주는 상태로 붕괴한다.

2. **확률은 토큰 평균의 지수다.**
   `p = exp( sum(logp * mask) / sum(mask) )` 로 (0,1] 범위를 갖는다.
   `tau1 = 0.8` 은 이 값에 대한 임계다. 시퀀스 합을 `log(0.8)` 과 비교하면
   언제나 참이 되어 cold start 가 100% 발동한다.

3. **swap 은 현재 확률 비교**(`p_neg > p_pos`), **cold 는**
   `max(p_pos, p_neg) < tau1` **이고 증가량이** `tau_delta` **미만일 때**
   양성 후보를 Target 응답으로 바꾼다.

4. **period break.** period 안에서 `min(p_pos, p_neg) < tau2` 가 되면 그
   period 를 끊고 다시 표집한다. 음의 항이 무한히 내려가는 것을 막는 장치다.

5. **생성은 왼쪽 padding, 손실은 오른쪽 padding.** decoder-only 모델의 배치
   생성에서 오른쪽 padding 을 쓰면 짧은 프롬프트가 padding 위치에서 생성을
   시작한다. 잘라낼 때는 **첫 EOS 를 포함**한다.

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

- **생성 비교의 상대는 Target 응답이다.** 데이터셋 정답(`ref`) 과 비교하면
  번역 품질을 재는 것이지 추출 충실도가 아니다. `textgen.score_text` 는
  `gold`(Target 응답) 하고만 비교한다.
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
  에서 고른다.
- 모든 arm 에 **같은 배율 격자**를 준다. 상한을 고른 arm 이 있으면 경고한다.
- 기여도 측정은 FP32 로 한다. bf16 은 칸별 차이(1e-4 수준)보다 오차가 커서
  승자가 30% 이상 바뀐다.
- 비교는 질의별 값에 대한 paired bootstrap 이다. 이 구간은 고정된 checkpoint
  에 대한 표본 불확실성이며, 학습 시드 반복을 대신하지 않는다.

## 관문

순서대로 걸리며, 실패하면 그 자리에서 멈춘다.

1. 설정 해시 - 기존 checkpoint 와 설정이 다르면 중단
2. 학습-평가 중복 - 0 이 아니면 중단
3. arm 생존 - 배율 1.0/0.5 중 어느 것도 BASE 를 개선하지 못하면 중단
4. 저장/복원 - 질의별 log-probability 차이가 1e-4 이상이면 중단
5. 다양성 - 칸별 코사인 중앙값이 `cos_max` 를 넘으면 중단 (합칠 것이 없다)
6. 통합 검증 - `eq:assembly_verification` 실패 시 greedy 복구

## 대조군

| arm | 뜻 |
|---|---|
| `<fleet>_all` | 전체 질의로 학습한 단일 모델. 상한선 |
| `local_k` | 개별 Local. 조각 1 개 분량 |
| `union_g` | fleet `g` 의 조각을 합쳐 학습한 단일 모델. **병합 없음** |
| `fleet_g` | 같은 조각들을 따로 학습한 뒤 평균. **병합 있음** |
| `metamon_layer` | `eq:layer_selection` 기반 조립 (주 결과) |
| `metamon_cell` | (layer, role) 마다 argmax. 집중도 대조군 |
| `weighted_t*` | `softmax(PartialScore / T)` 가중 평균. T 가 크면 soup |
| `soup` | 균등 평균 |
| `random_*` | 칸마다 균등 무작위 선택 |
| `shuffle_*` | metamon 의 점유율을 유지한 채 위치만 섞음 |
| `loo_k` | k 번째 Local 을 뺀 평균. `eq:dependency_mitigation` |
| ensemble | 출력 확률 평균. 추론 비용 K 배 |

`metamon - shuffle` 은 **위치를 고른 것**의 기여를, `shuffle - random` 은
**한 모델에 몰린 것**의 효과를 분리한다. 둘을 나누지 않으면 패배 원인을
확정할 수 없다.

`fleet_g - union_g` 는 **데이터량이 같은 짝**이다. 종속성 판정과 충실도 판정이
모두 이 짝 위에 선다.

## 질의 비용 (기본 설정)

| 항목 | 수 |
|---|---|
| 총 질의 | 3072 (train 2048 / sel 256 / check 256 / test 512) |
| Local 하나가 보는 조각 | 256 |
| `union_g` 가 보는 양 | 512 |
| 학습할 모델 | 13 |

`n_train` 을 줄이면 조각이 얇아져 Local 이 안 붙는다. 질의를 아껴야 하면
`k` 를 줄여 조각을 유지하는 쪽이 낫다.
