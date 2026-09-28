# METAMON 실험 결과 — 설계 문서(전체 실험 구성) 표 채우기

출처: `runs/gpt35` 1회 실행. TinyLlama-1.1B, K=16, fleet_size=4, `shard=cluster`, victim `gpt-3.5-turbo-1106`, WMT16 cs-en.
질의 4992 (train 4096 / sel 128 / check 256 / test 512). 생성은 test 상위 256, greedy.

> **BERTScore-F1 열은 비어 있습니다.** 값은 계산됐지만 로그에 arm 별로 출력되지 않았고, `03_evaluate.json` 이 써지기 전에 실행이 죽었습니다. 생성 결과는 이제 캐시되므로 03 재실행(몇 분)으로 채워집니다.

---

## E1 — 추출 성능 비교

`Fidelity F = M(surrogate, ref) / M(victim, ref)` (LoRD Eq.12). BLEU-4 / ROUGE-L 은 **ref(데이터셋 정답) 대비**로 적습니다 — LoRD Table 1 과 같은 축입니다. Target 응답(gold) 대비 값은 그 아래 별도 표에 있습니다.

| Method | BLEU-4 ↑ | ROUGE-L ↑ | BERTScore-F1 ↑ | Fidelity F ↑ |
|---|---|---|---|---|
| Target Model | 0.3343 | 0.6188 | 0.9553 | 1.000 |
| Basic θ (BASE) | 0.0046 | 0.0990 | — | 0.160 |
| MLE | — | — | — | — |
| KD | N/A (black-box) | N/A | N/A | N/A |
| local_0 + LoRD | 0.0669 | 0.2682 | — | 0.433 |
| local_1 + LoRD | 0.0866 | 0.3146 | — | 0.508 |
| local_2 + LoRD | 0.1328 | 0.4281 | — | 0.692 |
| local_3 + LoRD | 0.0585 | 0.2446 | — | 0.395 |
| local_4 + LoRD | 0.0912 | 0.3191 | — | 0.516 |
| local_5 + LoRD | 0.0976 | 0.3280 | — | 0.530 |
| local_6 + LoRD | 0.1325 | 0.4124 | — | 0.666 |
| local_7 + LoRD | 0.0777 | 0.3063 | — | 0.495 |
| local_8 + LoRD | 0.0888 | 0.3195 | — | 0.516 |
| local_9 + LoRD | 0.0881 | 0.3027 | — | 0.489 |
| local_10 + LoRD | 0.0900 | 0.3230 | — | 0.522 |
| local_11 + LoRD | 0.1894 | 0.4826 | — | 0.780 |
| local_12 + LoRD | 0.0843 | 0.3095 | — | 0.500 |
| local_13 + LoRD | 0.0492 | 0.2396 | — | 0.387 |
| local_14 + LoRD | 0.0857 | 0.3051 | — | 0.493 |
| local_15 + LoRD | 0.0795 | 0.3036 | — | 0.491 |
| **Uniform Merge (soup)** | 0.0809 | 0.2922 | — | 0.472 |
| **Best-Single (Oracle) = local_11** | 0.1894 | 0.4826 | — | **0.780** |
| Ours — metamon_layer | 0.1029 | 0.3611 | — | 0.584 |
| Ours — metamon_cell | 0.1040 | 0.3609 | — | 0.583 |
| Ours — metamon_greedy | 0.0931 | 0.3467 | — | 0.560 |
| Ours — greedy_soup | 0.0895 | 0.3109 | — | 0.502 |
| (참고) All-query LoRD | 0.0926 | 0.3266 | — | 0.528 |

**읽는 법.** 생성에서 `local_11`(F 0.780)을 이긴 병합 arm 이 없습니다. 최고 병합은 `metamon_layer` 0.584 입니다. MLE 행은 `--fleet sft` 로 따로 돌려야 채워지고, KD 는 Target logit 이 필요해 black-box 에서 불가능합니다.

### E1-b — Target 응답(gold) 대비 (추출 충실도 본 지표)

| Method | BLEU-4 ↑ | ROUGE-L ↑ |
|---|---|---|
| __base__ | 0.0054 | 0.1040 |
| local_0 | 0.1062 | 0.3229 |
| local_1 | 0.1495 | 0.3898 |
| local_2 | 0.2279 | 0.5319 |
| local_3 | 0.0923 | 0.2893 |
| local_4 | 0.1541 | 0.3926 |
| local_5 | 0.1647 | 0.4086 |
| local_6 | 0.2241 | 0.5055 |
| local_7 | 0.1269 | 0.3652 |
| local_8 | 0.1561 | 0.3980 |
| local_9 | 0.1383 | 0.3688 |
| local_10 | 0.1550 | 0.3949 |
| local_11 | 0.3341 | 0.5974 |
| local_12 | 0.1389 | 0.3767 |
| local_13 | 0.0823 | 0.2831 |
| local_14 | 0.1360 | 0.3652 |
| local_15 | 0.1328 | 0.3607 |
| soup | 0.1461 | 0.3746 |
| greedy_soup | 0.1577 | 0.3920 |
| weighted_t1.0 | 0.1566 | 0.3938 |
| metamon_greedy | 0.1612 | 0.4268 |
| metamon_cell | 0.1814 | 0.4446 |
| metamon_layer | 0.1729 | 0.4375 |
| lord_all | 0.1573 | 0.4026 |

---

## E2 — Surrogate Dependency

Dependency = surrogate 선택을 바꿨을 때의 **분산**. 작을수록 덜 휘둘립니다.

| Method | avgBF Dep. ↓ | BLEU-4 Dep. ↓ | ROUGE-L Dep. ↓ | BERTScore-F1 Dep. ↓ |
|---|---|---|---|---|
| Single-surrogate (n=16) | 1.493e-03 | 1.125e-03 | 4.148e-03 | — |
| METAMON merged m=4 (n=4) | 3.209e-03 | 미측정 | 미측정 | — |
| (대조) union m=4 (n=4) | 2.299e-04 | 미측정 | 미측정 | — |
| **Reduction (%)** | **-114.9%** | — | — | — |

**불성립입니다.** `merged 2.41e-03 > union 1.73e-04` 이고 avgBF Reduction 이 -114.9% 로 오히려 늘었습니다.

병합 크기 곡선:

| m | 묶음 n | 분산 | 평균 | σ²/m 예측 |
|---|---|---|---|---|
| 1 | 16 | 1.493e-03 | 0.41406 | 1.493e-03 |
| 2 | 8 | 2.990e-03 | 0.42251 | 7.467e-04 |
| 4 | 4 | 3.209e-03 | 0.44605 | 3.734e-04 |
| 8 | 2 | 0.000e+00 | 0.48403 | 1.867e-04 |

단조가 아닙니다. m=2, m=4 가 m=1 보다 큽니다. 비-IID 조각이라 묶음의 m 개가 같은 분포에서 뽑힌 것이 아니어서 `σ²/m` 의 전제가 깨집니다. **m=8 은 표본이 2개(둘 다 0.48403)라 분산 0 은 보고하면 안 됩니다.**

> 설계 문서의 `METAMON (Leave-One-Out)` 행은 이번 실행에 `loo_k` arm 이 평가 목록에서 빠져 있어 비어 있습니다. 03 에서 다시 켜면 채워집니다.

---

## E3 — 동일 자원 예산 비교

| Method | Local 구성 | GPU-hours ↓ | Inference Params ↓ | avgBF ↑ | BLEU-4 ↑ | ROUGE-L ↑ |
|---|---|---|---|---|---|---|
| Single **Large** + LoRD | Large × 1 | — | — | — | — | — |
| Single Same-size + LoRD (`lord_all`) | 1.1B × 1, 전체 질의 | 미측정 | 1.1B (1×) | 0.47171 | 0.0926 | 0.3266 |
| Small Fleet **Ensemble** | 1.1B × 16 | 미측정 | 17.6B (16×) | 0.45947 | 미측정 | 미측정 |
| **METAMON (soup)** | 1.1B × 16 → θ′ | 미측정 | 1.1B (**1×**) | **0.50299** | 0.0809 | 0.2922 |
| **METAMON (weighted)** | 1.1B × 16 → θ′ | 미측정 | 1.1B (**1×**) | **0.50999** | 0.0888 | 0.3121 |

**이 표에서 가장 강한 줄.** 병합(1×) 0.50299 vs 앙상블(16×) 0.45947 — 추론 비용 1/16 로 +0.04352 더 좋습니다.

`Single Large` 행이 비어 있습니다. **크기 축이 실험에 없습니다.** '더 작은 모델로 큰 모델의 답을 낸다' 를 주장하려면 이 행이 있어야 합니다. Target 응답은 surrogate 와 무관하므로 추가 API 비용 없이 채울 수 있습니다.

GPU-hours 는 이번 실행에서 계측하지 않았습니다. 측정된 것은 02 기여도 4113s(1.14h) 뿐입니다.

---

## E4 — Ablation Study

| Method | avgBF ↑ | Dependency ↓ | BLEU-4 ↑ | ROUGE-L ↑ | BERTScore-F1 ↑ |
|---|---|---|---|---|---|
| Full METAMON | **미실행** | — | — | — | — |
| = 이번 실행의 `metamon_layer` (ω=1, A=[0.25]) | 0.44635 | — | 0.1029 | 0.3611 | — |
| w/o Contribution Selection — Uniform (`soup`) | 0.50299 | 3.209e-03 | 0.0809 | 0.2922 | — |
| w/o Contribution Selection — Random (`random_mean`) | 0.48050 | — | 미측정 | 미측정 | — |
| w/o Behavioral OT — Layer Ratio Mapping | 미구현 (동질 fleet 에서 불활성) | — | — | — | — |
| w/o Entropic Regularization (τ=0) | 미구현 | — | — | — | — |
| w/o Norm Matching (`soup_raw`) | 0.50218 | — | 0.0812 | 0.2919 | — |
| w/o Layer-wise Selection (`metamon_cell`) | 0.46072 | — | 0.1040 | 0.3609 | — |
| w/o Assembly Verification | 미실행 (관문은 통과함: 0.93823 < 1.92958) | — | — | — | — |
| w/o Weight Loss (λ=0) | 미구현 | — | — | — | — |
| w/o Soft Weight (ω=1) | 0.44635 (= 이번 실행) | — | 0.1029 | 0.3611 | — |
| (추가 대조) `shuffle_mean` — 점유율 유지, 위치만 섞음 | 0.48127 | — | 미측정 | 미측정 | — |

**Ablation 이 거꾸로 나왔습니다.**

- `w/o Contribution Selection (Uniform)` 0.50299 > `metamon_layer` 0.44635 — **구성요소를 빼면 좋아집니다.**
- `w/o Contribution Selection (Random)` 0.48050 도 `metamon_layer` 보다 높습니다.
- `metamon_layer − shuffle_mean = -0.03493 [-0.03822, -0.03169]` (음수) — 기여도가 고른 위치가 **그 위치를 무작위로 섞은 것보다 나쁩니다.**
- `shuffle_mean − random_mean = +0.00077 [-0.00294, +0.00445]` (불확실) — 한 모델에 몰아주는 효과도 없습니다.

**Full METAMON 은 이번에도 실행되지 않았습니다.** ω=1(균등)이고 A=[0.25] 1점이라 위 표의 `metamon_layer` 는 설계상 `w/o Soft Weight` 행과 같습니다. `--omega soft --alphas 0.125 0.25 0.5` 로 돌려야 Full 행이 생깁니다.

---

## E5 — Sensitivity Analysis

| Parameter | Tested Values | avgBF | Dependency |
|---|---|---|---|
| Soft weight temperature (weighted τ·spread) | T=0.1 | 0.47938 | 미측정 |
| Soft weight temperature (weighted τ·spread) | T=0.3 | 0.49763 | 미측정 |
| Soft weight temperature (weighted τ·spread) | T=1.0 | 0.50999 | 미측정 |
| Soft weight temperature (weighted τ·spread) | T=3.0 | 0.50891 | 미측정 |
| Soft weight temperature (weighted τ·spread) | T=10.0 | 0.50471 | 미측정 |
| Scale candidates (A) | [0.25] 1점만 | — | — |
| 배율 격자 (scales) | [0, .125, .25, .5, .75, 1, 1.5, 2.5, 4] | 전 arm 탐색함 | — |
| OT regularization τ | 미구현 | — | — |
| Weight loss coefficient λ | 미실행 (λ1=0.5 고정) | — | — |
| Mapping R² threshold | 미구현 | — | — |
| Number of Local Models (K) | 16 만 (이전 실행 8) | — | — |
| Query Count (X) | 4992 만 | — | — |

가중 평균 온도만 실제로 훑었습니다. 최적 T=1.0 에서 0.50999 로 `soup` 0.50299 를 +0.00700 [+0.00389, +0.01003] 이깁니다 — 이번 실행에서 **기여도 정보가 도움이 된 유일한 지점**입니다. 다만 T=1.0 은 사실상 균등에 가까운 쪽입니다.

---

## E6 — Framework 내부 분석

| Analysis | 결과 | 확인 목적 달성? |
|---|---|---|
| Layer Source Heatmap (칸 점유) | `[0, 3, 3, 0, 0, 10, 2, 0, 18, 3, 52, 58, 2, 0, 2, 1]` | △ 상위 2개가 110/154 칸 |
| Layer 단위 점유 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 1, 0, 0, 0]` | ✗ `local_10`+`local_11` 이 21/22 층 |
| PartialScore Heatmap | `02_contribution.json` 에 154×16 전부 저장됨 | 그릴 수 있음 |
| Confidence Distribution | 중앙값 5.576e-03, 칸-layer 일치 62.3% | 그릴 수 있음 |
| Mapping R² Distribution | 미구현 | ✗ |

PartialScore=0 인 칸 0/154, 중앙값 1.145e-02. 통합 검증 통과 (0.93823 < 1.92958).

다양성 관문: 칸별 코사인 중앙값 0.003~0.101 로 매우 낮습니다. Local 들의 Δw 가 거의 직교합니다 — 합칠 것이 있다는 뜻이고, 실제로 균등 평균이 잘 먹힌 이유이기도 합니다.

---

## (설계 문서에 없는 표) 전제 — 상보성

병합을 판정하기 전에 **병합할 것이 있었는지**. 이번 실행에서 처음으로 섰습니다.

| | 이전 실행 (`shard=iid`) | 이번 실행 (`shard=cluster`) |
|---|---|---|
| 조각 feat_cosine | ~0.95 | **0.435** |
| oracle (질의별 최고 Local) | — | 0.52584 |
| 최고 단일 | — | 0.45761 |
| **headroom** | +0.00096 | **+0.06823** |
| 최상위 Local 승률 | 85% | **15.6%** (완전 상보 = 6.2%) |
| 기여도 점유 최댓값 | 77/154 | 58/154 |

**headroom 회수율** (병합이 상보성의 몇 %를 가져왔나):

| arm | 회수율 |
|---|---|
| weighted | +76.8% |
| greedy_soup | +71.2% |
| soup | +66.5% |
| metamon_greedy | -7.7% |
| metamon_layer | -16.5% |

---

## 판정 요약

| 항목 | 판정 |
|---|---|
| 전제 (상보성) | **성립** — headroom +0.068, 승률 15.6% |
| 주장 1 · 확률 | **성립** — `greedy_soup − best_local(loss) +0.06385 [+0.0572, +0.0707]`, `− lord_all +0.03449` |
| 주장 1 · 생성 | **불성립** — `local_11` F 0.780 vs 최고 병합 0.584 |
| 주장 2 · 종속성 | **불성립** — 곡선 단조 아님, merged > union |
| 기여도 선택 (고유 기전) | **기각** — shuffle 보다 −0.035, random 과 차이 없음 |
| 병합 1× > 앙상블 16× | **성립** — +0.04352 |

### 확률과 생성이 거꾸로 갑니다

병합 arm 7개에서 avgBF 순위와 생성 F 순위의 **Spearman = −0.821**.

| arm | avgBF | 생성 F(RL) |
|---|---|---|
| weighted_t1.0 | 0.50999 | 0.504 |
| greedy_soup | 0.50620 | 0.502 |
| soup | 0.50299 | 0.472 |
| lord_all | 0.47171 | 0.528 |
| metamon_cell | 0.46072 | 0.583 |
| metamon_greedy | 0.45235 | 0.560 |
| metamon_layer | 0.44635 | 0.584 |

평균은 **주저하는** 모델을 만듭니다 — 참조 문장의 likelihood 는 높은데 greedy 로 뽑으면 밋밋합니다. 선택은 **단정하는** 모델을 만듭니다 — likelihood 는 낮은데 문장은 낫습니다. `eq:sim` 은 이 차이를 볼 수 없고, **LoRD Table 1 은 생성입니다.**
