# -*- coding: utf-8 -*-
"""METAMON
Model Extraction via Target-driven Aggregation of Multiple Open-source
Neural-surrogates.

파트 구성
    config        하이퍼파라미터와 설정 해시 관문
    data          질의 구성, 분할, 학습-평가 중복 검사
    modeling      모델 인스턴스 하나와 7 종 역할 가중치 공간
    metrics       eq:mean_log_probability / sim / avgBF / weighted_loss /
                  soft_weight / dependency / paired bootstrap
    lord          LoRD (LoRD-MEA 원본 이식: 토큰 단위 log_clip, swap, cold,
                  period break)
    sft           대조군 fleet
    fleet         fleet 준비와 생존 관문, 저장/복원 검증
    weightspace   공통 가중치 공간(위치별 norm 정규화)과 다양성 관문
    contribution  eq:perturbed_loss / partial_score / layer_selection /
                  representative_update / assembly_verification
    aggregate     metamon / soup / random / shuffle / weighted / leave-one-out
    evaluate      배율 격자(check) 와 최종 비교(test)
    dependency    Surrogate Dependency 와 출력 앙상블 baseline
"""
__version__ = "0.1.0"
