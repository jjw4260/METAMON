# -*- coding: utf-8 -*-
"""이기종 fleet 용 Common Weight Space Projection.

    roles    구조마다 다른 선형 계층을 7 역할(rho)로 맞춘다 (Phi-3 의 합쳐진 행렬 분리 포함)
    spans    tokenizer 가 달라도 같은 단어 구간 c 의 activation 을 짝짓는다
    ot       eq:alignment_cost, eq:ot_alignment, Layer Ratio 대조군
    ridge    eq:input_map, eq:output_map 과 R^2
"""
