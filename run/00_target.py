# -*- coding: utf-8 -*-
"""실행 0. Target Model 에 질의하고 응답을 모은다.

    # 실제 API
    export OPENAI_API_KEY=...
    python run/00_target.py --out runs/gpt35 --provider openai \
        --model gpt-3.5-turbo-1106 --budget 2048

    # 로컬 victim (비용 0, 재현 가능)
    python run/00_target.py --out runs/llama3 --provider hf \
        --model meta-llama/Meta-Llama-3-8B-Instruct

    # 배관 점검용 (질의 0, 정답 문장 사용)
    python run/00_target.py --out runs/smoke --provider reference

응답은 두 곳에 남는다.
    <out>/target/cache.jsonl    질의-응답 원본. 중간에 끊겨도 산 것은 남는다.
    <out>/target/dataset.json   분할에 쓸 최종 데이터셋

주의. eq:sim 은 Target 응답에 모델이 부여하는 확률이다. 따라서 sel / check /
test 도 Target 응답이 있어야 한다. 총 질의 수 = n_train + n_sel + n_check +
n_test 다. 이 비용을 줄이려면 n_test 를 줄이는 게 아니라 n_train 을 줄인다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Pipeline.config import Config
from Pipeline.data import Splits, attach, build_queries, instruction, save_dataset
from Pipeline.modeling import load_tokenizer
from Pipeline.target import ResponseCache, build_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--provider", default="openai",
                    choices=["openai", "hf", "reference"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--budget", type=int, default=None,
                    help="새로 보낼 질의 상한. 캐시 적중은 세지 않는다")
    ap.add_argument("--subset", default="cs-en")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()

    cfg = Config(out_root=a.out, subset=a.subset, seed=a.seed,
                 target_provider=a.provider,
                 target_temperature=a.temperature,
                 target_max_tokens=a.max_tokens,
                 target_base_url=a.base_url)
    if a.model:
        cfg.target_model = a.model
    if a.budget is not None:
        cfg.query_budget = a.budget
    cfg.makedirs()

    tok = load_tokenizer(cfg)
    items = build_queries(cfg, tok)
    need = cfg.n_train + cfg.n_sel + cfg.n_check + cfg.n_test
    system = "Instruction: " + instruction(cfg)

    print("[Target]")
    print(f"  제공자        {cfg.target_provider}")
    print(f"  모델          {cfg.target_model if a.provider != 'reference' else '-'}")
    print(f"  system        {system}")
    print(f"  user 예시     {items[0]['src'][:70]}")
    print(f"  필요 질의     {need}  (train {cfg.n_train} / sel {cfg.n_sel} / "
          f"check {cfg.n_check} / test {cfg.n_test})")
    print(f"  신규 상한     {cfg.query_budget}")
    if cfg.target_provider == "reference":
        print("  *** 정답 문장을 Target 응답 대신 쓴다. 이 결과는 추출 충실도가 아니다.")

    cache = ResponseCache(cfg.cache_path)
    print(f"  캐시          {cfg.cache_path}  기존 {len(cache)}개")
    target = build_target(cfg, refs=[x["ref"] for x in items])
    resp = target.batch(system, [x["src"] for x in items], cache,
                        budget=cfg.query_budget)
    cache.close()

    full = attach(items, resp, cfg, tok)
    sp = Splits(cfg, full)
    sp.assert_clean()

    meta = {"provider": cfg.target_provider, "model": cfg.target_model,
            "system": system, "subset": cfg.subset, "seed": cfg.seed,
            "n": [cfg.n_train, cfg.n_sel, cfg.n_check, cfg.n_test],
            "queries_sent": target.sent, "cache_hits": target.cached,
            "query_hash": sp.query_hash(), "shard_hash": sp.shard_hash(),
            "temperature": cfg.target_temperature,
            "max_tokens": cfg.target_max_tokens}
    save_dataset(cfg.dataset_path, full, meta)

    print(f"\n[회계] 신규 질의 {target.sent}   캐시 적중 {target.cached}")
    print(f"  분할          {sp.summary()}")
    print(f"  저장          {cfg.dataset_path}")
    json.dump(meta, open(os.path.join(cfg.log_dir, "00_target.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
