# -*- coding: utf-8 -*-
"""Target Model (victim). 공격자가 관측할 수 있는 것은 응답뿐이다.

LoRD-MEA 원본 프로토콜을 따른다.

    training_data_collecting_openai.chatWithOpenAI_APIs
        messages = [{"role": "system", "content": "Instruction: " + pp},
                    {"role": "user",   "content": q}]

    wmt_process.load_wmt_datals
        pp("cs-en") = "Translate the sentence from Czech to English Please."
        local prompt = f"Instruction: {pp} User: {x} Assistant: "

질의는 돈이다. 세 가지를 지킨다.
    1. 디스크 캐시. 같은 (모델, system, user) 는 두 번 묻지 않는다.
    2. 질의 예산. 상한을 넘으면 던지지 않고 중단한다.
    3. 회계. 실제로 보낸 질의 수를 기록한다. LoRD 의 핵심 주장이 질의 효율이다.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

TASK_PROMPT: Dict[str, str] = {
    "cs-en": "Translate the sentence from Czech to English Please.",
    "de-en": "Translate the sentence from German to English Please.",
    "fi-en": "Translate the sentence from Finnish to English Please.",
    "ro-en": "Translate the sentence from Romanian to English Please.",
    "ru-en": "Translate the sentence from Russian to English Please.",
    "tr-en": "Translate the sentence from Turkish to English Please.",
}


def _key(model: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    for s in (model, system, user):
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class ResponseCache:
    """JSONL 추가 기록. 중간에 끊겨도 이미 산 응답은 남는다."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.mem: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        self.mem[r["key"]] = r["response"]
                    except json.JSONDecodeError:
                        continue
        self.fh = open(path, "a", encoding="utf-8")

    def get(self, k: str) -> Optional[str]:
        return self.mem.get(k)

    def put(self, k: str, model: str, system: str, user: str, resp: str) -> None:
        self.mem[k] = resp
        self.fh.write(json.dumps(
            {"key": k, "model": model, "system": system, "user": user,
             "response": resp}, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()

    def __len__(self) -> int:
        return len(self.mem)


class TargetModel:
    """공통 인터페이스. respond 는 응답 문자열만 돌려준다."""

    name = "target"

    def respond(self, system: str, user: str) -> str:
        raise NotImplementedError

    # 회계
    sent = 0
    cached = 0

    def batch(self, system: str, users: Sequence[str], cache: ResponseCache,
              budget: Optional[int] = None, log=print) -> List[str]:
        out: List[str] = []
        for i, u in enumerate(users):
            k = _key(self.name, system, u)
            hit = cache.get(k)
            if hit is not None:
                self.cached += 1
                out.append(hit)
                continue
            if budget is not None and self.sent >= budget:
                raise SystemExit(
                    f"질의 예산 {budget} 을 초과했다. {i}/{len(users)} 에서 중단. "
                    f"캐시에는 {len(cache)} 개가 남아 있다.")
            r = self.respond(system, u)
            cache.put(k, self.name, system, u, r)
            self.sent += 1
            out.append(r)
            if (i + 1) % 50 == 0:
                log(f"    {i+1}/{len(users)}  신규 {self.sent}  캐시 {self.cached}")
        return out


class ReferenceTarget(TargetModel):
    """데이터셋의 정답 문장을 Target 응답 대신 쓴다.

    질의 비용이 0 이라 배관을 점검할 때 쓴다. 이걸로 나온 수치는
    '추출 충실도' 가 아니다. 보고할 때 반드시 그렇게 적는다.
    """

    name = "reference"

    def __init__(self, refs: Sequence[str]):
        self.refs = list(refs)
        self._i = 0

    def respond(self, system: str, user: str) -> str:
        r = self.refs[self._i]
        self._i += 1
        return r

    def batch(self, system, users, cache, budget=None, log=print) -> List[str]:
        self._i = 0
        return [self.respond(system, u) for u in users]


class OpenAITarget(TargetModel):
    """OpenAI 호환 API. 키는 환경변수에서만 읽는다.

    base_url 을 주면 같은 스키마를 쓰는 다른 제공자에도 붙는다.
    """

    def __init__(self, model: str = "gpt-3.5-turbo-1106",
                 temperature: float = 1.0, max_tokens: int = 256,
                 base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 retries: int = 5, backoff: float = 2.0, sleep: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise SystemExit("openai 패키지가 필요하다: pip install openai") from e
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"환경변수 {api_key_env} 가 비어 있다.")
        self.client = OpenAI(api_key=key, base_url=base_url) if base_url \
            else OpenAI(api_key=key)
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep

    def respond(self, system: str, user: str) -> str:
        last = None
        for a in range(self.retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if self.sleep:
                    time.sleep(self.sleep)
                return (res.choices[0].message.content or "").strip()
            except Exception as e:          # rate limit, 일시 오류
                last = e
                time.sleep(self.backoff ** a)
        raise SystemExit(f"API 호출이 {self.retries} 번 실패했다: {last}")


class HFTarget(TargetModel):
    """로컬 HuggingFace 모델을 victim 으로 쓴다.

    API 비용 없이 실제 '모델 응답' 으로 추출을 돌릴 수 있다.
    재현 가능한 본실험을 만들 때 이쪽이 낫다.
    """

    def __init__(self, model_id: str, device: str = "cuda",
                 temperature: float = 1.0, top_p: float = 0.95,
                 max_new_tokens: int = 256, dtype: str = "bfloat16",
                 chat_template: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.chat_template = chat_template and (
            getattr(self.tok, "chat_template", None) is not None)

    def _render(self, system: str, user: str) -> str:
        if self.chat_template:
            return self.tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        return f"Instruction: {system} User: {user} Assistant: "

    def respond(self, system: str, user: str) -> str:
        text = self._render(system, user)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5), top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


def build_target(cfg, refs: Optional[Sequence[str]] = None) -> TargetModel:
    p = cfg.target_provider
    if p == "reference":
        if refs is None:
            raise SystemExit("reference Target 은 정답 문장이 필요하다.")
        return ReferenceTarget(refs)
    if p == "openai":
        return OpenAITarget(model=cfg.target_model,
                            temperature=cfg.target_temperature,
                            max_tokens=cfg.target_max_tokens,
                            base_url=cfg.target_base_url or None)
    if p == "hf":
        return HFTarget(cfg.target_model, max_new_tokens=cfg.target_max_tokens,
                        temperature=cfg.target_temperature)
    raise SystemExit(f"알 수 없는 target_provider: {p}")
