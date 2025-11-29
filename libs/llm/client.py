from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

try:
    import openai
except ImportError:
    openai = None


@dataclass
class LLMConfig:
    backend: str                  # "openai", "tgi", "vllm"
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512


class LLMClient:
    """
    Unified wrapper around:
      - OpenAI Chat Completions API
      - Local TGI/vLLM HTTP server (OpenAI-compatible or custom)
    """

    def __init__(self, config: LLMConfig):
        self.cfg = config

        if self.cfg.backend == "openai" and openai is None:
            raise RuntimeError("openai package not installed, run `pip install openai`")

        if self.cfg.backend == "openai":
            openai.api_key = self.cfg.api_key
            if self.cfg.api_base:
                openai.base_url = self.cfg.api_base

    @classmethod
    def from_env(cls) -> "LLMClient":
        backend = os.getenv("LLM_BACKEND", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-4o")
        api_base = os.getenv("LLM_API_BASE")  # e.g. https://api.openai.com/v1 OR http://tgi:8080/v1
        api_key = os.getenv("LLM_API_KEY")    # for OpenAI or any server that needs it

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))

        return cls(
            LLMConfig(
                backend=backend,
                model=model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    def chat(self, messages: List[Dict[str, str]], override_model: Optional[str] = None) -> str:
        model = override_model or self.cfg.model
        if self.cfg.backend == "openai":
            return self._chat_openai(messages, model)
        elif self.cfg.backend in ("tgi", "vllm"):
            return self._chat_tgi_style(messages, model)
        else:
            raise ValueError(f"Unsupported LLM backend: {self.cfg.backend}")

    # -------- OpenAI backend --------

    def _chat_openai(self, messages: List[Dict[str, str]], model: str) -> str:
        # Using new OpenAI client style
        resp = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content

    # -------- TGI / vLLM backend (HTTP) --------

    def _chat_tgi_style(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Assumes an OpenAI-compatible /v1/chat/completions endpoint.
        Many vLLM & TGI setups now mimic this.
        For pure TGI you might convert `messages` to a single prompt string.
        """
        if not self.cfg.api_base:
            raise RuntimeError("LLM_API_BASE must be set for TGI/vLLM backends")

        url = self.cfg.api_base.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
