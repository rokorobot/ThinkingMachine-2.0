import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import openai

class LLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

class VLLMClient(LLMClient):
    def __init__(self, base_url: str, model: str):
        # vLLM is OpenAI-compatible usually, so we can reuse the OpenAI client with a different base_url
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

def get_llm_client() -> LLMClient:
    # Simple factory
    if os.getenv("USE_VLLM", "false").lower() == "true":
        return VLLMClient(base_url=os.getenv("VLLM_URL"), model=os.getenv("VLLM_MODEL"))
    return OpenAIClient()
