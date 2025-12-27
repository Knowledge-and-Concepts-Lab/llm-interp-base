from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
from openai import OpenAI


@dataclass
class APIResult:
    text: str
    raw: Any = None
    usage: Optional[Dict[str, Any]] = None
    provider: str = ""
    model: str = ""
    response_id: Optional[str] = None


class APIWrapper:
    def __init__(self, provider: str, model: str, api_key: str, base_url: Optional[str] = None):
        self.provider = provider.lower().strip()
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url

        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url else OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError("provider must be 'openai' or 'gemini'")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_output_tokens: int = 256,
    ) -> APIResult:

        if self.provider == "openai":
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=int(max_output_tokens),
            )
            return APIResult(
                text=resp.output_text or "",
                raw=resp,
                usage=getattr(resp, "usage", None),
                provider="openai",
                model=self.model_name,
                response_id=getattr(resp, "id", None),
            )

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"max_output_tokens": int(max_output_tokens)},
        )
        return APIResult(
            text=resp.text or "",
            raw=resp,
            usage=getattr(resp, "usage_metadata", None),
            provider="gemini",
            model=self.model_name,
            response_id=getattr(resp, "id", None),
        )
