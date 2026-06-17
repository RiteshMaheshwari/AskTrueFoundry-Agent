from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from time import sleep
from typing import Any

import numpy as np
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from asktruefoundry.config import Settings


APP_NAME = "asktruefoundry"
EmbeddingProgressCallback = Callable[[int, int], None]


class GenerationStoppedError(RuntimeError):
    def __init__(
        self,
        *,
        finish_reason: str | None,
        model: str | None,
        content: str,
    ) -> None:
        reason = finish_reason or "unknown"
        super().__init__(f"Gateway generation stopped before a complete answer: finish_reason={reason}")
        self.finish_reason = finish_reason
        self.model = model
        self.content = content

    def to_error(self) -> dict[str, Any]:
        return {
            "type": "generation_stopped",
            "finish_reason": self.finish_reason,
            "model": self.model,
            "partial_content": self.content,
        }


class GatewayClient:
    """Small wrapper around the TrueFoundry AI Gateway OpenAI-compatible API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.truefoundry_api_key,
            base_url=settings.gateway_base_url,
        )

    def _headers(self, traffic: str) -> dict[str, str]:
        metadata = {"application": APP_NAME, "traffic": traffic}
        return {
            "X-TFY-METADATA": json.dumps(metadata),
            "X-TFY-LOGGING-CONFIG": json.dumps({"enabled": True}),
        }

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        progress: EmbeddingProgressCallback | None = None,
    ) -> np.ndarray:
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if not batch:
                continue
            response = self._create_embeddings_with_retries(
                batch=batch,
                retries=retries,
                backoff_seconds=backoff_seconds,
            )
            vectors.extend(item.embedding for item in response.data)
            if progress:
                progress(min(start + len(batch), total), total)
        return np.array(vectors, dtype=np.float32)

    def _create_embeddings_with_retries(
        self,
        batch: Sequence[str],
        retries: int,
        backoff_seconds: float,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self.client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=list(batch),
                    extra_headers=self._headers("embedding"),
                )
            except (APIConnectionError, APIError, APIStatusError) as exc:
                last_error = exc
                if attempt >= retries or not _is_retryable_gateway_error(exc):
                    raise
                sleep(backoff_seconds * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Embedding request failed without an exception")

    def generate_answer(self, messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.generation_model,
            messages=messages,
            max_tokens=max_tokens or self.settings.generation_max_tokens,
            extra_headers=self._headers("generation"),
        )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        finish_reason = choice.finish_reason
        if finish_reason not in {None, "stop"}:
            raise GenerationStoppedError(
                finish_reason=finish_reason,
                model=getattr(response, "model", None),
                content=content,
            )
        return content


def _is_retryable_gateway_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return isinstance(exc, APIConnectionError)
    return status_code in {408, 409, 425, 429} or status_code >= 500
