from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SourcePage:
    url: str
    title: str
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    url: str
    title: str
    text: str
    chunk_index: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=str(payload["id"]),
            url=str(payload["url"]),
            title=str(payload["title"]),
            text=str(payload["text"]),
            chunk_index=int(payload["chunk_index"]),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    record: ChunkRecord
    score: float


@dataclass(frozen=True)
class AnswerResult:
    status: str
    answer: str
    sources: list[dict[str, str]]
    retrieved_chunks: list[dict[str, str | float]]
    error: dict[str, Any] | None = None
