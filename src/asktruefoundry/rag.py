from __future__ import annotations

from openai import APIStatusError

from asktruefoundry.config import Settings
from asktruefoundry.gateway import GatewayClient, GenerationStoppedError
from asktruefoundry.index import cosine_top_k, load_embeddings, load_records
from asktruefoundry.models import AnswerResult, ChunkRecord, RetrievedChunk


UNKNOWN_ANSWER = "I don't know based on TrueFoundry docs/blog."


class AskTrueFoundryRag:
    def __init__(
        self,
        settings: Settings,
        gateway: GatewayClient | None = None,
        records: list[ChunkRecord] | None = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway or GatewayClient(settings)
        self.records = records if records is not None else load_records(settings.index_path)
        self.embeddings = load_embeddings(settings.embeddings_path)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        query_embedding = self.gateway.embed_texts([question])[0]
        return cosine_top_k(
            query_embedding=query_embedding,
            records=self.records,
            embeddings=self.embeddings,
            top_k=top_k or self.settings.top_k,
            min_similarity=self.settings.min_similarity,
        )

    def answer(self, question: str, max_sources: int | None = None) -> AnswerResult:
        top_k = max_sources or self.settings.top_k
        retrieved: list[RetrievedChunk] = []
        try:
            retrieved = self.retrieve(question, top_k=top_k)
            if not retrieved:
                return AnswerResult(
                    status="no_evidence",
                    answer=UNKNOWN_ANSWER,
                    sources=[],
                    retrieved_chunks=[],
                )
            messages = build_messages(
                question,
                retrieved,
                max_context_chars_per_source=self.settings.max_context_chars_per_source,
            )
            answer = self.gateway.generate_answer(messages)
            if not answer:
                answer = UNKNOWN_ANSWER
            if is_unknown_answer(answer):
                return AnswerResult(
                    status="no_evidence",
                    answer=UNKNOWN_ANSWER,
                    sources=[],
                    retrieved_chunks=serialize_retrieved(retrieved),
                )
            return AnswerResult(
                status="ok",
                answer=ensure_sources(answer, retrieved),
                sources=dedupe_sources(retrieved),
                retrieved_chunks=serialize_retrieved(retrieved),
            )
        except APIStatusError as exc:
            if exc.status_code == 429:
                return AnswerResult(
                    status="rate_limited",
                    answer=(
                        "The TrueFoundry AI Gateway rate limit was reached. "
                        "Please wait a minute and retry."
                    ),
                    sources=[],
                    retrieved_chunks=[],
                )
            raise
        except GenerationStoppedError as exc:
            if exc.content.strip() and retrieved and not is_unknown_answer(exc.content):
                partial_answer = ensure_sources(exc.content.strip(), retrieved)
                return AnswerResult(
                    status="generation_stopped",
                    answer=(
                        f"{partial_answer}\n\n"
                        f"Note: the model stopped before a complete finish "
                        f"(finish_reason={exc.finish_reason}), so this answer may be incomplete."
                    ),
                    sources=dedupe_sources(retrieved),
                    retrieved_chunks=serialize_retrieved(retrieved),
                    error=exc.to_error(),
                )
            return AnswerResult(
                status="generation_stopped",
                answer=(
                    "The model stopped before producing a complete answer "
                    f"(finish_reason={exc.finish_reason}). This usually means "
                    "the selected model consumed the response budget, often with "
                    "reasoning tokens. Use a non-reasoning/smaller model or "
                    "increase the generation token budget."
                ),
                sources=[],
                retrieved_chunks=[],
                error=exc.to_error(),
            )


def build_messages(
    question: str,
    retrieved: list[RetrievedChunk],
    max_context_chars_per_source: int = 1800,
) -> list[dict[str, str]]:
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        record = item.record
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {i}] {record.title}",
                    f"URL: {record.url}",
                    f"Text: {_trim_context(record.text, max_context_chars_per_source)}",
                ]
            )
        )
    context = "\n\n---\n\n".join(context_blocks)
    system = (
        "You are AskTrueFoundry. Answer questions using only the supplied "
        "TrueFoundry docs/blog context. Be concise and practical. If the context "
        f"does not contain the answer, reply exactly: {UNKNOWN_ANSWER} "
        "Do not include sources when you do not know the answer. For supported "
        "answers, include source webpage URLs in the answer. Keep the answer under "
        "250 words unless the user explicitly asks for more detail."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _trim_context(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 15].rstrip() + " ... [trimmed]"


def dedupe_sources(retrieved: list[RetrievedChunk]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for item in retrieved:
        record = item.record
        if record.url in seen:
            continue
        seen.add(record.url)
        sources.append({"title": record.title, "url": record.url})
    return sources


def is_unknown_answer(answer: str) -> bool:
    normalized = " ".join(answer.strip().split())
    return normalized.startswith(UNKNOWN_ANSWER)


def ensure_sources(answer: str, retrieved: list[RetrievedChunk]) -> str:
    sources = dedupe_sources(retrieved)
    missing = [source for source in sources if source["url"] not in answer]
    if not missing:
        return answer
    source_lines = "\n".join(f"- {source['title']}: {source['url']}" for source in sources)
    return f"{answer.rstrip()}\n\nSources:\n{source_lines}"


def serialize_retrieved(retrieved: list[RetrievedChunk]) -> list[dict[str, str | float]]:
    return [
        {
            "id": item.record.id,
            "title": item.record.title,
            "url": item.record.url,
            "score": round(item.score, 4),
        }
        for item in retrieved
    ]
