from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asktruefoundry.models import ChunkRecord, RetrievedChunk, SourcePage
from asktruefoundry.text import chunk_text


def build_chunk_records(
    pages: list[SourcePage],
    chunk_words: int = 450,
    overlap_words: int = 80,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for page_index, page in enumerate(pages):
        chunks = chunk_text(page.text, chunk_words=chunk_words, overlap_words=overlap_words)
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    id=f"p{page_index:05d}-c{chunk_index:03d}",
                    url=page.url,
                    title=page.title,
                    text=chunk,
                    chunk_index=chunk_index,
                )
            )
    return records


def save_index(records: list[ChunkRecord], embeddings: np.ndarray, index_path: Path, embeddings_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    with index_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    np.save(embeddings_path, embeddings.astype(np.float32))


def load_records(index_path: Path) -> list[ChunkRecord]:
    if not index_path.exists():
        raise FileNotFoundError(f"RAG index not found: {index_path}")
    records: list[ChunkRecord] = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(ChunkRecord.from_json(json.loads(line)))
    return records


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path).astype(np.float32)


def cosine_top_k(
    query_embedding: np.ndarray,
    records: list[ChunkRecord],
    embeddings: np.ndarray,
    top_k: int,
    min_similarity: float,
) -> list[RetrievedChunk]:
    if len(records) != len(embeddings):
        raise ValueError(
            f"Index mismatch: {len(records)} records but {len(embeddings)} embeddings"
        )
    if len(records) == 0:
        return []

    query = query_embedding.astype(np.float32)
    query_norm = np.linalg.norm(query)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    denominator = emb_norms * query_norm
    denominator[denominator == 0] = 1e-12
    scores = embeddings @ query / denominator
    ordered = np.argsort(scores)[::-1]

    results: list[RetrievedChunk] = []
    for idx in ordered:
        score = float(scores[idx])
        if score < min_similarity:
            continue
        results.append(RetrievedChunk(record=records[int(idx)], score=score))
        if len(results) >= top_k:
            break
    return results
