from __future__ import annotations

import unittest
from pathlib import Path

import httpx
import numpy as np

from asktruefoundry.config import Settings
from asktruefoundry.crawler import discover_urls, is_allowed_url
from asktruefoundry.gateway import GenerationStoppedError
from asktruefoundry.index import cosine_top_k
from asktruefoundry.models import ChunkRecord, RetrievedChunk
from asktruefoundry.rag import AskTrueFoundryRag, UNKNOWN_ANSWER, build_messages, ensure_sources
from asktruefoundry.text import chunk_text, extract_text_from_html


class TextTests(unittest.TestCase):
    def test_extract_text_from_html_removes_script_content(self) -> None:
        title, body = extract_text_from_html(
            "<html><head><title>TF Docs</title><script>bad()</script></head>"
            "<body><h1>Deploy</h1><p>TrueFoundry service deployment.</p></body></html>"
        )

        self.assertEqual(title, "TF Docs")
        self.assertIn("TrueFoundry service deployment", body)
        self.assertNotIn("bad()", body)

    def test_chunk_text_uses_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_text(text, chunk_words=5, overlap_words=2)

        self.assertEqual(chunks, ["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9"])


class RetrievalTests(unittest.TestCase):
    def test_cosine_top_k_orders_by_similarity(self) -> None:
        records = [
            ChunkRecord("a", "https://www.truefoundry.com/docs/a", "A", "alpha", 0),
            ChunkRecord("b", "https://www.truefoundry.com/docs/b", "B", "beta", 0),
            ChunkRecord("c", "https://www.truefoundry.com/docs/c", "C", "gamma", 0),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )

        results = cosine_top_k(
            query_embedding=np.array([1.0, 0.0], dtype=np.float32),
            records=records,
            embeddings=embeddings,
            top_k=2,
            min_similarity=0.0,
        )

        self.assertEqual([item.record.id for item in results], ["a", "c"])


class CrawlerTests(unittest.TestCase):
    def test_discover_urls_continues_when_one_seed_fails(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/bad.txt"):
                return httpx.Response(500)
            return httpx.Response(
                200,
                text="Useful page https://www.truefoundry.com/docs/ai-gateway",
                headers={"content-type": "text/plain"},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        urls = discover_urls(
            client,
            seeds=(
                "https://www.truefoundry.com/bad.txt",
                "https://www.truefoundry.com/llms.txt",
            ),
        )

        self.assertEqual(urls, ["https://www.truefoundry.com/docs/ai-gateway"])

    def test_allowed_urls_cover_llms_txt_path_groups(self) -> None:
        self.assertTrue(is_allowed_url("https://www.truefoundry.com/vs/litellm"))
        self.assertTrue(is_allowed_url("https://www.truefoundry.com/solutions/finance"))
        self.assertTrue(is_allowed_url("https://www.truefoundry.com/agent-gateway"))
        self.assertFalse(is_allowed_url("https://www.truefoundry.com/login"))


class PromptTests(unittest.TestCase):
    def test_prompt_requires_unknown_answer(self) -> None:
        record = ChunkRecord(
            "a",
            "https://www.truefoundry.com/docs/ai-gateway",
            "AI Gateway",
            "AI Gateway routes model traffic.",
            0,
        )
        messages = build_messages("What is unrelated?", [RetrievedChunk(record, 0.9)])

        self.assertIn(UNKNOWN_ANSWER, messages[0]["content"])
        self.assertIn(record.url, messages[1]["content"])

    def test_prompt_trims_long_context(self) -> None:
        record = ChunkRecord(
            "a",
            "https://www.truefoundry.com/docs/ai-gateway",
            "AI Gateway",
            " ".join(["longcontext"] * 100),
            0,
        )
        messages = build_messages(
            "What is AI Gateway?",
            [RetrievedChunk(record, 0.9)],
            max_context_chars_per_source=80,
        )

        self.assertIn("[trimmed]", messages[1]["content"])
        self.assertLess(len(messages[1]["content"]), 400)

    def test_ensure_sources_appends_missing_urls(self) -> None:
        record = ChunkRecord(
            "a",
            "https://www.truefoundry.com/blog/example",
            "Example Blog",
            "TrueFoundry blog text.",
            0,
        )
        answer = ensure_sources("A concise answer.", [RetrievedChunk(record, 0.8)])

        self.assertIn("Sources:", answer)
        self.assertIn(record.url, answer)

    def test_unknown_answer_does_not_return_sources(self) -> None:
        record = ChunkRecord(
            "a",
            "https://www.truefoundry.com/blog/example",
            "Example Blog",
            "TrueFoundry blog text.",
            0,
        )
        settings = Settings(
            truefoundry_api_key="test",
            gateway_base_url="https://gateway.truefoundry.ai",
            generation_model="test-model",
            embedding_model="test-embedding",
            data_dir=Path("data"),
            index_path=Path("data/index.jsonl"),
            embeddings_path=Path("data/embeddings.npy"),
            top_k=1,
            min_similarity=0.0,
        )

        class FakeGateway:
            def embed_texts(self, texts):
                return np.array([[1.0, 0.0]], dtype=np.float32)

            def generate_answer(self, messages):
                return (
                    f"{UNKNOWN_ANSWER}\n\nSource webpage URLs:\n"
                    "- https://www.truefoundry.com/blog/example"
                )

        rag = AskTrueFoundryRag(settings, gateway=FakeGateway(), records=[record])
        rag.embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        result = rag.answer("unsupported question")

        self.assertEqual(result.status, "no_evidence")
        self.assertEqual(result.answer, UNKNOWN_ANSWER)
        self.assertEqual(result.sources, [])

    def test_generation_length_finish_reason_is_exposed(self) -> None:
        record = ChunkRecord(
            "a",
            "https://www.truefoundry.com/docs/ai-gateway",
            "AI Gateway",
            "TrueFoundry AI Gateway routes and monitors LLM requests.",
            0,
        )
        settings = Settings(
            truefoundry_api_key="test",
            gateway_base_url="https://gateway.truefoundry.ai",
            generation_model="test-model",
            embedding_model="test-embedding",
            data_dir=Path("data"),
            index_path=Path("data/index.jsonl"),
            embeddings_path=Path("data/embeddings.npy"),
            top_k=1,
            min_similarity=0.0,
        )

        class FakeGateway:
            def embed_texts(self, texts):
                return np.array([[1.0, 0.0]], dtype=np.float32)

            def generate_answer(self, messages):
                raise GenerationStoppedError(
                    finish_reason="length",
                    model="openai/reasoning-model",
                    content="AI Gateway routes model traffic.",
                )

        rag = AskTrueFoundryRag(settings, gateway=FakeGateway(), records=[record])
        rag.embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

        result = rag.answer("What is AI Gateway?")

        self.assertEqual(result.status, "generation_stopped")
        self.assertIn("finish_reason=length", result.answer)
        self.assertIn("AI Gateway routes model traffic.", result.answer)
        self.assertIn(record.url, result.answer)
        self.assertEqual(result.sources, [{"title": record.title, "url": record.url}])
        self.assertEqual(result.retrieved_chunks[0]["id"], record.id)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error["finish_reason"], "length")
        self.assertEqual(result.error["partial_content"], "AI Gateway routes model traffic.")


if __name__ == "__main__":
    unittest.main()
