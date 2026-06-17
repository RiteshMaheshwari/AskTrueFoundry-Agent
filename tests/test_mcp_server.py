from __future__ import annotations

import unittest
from unittest.mock import patch

from asktruefoundry.mcp_server import ask_truefoundry
from asktruefoundry.models import AnswerResult


class McpServerTests(unittest.TestCase):
    def test_empty_question_returns_structured_error(self) -> None:
        result = ask_truefoundry("")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["retrieved_chunks"], [])

    def test_tool_returns_rag_result_shape(self) -> None:
        class FakeRag:
            def answer(self, question: str, max_sources: int | None = None) -> AnswerResult:
                return AnswerResult(
                    status="ok",
                    answer=f"answer for {question}",
                    sources=[{"title": "Docs", "url": "https://www.truefoundry.com/docs"}],
                    retrieved_chunks=[{"id": "c1", "title": "Docs", "url": "https://www.truefoundry.com/docs", "score": 0.9}],
                )

        with patch("asktruefoundry.mcp_server._rag", return_value=FakeRag()):
            result = ask_truefoundry("What is TrueFoundry?", max_sources=2)

        self.assertEqual(result["status"], "ok")
        self.assertIn("answer for", result["answer"])
        self.assertEqual(result["sources"][0]["url"], "https://www.truefoundry.com/docs")
        self.assertEqual(result["retrieved_chunks"][0]["id"], "c1")
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
