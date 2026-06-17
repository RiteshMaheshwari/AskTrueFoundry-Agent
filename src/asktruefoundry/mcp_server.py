from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP
from openai import APIStatusError

from asktruefoundry.config import Settings
from asktruefoundry.rag import AskTrueFoundryRag


SERVER_NAME = "asktruefoundry"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MCP_PATH = "/mcp"


mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "AskTrueFoundry answers questions about TrueFoundry using only the "
        "indexed TrueFoundry docs, website, and blog corpus. Answers include "
        "source webpage URLs when supported by retrieved evidence."
    ),
    host=os.getenv("ASKTF_MCP_HOST", DEFAULT_HOST),
    port=int(os.getenv("ASKTF_MCP_PORT", str(DEFAULT_PORT))),
    streamable_http_path=os.getenv("ASKTF_MCP_PATH", DEFAULT_MCP_PATH),
    stateless_http=True,
)


@lru_cache(maxsize=1)
def _rag() -> AskTrueFoundryRag:
    return AskTrueFoundryRag(Settings.from_env())


@mcp.tool(
    name="ask_truefoundry",
    description=(
        "Answer a question about TrueFoundry using only the indexed "
        "TrueFoundry docs, website, and blog corpus. Returns the answer, "
        "source URLs, status, and retrieved chunk metadata."
    ),
)
def ask_truefoundry(question: str, max_sources: int = 4) -> dict[str, Any]:
    if not question.strip():
        return {
            "status": "error",
            "answer": "Question must not be empty.",
            "sources": [],
            "retrieved_chunks": [],
        }

    try:
        result = _rag().answer(question=question, max_sources=max_sources)
        return {
            "status": result.status,
            "answer": result.answer,
            "sources": result.sources,
            "retrieved_chunks": result.retrieved_chunks,
            "error": result.error,
        }
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "answer": (
                "Local RAG index is missing. Run "
                "`python scripts/ingest.py --max-pages 1000` before calling "
                "`ask_truefoundry`."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "error": str(exc),
        }
    except APIStatusError as exc:
        if exc.status_code == 429:
            return {
                "status": "rate_limited",
                "answer": (
                    "The TrueFoundry AI Gateway rate limit was reached. "
                    "Please wait a minute and retry."
                ),
                "sources": [],
                "retrieved_chunks": [],
            }
        return {
            "status": "error",
            "answer": f"TrueFoundry AI Gateway returned HTTP {exc.status_code}.",
            "sources": [],
            "retrieved_chunks": [],
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "AskTrueFoundry failed while processing the request.",
            "sources": [],
            "retrieved_chunks": [],
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
