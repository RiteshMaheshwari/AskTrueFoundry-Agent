from __future__ import annotations

import argparse
import json

from asktruefoundry.config import Settings
from asktruefoundry.rag import AskTrueFoundryRag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question against the TrueFoundry RAG index.")
    parser.add_argument("question", nargs="+", help="Question to ask AskTrueFoundry.")
    parser.add_argument("--max-sources", type=int, default=None, help="Maximum source chunks to retrieve.")
    parser.add_argument("--json", action="store_true", help="Print the full structured result as JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    question = " ".join(args.question)
    rag = AskTrueFoundryRag(Settings.from_env())
    result = rag.answer(question, max_sources=args.max_sources)
    if args.json:
        print(
            json.dumps(
                {
                    "status": result.status,
                    "answer": result.answer,
                    "sources": result.sources,
                    "retrieved_chunks": result.retrieved_chunks,
                    "error": result.error,
                },
                indent=2,
            )
        )
        return
    print(result.answer)


if __name__ == "__main__":
    main()
