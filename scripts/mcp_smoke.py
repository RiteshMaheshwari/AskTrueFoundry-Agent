from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _serialize_tool_result(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "is_error": getattr(result, "isError", False),
        "content": [],
    }
    for item in getattr(result, "content", []) or []:
        if hasattr(item, "text"):
            try:
                payload["content"].append(json.loads(item.text))
            except json.JSONDecodeError:
                payload["content"].append(item.text)
        else:
            payload["content"].append(str(item))
    return payload


async def run(url: str, question: str | None, max_sources: int) -> None:
    async with streamablehttp_client(url) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            if question is None:
                return

            result = await session.call_tool(
                "ask_truefoundry",
                arguments={"question": question, "max_sources": max_sources},
            )
            print(json.dumps(_serialize_tool_result(result), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the local AskTrueFoundry MCP server.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp", help="MCP streamable HTTP URL.")
    parser.add_argument("--question", default=None, help="Optional question to call ask_truefoundry.")
    parser.add_argument("--max-sources", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args.url, args.question, args.max_sources))


if __name__ == "__main__":
    main()
