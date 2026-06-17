from __future__ import annotations

import argparse
import sys

from asktruefoundry.config import Settings
from asktruefoundry.crawler import crawl_truefoundry_report
from asktruefoundry.gateway import GatewayClient
from asktruefoundry.index import build_chunk_records, save_index


DEFAULT_MAX_PAGES = 1000
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_FETCH_RETRIES = 2
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_RETRIES = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the AskTrueFoundry local RAG index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Maximum pages to crawl.")
    parser.add_argument("--chunk-words", type=int, default=450, help="Words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=80, help="Chunk overlap in words.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print crawl progress after every N attempted pages. Use 0 to disable.",
    )
    parser.add_argument("--fetch-retries", type=int, default=DEFAULT_FETCH_RETRIES, help="Retries per page fetch.")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Number of chunks per AI Gateway embedding request.",
    )
    parser.add_argument(
        "--embedding-retries",
        type=int,
        default=DEFAULT_EMBEDDING_RETRIES,
        help="Retries per AI Gateway embedding batch.",
    )
    return parser


def _print_crawl_progress(done: int, total: int, kept: int, skipped: int, failed: int) -> None:
    print(
        f"[crawl] processed={done}/{total} kept={kept} skipped={skipped} failed={failed}",
        flush=True,
    )


def _print_embedding_progress(done: int, total: int) -> None:
    print(f"[embed] embedded={done}/{total} chunks", flush=True)


def run_ingestion(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    print(f"Discovering and fetching TrueFoundry pages (max_pages={args.max_pages})...")
    report = crawl_truefoundry_report(
        max_pages=args.max_pages,
        progress_every=args.progress_every,
        retries=args.fetch_retries,
        progress=_print_crawl_progress if args.progress_every > 0 else None,
    )
    pages = report.pages
    print(
        "Crawl summary: "
        f"discovered={report.discovered_count}, selected={report.selected_count}, "
        f"kept={len(report.pages)}, skipped_content_type={report.skipped_content_type}, "
        f"skipped_short_text={report.skipped_short_text}, failed={len(report.failed)}"
    )
    if report.failed:
        examples = ", ".join(f"{issue.url} ({issue.reason})" for issue in report.failed[:5])
        print(f"Fetch failures, first {min(5, len(report.failed))}: {examples}", file=sys.stderr)

    if not pages:
        print("No pages fetched; cannot build index.", file=sys.stderr)
        return 1

    records = build_chunk_records(
        pages,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    if not records:
        print("No chunks generated; cannot build index.", file=sys.stderr)
        return 1

    print(f"Fetched {len(pages)} pages and generated {len(records)} chunks.")
    print("Embedding chunks through TrueFoundry AI Gateway...")
    gateway = GatewayClient(settings)
    embeddings = gateway.embed_texts(
        [record.text for record in records],
        batch_size=args.embedding_batch_size,
        retries=args.embedding_retries,
        progress=_print_embedding_progress,
    )
    save_index(records, embeddings, settings.index_path, settings.embeddings_path)
    print(f"Wrote {settings.index_path}")
    print(f"Wrote {settings.embeddings_path}")
    return 0


def main() -> None:
    parser = build_parser()
    raise SystemExit(run_ingestion(parser.parse_args()))


if __name__ == "__main__":
    main()
