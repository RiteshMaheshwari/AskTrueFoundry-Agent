from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import sleep
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from asktruefoundry.models import SourcePage
from asktruefoundry.text import extract_text_from_html


USER_AGENT = "AskTrueFoundry crawler/0.1"
DEFAULT_SEEDS = (
    "https://www.truefoundry.com/llms.txt",
    "https://www.truefoundry.com/sitemap.xml",
)
RETRY_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

ALLOWED_HOSTS = {"www.truefoundry.com", "docs.truefoundry.com"}
ALLOWED_PATH_PATTERNS = (
    re.compile(r"^/docs(/|$)"),
    re.compile(r"^/blog(/|$)"),
    re.compile(r"^/compare(/|$)"),
    re.compile(r"^/vs(/|$)"),
    re.compile(r"^/solutions(/|$)"),
    re.compile(r"^/ai-gateway(/|$)"),
    re.compile(r"^/mcp-gateway(/|$)"),
    re.compile(r"^/agent-gateway(/|$)"),
    re.compile(r"^/tracing(/|$)"),
    re.compile(r"^/models(/|$)"),
    re.compile(r"^/resources(/|$)"),
    re.compile(r"^/glossary(/|$)"),
    re.compile(r"^/product-tour(/|$)"),
    re.compile(r"^/pricing(/|$)"),
    re.compile(r"^/why-truefoundry(/|$)"),
    re.compile(r"^/case-study(/|$)"),
    re.compile(r"^/case-studies(/|$)"),
)
SKIP_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".mp4",
)


@dataclass(frozen=True)
class FetchIssue:
    url: str
    reason: str


@dataclass
class CrawlReport:
    pages: list[SourcePage]
    discovered_count: int
    selected_count: int
    skipped_content_type: int = 0
    skipped_short_text: int = 0
    failed: list[FetchIssue] = field(default_factory=list)

    @property
    def attempted_count(self) -> int:
        return self.selected_count


ProgressCallback = Callable[[int, int, int, int, int], None]


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    parsed = parsed._replace(fragment="", query="")
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", "", ""))


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() not in ALLOWED_HOSTS:
        return False
    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        return False
    return any(pattern.search(parsed.path) for pattern in ALLOWED_PATH_PATTERNS)


def _extract_urls_from_text(text: str, base_url: str) -> set[str]:
    urls = set(re.findall(r"https?://[^\s)'\"<>]+", text))
    urls.update(urljoin(base_url, match) for match in re.findall(r"href=[\"']([^\"']+)[\"']", text))
    return {normalize_url(url) for url in urls if is_allowed_url(normalize_url(url))}


def _extract_urls_from_sitemap(xml_text: str) -> set[str]:
    urls: set[str] = set()
    root = ET.fromstring(xml_text)
    for node in root.iter():
        if node.tag.endswith("loc") and node.text:
            url = normalize_url(node.text)
            if is_allowed_url(url):
                urls.add(url)
    return urls


def discover_urls(client: httpx.Client, seeds: Iterable[str] = DEFAULT_SEEDS) -> list[str]:
    discovered: set[str] = set()
    for seed in seeds:
        try:
            response = _get_with_retries(
                client=client,
                url=seed,
                retries=2,
                backoff_seconds=0.8,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        content_type = response.headers.get("content-type", "")
        if "xml" in content_type or seed.endswith(".xml"):
            try:
                discovered.update(_extract_urls_from_sitemap(response.text))
                continue
            except ET.ParseError:
                pass
        discovered.update(_extract_urls_from_text(response.text, seed))
    return sorted(discovered)


def _get_with_retries(
    client: httpx.Client,
    url: str,
    retries: int,
    backoff_seconds: float,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code in RETRY_STATUS_CODES and attempt < retries:
                sleep(backoff_seconds * (2**attempt))
                continue
            return response
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < retries:
                sleep(backoff_seconds * (2**attempt))
                continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch URL after retries: {url}")


def fetch_pages(
    urls: Iterable[str],
    timeout_seconds: float = 20.0,
    retries: int = 2,
    backoff_seconds: float = 0.8,
    progress_every: int = 25,
    progress: ProgressCallback | None = None,
) -> CrawlReport:
    url_list = list(urls)
    pages: list[SourcePage] = []
    failed: list[FetchIssue] = []
    skipped_content_type = 0
    skipped_short_text = 0
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_seconds) as client:
        for index, url in enumerate(url_list, start=1):
            try:
                response = _get_with_retries(
                    client=client,
                    url=url,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            except httpx.HTTPError as exc:
                failed.append(FetchIssue(url=url, reason=exc.__class__.__name__))
                if progress and progress_every > 0 and index % progress_every == 0:
                    progress(index, len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))
                continue

            if response.status_code >= 400:
                failed.append(FetchIssue(url=url, reason=f"HTTP {response.status_code}"))
                if progress and progress_every > 0 and index % progress_every == 0:
                    progress(index, len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))
                continue
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                skipped_content_type += 1
                if progress and progress_every > 0 and index % progress_every == 0:
                    progress(index, len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))
                continue
            title, text = extract_text_from_html(response.text)
            if len(text.split()) < 80:
                skipped_short_text += 1
                if progress and progress_every > 0 and index % progress_every == 0:
                    progress(index, len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))
                continue
            pages.append(SourcePage(url=normalize_url(str(response.url)), title=title or url, text=text))

            if progress and progress_every > 0 and index % progress_every == 0:
                progress(index, len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))

    if progress and progress_every > 0 and url_list and len(url_list) % progress_every != 0:
        progress(len(url_list), len(url_list), len(pages), skipped_content_type + skipped_short_text, len(failed))

    return CrawlReport(
        pages=pages,
        discovered_count=len(url_list),
        selected_count=len(url_list),
        skipped_content_type=skipped_content_type,
        skipped_short_text=skipped_short_text,
        failed=failed,
    )


def crawl_truefoundry_report(
    max_pages: int | None = None,
    progress_every: int = 25,
    retries: int = 2,
    progress: ProgressCallback | None = None,
) -> CrawlReport:
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=20.0) as client:
        urls = discover_urls(client)
    discovered_count = len(urls)
    if max_pages is not None:
        urls = urls[:max_pages]
    report = fetch_pages(
        urls,
        retries=retries,
        progress_every=progress_every,
        progress=progress,
    )
    report.discovered_count = discovered_count
    report.selected_count = len(urls)
    return report


def crawl_truefoundry(max_pages: int | None = None) -> list[SourcePage]:
    return crawl_truefoundry_report(max_pages=max_pages).pages
