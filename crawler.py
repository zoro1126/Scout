"""
spec_crawler.py
===============
Dynamic Web Crawler with Structured JSONL Output
Implementation strictly based on "Specs for Crawler.pdf"

Architecture:
    Config -> CrawlerConfig
    URL handling -> URLManager
    Robots.txt -> RobotsManager
    Fetching -> HttpFetcher + BrowserFetcher
    Dynamic detection -> DynamicDetector
    Content extraction -> ContentExtractor
    Text chunking -> TextChunker
    JSONL output -> OutputWriter
    Worker orchestration -> CrawlerWorker
    Entry point -> Crawler (main controller)

Compliance:
    - OOP (every concern is a class)
    - PEP8 (4-space indent, docstrings, type hints, max ~99 chars/line)
    - Modular (each class is independently testable)

Dependencies:
    pip install aiohttp beautifulsoup4 playwright lxml tiktoken
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# Third-party (install via pip)
import aiohttp
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Optional browser backends – imported lazily inside BrowserFetcher
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("spec_crawler")


# ===========================================================================
# 1. Configuration  (Section 10 of spec)
# ===========================================================================

@dataclass
class CrawlerConfig:
    """
    All configurable parameters for the crawler.

    Environment variables take precedence over default values.
    Usage::

        cfg = CrawlerConfig.from_env()
    """

    start_url: str = "https://example.com"
    max_pages: int = 10_000              # effectively unlimited
    concurrency: int = 32               # 32 parallel async workers
    use_browser: bool = False           # auto-upgraded per page if JS detected
    browser_type: str = "playwright"
    request_timeout: int = 20           # seconds per HTTP request
    page_timeout: int = 45000          # ms: browser page load timeout
    output_path: str = "crawl_output.jsonl"
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    rate_limit_delay: float = 0.1       # minimal politeness delay
    max_retries: int = 5               # aggressive retry on failure
    user_agents: List[str] = field(default_factory=lambda: [
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
         "AppleWebKit/537.36 (KHTML, like Gecko) "
         "Chrome/122.0.0.0 Safari/537.36"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) "
         "AppleWebKit/605.1.15 (KHTML, like Gecko) "
         "Version/17.2 Safari/605.1.15"),
        ("Mozilla/5.0 (X11; Linux x86_64) "
         "Gecko/20100101 Firefox/123.0"),
    ])

    @classmethod
    def from_env(cls) -> "CrawlerConfig":
        """Build config from environment variables (12-factor style)."""
        use_browser_raw = os.getenv("USE_BROWSER", "false").lower()
        return cls(
            start_url=os.getenv("START_URL", cls.__dataclass_fields__["start_url"].default),
            max_pages=int(os.getenv("MAX_PAGES", str(cls.__dataclass_fields__["max_pages"].default))),
            concurrency=int(os.getenv("CONCURRENCY", str(cls.__dataclass_fields__["concurrency"].default))),
            use_browser=use_browser_raw in ("1", "true", "yes"),
            browser_type=os.getenv("BROWSER_TYPE", "playwright"),
            request_timeout=int(
                os.getenv("REQUEST_TIMEOUT",
                          str(cls.__dataclass_fields__["request_timeout"].default))
            ),
            page_timeout=int(
                os.getenv("PAGE_TIMEOUT",
                          str(cls.__dataclass_fields__["page_timeout"].default))
            ),
            output_path=os.getenv("OUTPUT_PATH", "crawl_output.jsonl"),
        )

    def random_user_agent(self) -> str:
        """Return a random UA from the pool."""
        return random.choice(self.user_agents)


# ===========================================================================
# 2. URL Manager  (Section 3.1 of spec)
# ===========================================================================

class URLManager:
    """
    Manages URL normalization, de-duplication, and domain restriction.

    Maintains:
        - ``seen``     : already-visited URLs
        - ``scheduled``: URLs queued but not yet visited
        - ``queue``    : asyncio.Queue for worker consumption

    All URL operations are domain-locked unless ``allow_subdomains=True``.
    """

    # Non-HTML extensions to skip
    _SKIP_EXTENSIONS: Set[str] = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".ico", ".mp4", ".mp3", ".avi", ".mov", ".pdf",
        ".zip", ".rar", ".gz", ".exe", ".dmg",
        ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    }

    def __init__(
        self,
        start_url: str,
        allow_subdomains: bool = False,
        restrict_path: Optional[str] = None,
    ) -> None:
        parsed = urllib.parse.urlparse(start_url)
        self.base_domain: str = parsed.netloc
        self.base_scheme: str = parsed.scheme
        self.allow_subdomains = allow_subdomains
        self.restrict_path = restrict_path  # e.g. "/blog"

        self.seen: Set[str] = set()
        self.scheduled: Set[str] = set()
        self.queue: asyncio.Queue = asyncio.Queue()

        # Seed
        canonical = self._canonicalize(start_url)
        if canonical:
            self.queue.put_nowait(canonical)
            self.scheduled.add(canonical)

    # ------------------------------------------------------------------
    def _canonicalize(self, raw_url: str, base: str = "") -> Optional[str]:
        """
        Normalize a URL: resolve relative refs, strip fragments,
        lowercase scheme+host, drop trailing slashes for non-root paths.
        Returns None if the URL should be skipped.
        """
        try:
            url = urllib.parse.urljoin(base, raw_url) if base else raw_url
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return None

        # Scheme guard
        if parsed.scheme not in ("http", "https"):
            return None

        # Extension filter (Section 7)
        ext = Path(parsed.path).suffix.lower()
        if ext in self._SKIP_EXTENSIONS:
            return None

        # Domain restriction
        host = parsed.netloc.lower()
        if self.allow_subdomains:
            if not host.endswith(self.base_domain):
                return None
        else:
            if host != self.base_domain:
                return None

        # Path restriction
        if self.restrict_path and not parsed.path.startswith(self.restrict_path):
            return None

        # Rebuild without fragment
        clean = urllib.parse.urlunparse((
            parsed.scheme,
            host,
            parsed.path.rstrip("/") or "/",
            parsed.params,
            parsed.query,
            "",                  # no fragment
        ))
        return clean

    # ------------------------------------------------------------------
    def enqueue_links(self, links: List[str], base_url: str) -> None:
        """Validate and enqueue newly discovered links."""
        for raw in links:
            canonical = self._canonicalize(raw, base=base_url)
            if canonical and canonical not in self.seen and canonical not in self.scheduled:
                self.scheduled.add(canonical)
                self.queue.put_nowait(canonical)

    def mark_visited(self, url: str) -> None:
        """Record a URL as fully processed."""
        self.seen.add(url)


# ===========================================================================
# 3. Robots.txt Manager  (Section 3.3 of spec)
# ===========================================================================

class RobotsManager:
    """
    Fetches and respects robots.txt for each domain encountered.
    Results are cached per domain to avoid redundant fetches.
    """

    def __init__(self, user_agent: str = "*") -> None:
        self._agent = user_agent
        self._cache: dict = {}

    async def is_allowed(self, url: str, session: aiohttp.ClientSession) -> bool:
        """Return True if crawling ``url`` is permitted by robots.txt."""
        parsed = urllib.parse.urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url not in self._cache:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    text = await resp.text(errors="replace")
                    parser.parse(text.splitlines())
            except Exception:
                # If robots.txt is unreachable, allow by default
                parser.allow_all = True
            self._cache[robots_url] = parser

        return self._cache[robots_url].can_fetch(self._agent, url)


# ===========================================================================
# 4. HTTP Fetcher  (Section 4.1-A of spec)
# ===========================================================================

@dataclass
class FetchResult:
    """Value object returned by any fetcher."""

    url: str
    final_url: str
    status: int
    content_type: str
    html: str
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


class HttpFetcher:
    """
    Fast, async HTTP fetcher using aiohttp.
    Implements exponential-backoff retries (Section 9).
    """

    def __init__(self, config: CrawlerConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    async def fetch(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> FetchResult:
        """Fetch ``url`` with retries. Returns a :class:`FetchResult`."""
        headers = {"User-Agent": self._cfg.random_user_agent()}
        timeout = aiohttp.ClientTimeout(total=self._cfg.request_timeout)
        last_error: str = ""

        for attempt in range(self._cfg.max_retries):
            try:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                    ssl=False,
                ) as resp:
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "")
                    final_url = str(resp.url)

                    # Rate-limit retry (Section 9)
                    if status == 429:
                        wait = 2 ** attempt
                        log.warning("429 on %s – sleeping %ss", url, wait)
                        await asyncio.sleep(wait)
                        continue

                    # Server error retry (Section 9)
                    if status >= 500:
                        wait = 2 ** attempt
                        log.warning("5xx %s on %s – sleeping %ss", status, url, wait)
                        await asyncio.sleep(wait)
                        continue

                    html = await resp.text(errors="replace")
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status=status,
                        content_type=content_type,
                        html=html,
                    )

            except asyncio.TimeoutError:
                last_error = "Timeout"
                wait = 2 ** attempt
                log.warning("Timeout on %s (attempt %d) – sleeping %ss", url, attempt + 1, wait)
                await asyncio.sleep(wait)

            except Exception as exc:
                last_error = str(exc)
                wait = 2 ** attempt
                log.warning("Error on %s: %s – sleeping %ss", url, exc, wait)
                await asyncio.sleep(wait)

        return FetchResult(
            url=url, final_url=url, status=0,
            content_type="", html="", error=last_error,
        )


# ===========================================================================
# 5. Dynamic Content Detector  (Section 5.1 of spec)
# ===========================================================================

class DynamicDetector:
    """
    Heuristics to decide whether a page requires browser-based rendering.

    Triggers if ANY of the following is true:
        - HTML smaller than 2 KB
        - Contains ``<div id="root">`` or ``<div id="app">``
        - Detects skeleton/spinner patterns
    """

    _SPA_PATTERNS = re.compile(
        r'<div[^>]+id=["\'](?:root|app)["\']',
        re.IGNORECASE,
    )
    _SKELETON_PATTERNS = re.compile(
        r'class=["\'][^"\']*(?:skeleton|shimmer|loading-placeholder)[^"\']*["\']',
        re.IGNORECASE,
    )

    def needs_browser(self, html: str) -> bool:
        """Return True if the page HTML suggests JS-rendered content."""
        if len(html.encode()) < 2048:
            log.debug("HTML < 2KB – flagging for browser render")
            return True
        if self._SPA_PATTERNS.search(html):
            log.debug("SPA root/app div detected – flagging for browser render")
            return True
        if self._SKELETON_PATTERNS.search(html):
            log.debug("Skeleton layout detected – flagging for browser render")
            return True
        return False


# ===========================================================================
# 6. Browser Fetcher  (Section 4.1-B + Section 5.2 & 5.3 of spec)
# ===========================================================================

class BrowserFetcher:
    """
    Browser-based fetcher supporting Playwright and Selenium.

    Anti-bot measures applied (Section 5.3):
        - Navigator.webdriver spoofed to undefined
        - Randomised User-Agent
        - Realistic viewport size
    """

    def __init__(self, config: CrawlerConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    async def fetch(self, url: str) -> FetchResult:
        """Dispatch to the correct browser backend."""
        if self._cfg.browser_type == "playwright":
            return await self._fetch_playwright(url)
        return self._fetch_selenium(url)  # type: ignore

    # ---- Playwright backend ------------------------------------------
    async def _fetch_playwright(self, url: str) -> FetchResult:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            return FetchResult(
                url=url, final_url=url, status=0,
                content_type="", html="",
                error=f"Playwright not installed: {exc}",
            )

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    user_agent=self._cfg.random_user_agent(),
                    viewport={"width": 1280, "height": 900},
                )
                page = await context.new_page()

                # Anti-bot: hide webdriver property (Section 5.3)
                await page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                )

                resp = None
                try:
                    resp = await page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=self._cfg.page_timeout,
                    )
                except Exception:
                    # Fall back to domcontentloaded
                    try:
                        resp = await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=self._cfg.page_timeout,
                        )
                    except Exception as inner_exc:
                        await browser.close()
                        return FetchResult(
                            url=url, final_url=url, status=0,
                            content_type="", html="", error=str(inner_exc),
                        )

                # Simulate scrolls to trigger lazy loading (Section 5.2)
                await self._scroll_page(page)

                html = await page.content()
                status = resp.status if resp else 200
                content_type = (resp.headers.get("content-type", "") if resp else "")
                final_url = page.url

                await browser.close()
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status=status,
                    content_type=content_type,
                    html=html,
                )
        except Exception as exc:
            return FetchResult(
                url=url, final_url=url, status=0,
                content_type="", html="", error=str(exc),
            )

    # ---- Playwright scroll helper -----------------------------------
    @staticmethod
    async def _scroll_page(page, scrolls: int = 4, delay_ms: int = 800) -> None:
        """
        Scroll page bottom-to-top cycle to trigger lazy loading.
        Performs 2-5 scrolls with configurable delay (Section 5.2).
        """
        for _ in range(scrolls):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(delay_ms / 1000)
        await page.evaluate("window.scrollTo(0, 0)")

    # ---- Selenium backend (sync wrapped) ----------------------------
    def _fetch_selenium(self, url: str) -> FetchResult:
        try:
            from selenium import webdriver as sel_webdriver  # type: ignore
            from selenium.webdriver.chrome.options import Options  # type: ignore
            from selenium.webdriver.common.by import By  # type: ignore
        except ImportError as exc:
            return FetchResult(
                url=url, final_url=url, status=0,
                content_type="", html="",
                error=f"Selenium not installed: {exc}",
            )

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument(f"--user-agent={self._cfg.random_user_agent()}")
        opts.add_argument("--window-size=1280,900")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        try:
            driver = sel_webdriver.Chrome(options=opts)
            # Anti-bot
            driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"},
            )
            driver.set_page_load_timeout(self._cfg.page_timeout // 1000)
            driver.get(url)
            time.sleep(1.0)  # allow JS execution

            # Scroll (Section 5.2)
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.8)

            html = driver.page_source
            final_url = driver.current_url
            driver.quit()
            return FetchResult(
                url=url, final_url=final_url,
                status=200, content_type="text/html", html=html,
            )
        except Exception as exc:
            return FetchResult(
                url=url, final_url=url, status=0,
                content_type="", html="", error=str(exc),
            )


# ===========================================================================
# 7. Content Extractor  (Sections 6, 7 of spec)
# ===========================================================================

class ContentExtractor:
    """
    Parses raw HTML into clean plain text and extracts hyperlinks.

    Processing steps (Section 6.1):
        1. Parse with BeautifulSoup
        2. Strip script, style, header, footer, nav
        3. Extract text preserving heading/paragraph hierarchy
        4. Extract ``<a href>`` links
    """

    # Tags to remove entirely before text extraction
    _REMOVE_TAGS = ["script", "style", "noscript", "iframe",
                    "header", "footer", "nav", "aside", "form"]

    def extract(self, html: str) -> tuple[str, List[str]]:
        """
        Returns ``(clean_text, links)``.

        ``clean_text`` – plain-text body of the page.
        ``links``      – list of raw ``href`` values found on the page.
        """
        soup = BeautifulSoup(html, "html.parser")

        # 6.2 Rule: do not rely solely on innerText;
        #     full rendered HTML → structured parser
        for tag in self._REMOVE_TAGS:
            for node in soup.find_all(tag):
                node.decompose()

        # Preserve headings with Markdown-like markers for readability
        for heading_tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading_tag.name[1])
            heading_tag.insert_before("\n" + "#" * level + " ")
            heading_tag.insert_after("\n")

        # Extract clean text with meaningful whitespace
        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Section 6.2 fragment rules: skip empty
        if not text:
            return "", []

        # Link extraction (Section 7)
        links = [
            a.get("href", "")
            for a in soup.find_all("a", href=True)
            if a.get("href", "").strip()
        ]

        return text, links


# ===========================================================================
# 8. Text Chunker  (Section 8 of spec — MUST remain unchanged)
# ===========================================================================

class TextChunker:
    """
    Splits plain text into token-aware, overlapping chunks.

    Spec requirement (Section 8):
        - Respect max token limit
        - Include overlap between chunks
        - Logic must remain unchanged for downstream compatibility

    Token counting uses a simple whitespace-word heuristic by default.
    If ``tiktoken`` is installed, the "cl100k_base" BPE encoding is used.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self._max = max_tokens
        self._overlap = overlap_tokens
        self._enc = self._load_tiktoken()

    @staticmethod
    def _load_tiktoken():
        """Try to load tiktoken; fall back to None (word-based counting)."""
        try:
            import tiktoken  # noqa: WPS433
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """Return a list of tokens (BPE ids cast to str, or words)."""
        if self._enc:
            return [str(t) for t in self._enc.encode(text)]
        return text.split()

    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        if self._enc:
            return self._enc.decode([int(t) for t in tokens])
        return " ".join(tokens)

    # ------------------------------------------------------------------
    def chunk(self, text: str) -> List[str]:
        """
        Split ``text`` into overlapping chunks of at most ``max_tokens``
        with ``overlap_tokens`` overlap between consecutive chunks.

        Empty or whitespace-only inputs return an empty list (Section 6.2).
        """
        text = text.strip()
        if not text:
            return []

        tokens = self._tokenize(text)
        if not tokens:
            return []

        chunks: List[str] = []
        step = self._max - self._overlap
        if step <= 0:
            step = self._max

        start = 0
        while start < len(tokens):
            end = min(start + self._max, len(tokens))
            chunk_text = self._detokenize(tokens[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start += step

        return chunks


# ===========================================================================
# 9. Output Writer  (Section 2 of spec)
# ===========================================================================

class OutputWriter:
    """
    Writes extracted chunks to JSONL format as specified in Section 2.

    Each record:
    ::

        {
          "meta": {
            "url": "<final_url>",
            "time": <unix_timestamp>,
            "status": <http_status_code>,
            "content_type": "<mime_type>"
          },
          "chunk": "<text_chunk>"
        }

    Keys ``meta`` and ``chunk`` are fixed (non-negotiable, Section 2).
    """

    def __init__(self, output_path: str) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so a resumed crawl doesn't truncate existing data
        self._fh = self._path.open("a", encoding="utf-8")
        log.info("OutputWriter writing to: %s", self._path.resolve())

    def write(
        self,
        chunk: str,
        fetch_result: FetchResult,
    ) -> None:
        """Serialize and write one JSONL record."""
        record = {
            "meta": {
                "url": fetch_result.final_url,
                "time": fetch_result.timestamp,
                "status": fetch_result.status,
                "content_type": fetch_result.content_type,
            },
            "chunk": chunk,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the file handle."""
        self._fh.flush()
        self._fh.close()
        log.info("OutputWriter closed – output at: %s", self._path.resolve())

    # Context-manager support
    def __enter__(self) -> "OutputWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ===========================================================================
# 10. Rate Limiter  (Section 3.2 per-host politeness)
# ===========================================================================

class RateLimiter:
    """
    Per-host politeness delay.
    Keeps the last-access timestamp per host and sleeps if needed.
    """

    def __init__(self, delay: float = 0.5) -> None:
        self._delay = delay
        self._last_access: dict = {}
        self._lock = asyncio.Lock()

    async def wait(self, url: str) -> None:
        """Block until the configured delay has elapsed for this host."""
        host = urllib.parse.urlparse(url).netloc
        async with self._lock:
            last = self._last_access.get(host, 0.0)
            delta = time.monotonic() - last
            if delta < self._delay:
                await asyncio.sleep(self._delay - delta)
            self._last_access[host] = time.monotonic()


# ===========================================================================
# 11. Crawler Worker  (Section 3.2 + Section 11 architecture)
# ===========================================================================

class CrawlerWorker:
    """
    Single async worker that:
        1. Pulls a URL from the shared queue
        2. Checks robots.txt
        3. Fetches via HTTP (or upgrades to browser if dynamic)
        4. Extracts content and links
        5. Chunks text
        6. Writes JSONL records
        7. Enqueues discovered links
    """

    def __init__(
        self,
        worker_id: int,
        config: CrawlerConfig,
        url_manager: URLManager,
        robots_manager: RobotsManager,
        http_fetcher: HttpFetcher,
        browser_fetcher: BrowserFetcher,
        detector: DynamicDetector,
        extractor: ContentExtractor,
        chunker: TextChunker,
        writer: OutputWriter,
        rate_limiter: RateLimiter,
        session: aiohttp.ClientSession,
        pages_crawled: list,
        lock: asyncio.Lock,
    ) -> None:
        self._id = worker_id
        self._cfg = config
        self._url_mgr = url_manager
        self._robots = robots_manager
        self._http = http_fetcher
        self._browser = browser_fetcher
        self._detector = detector
        self._extractor = extractor
        self._chunker = chunker
        self._writer = writer
        self._rate = rate_limiter
        self._session = session
        self._pages_crawled = pages_crawled
        self._lock = lock

    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Worker loop – runs until the queue is exhausted or max_pages hit."""
        while True:
            # Check page cap
            async with self._lock:
                if len(self._pages_crawled) >= self._cfg.max_pages:
                    break

            try:
                url = self._url_mgr.queue.get_nowait()
            except asyncio.QueueEmpty:
                # Brief wait before declaring done (other workers may enqueue)
                await asyncio.sleep(0.3)
                if self._url_mgr.queue.empty():
                    break
                continue

            # Skip if already visited (race-condition guard)
            if url in self._url_mgr.seen:
                self._url_mgr.queue.task_done()
                continue

            self._url_mgr.mark_visited(url)

            # Robots.txt gate (Section 3.3)
            allowed = await self._robots.is_allowed(url, self._session)
            if not allowed:
                log.info("[W%d] Robots.txt disallows: %s", self._id, url)
                self._url_mgr.queue.task_done()
                continue

            # Per-host rate limiting
            await self._rate.wait(url)

            log.info("[W%d] Crawling (%d/%d): %s",
                     self._id, len(self._pages_crawled) + 1,
                     self._cfg.max_pages, url)

            # Phase 1: HTTP fetch
            result = await self._http.fetch(url, self._session)

            # Phase 2: upgrade to browser if dynamic (Section 5)
            use_browser = self._cfg.use_browser
            if not use_browser and result.ok:
                use_browser = self._detector.needs_browser(result.html)

            if use_browser:
                log.info("[W%d] Upgrading to browser render: %s", self._id, url)
                browser_result = await self._browser.fetch(url)
                if browser_result.ok:
                    result = browser_result
                else:
                    # Fallback to raw HTTP parse (Section 9 fallback)
                    log.warning("[W%d] Browser fallback to raw parse: %s", self._id, url)

            if not result.ok:
                log.warning("[W%d] Skipping %s – %s", self._id,
                            url, result.error or f"HTTP {result.status}")
                self._url_mgr.queue.task_done()
                continue

            # Phase 3: Extract content and links
            text, links = self._extractor.extract(result.html)

            # Phase 4: Chunk and write JSONL (Section 2 + Section 8)
            chunks = self._chunker.chunk(text)
            for chunk in chunks:
                if chunk.strip():  # No empty chunks (Section 6.2)
                    self._writer.write(chunk, result)

            # Phase 5: Enqueue discovered links (Section 7)
            self._url_mgr.enqueue_links(links, base_url=result.final_url)

            # Track progress
            async with self._lock:
                self._pages_crawled.append(url)

            self._url_mgr.queue.task_done()

        log.info("[W%d] Worker finished.", self._id)


# ===========================================================================
# 12. Main Crawler Controller  (Section 11 architecture diagram)
# ===========================================================================

class Crawler:
    """
    Top-level orchestrator.

    Wires together all components and launches the async worker pool.

    Usage::

        cfg = CrawlerConfig.from_env()
        crawler = Crawler(cfg)
        asyncio.run(crawler.run())
    """

    def __init__(self, config: CrawlerConfig) -> None:
        self._cfg = config

    async def run(self) -> None:
        """Execute the full crawl pipeline asynchronously."""
        log.info("=== Spec Crawler Starting ===")
        log.info("Start URL : %s", self._cfg.start_url)
        log.info("Max pages : %d", self._cfg.max_pages)
        log.info("Concurrency: %d", self._cfg.concurrency)
        log.info("Browser   : %s (%s)", self._cfg.use_browser, self._cfg.browser_type)
        log.info("Output    : %s", self._cfg.output_path)

        # Shared state
        url_manager = URLManager(self._cfg.start_url)
        robots_manager = RobotsManager(user_agent="SpecCrawlerBot/1.0")
        http_fetcher = HttpFetcher(self._cfg)
        browser_fetcher = BrowserFetcher(self._cfg)
        detector = DynamicDetector()
        extractor = ContentExtractor()
        chunker = TextChunker(
            max_tokens=self._cfg.chunk_max_tokens,
            overlap_tokens=self._cfg.chunk_overlap_tokens,
        )
        rate_limiter = RateLimiter(delay=self._cfg.rate_limit_delay)
        pages_crawled: list = []
        lock = asyncio.Lock()

        connector = aiohttp.TCPConnector(ssl=False, limit=self._cfg.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            with OutputWriter(self._cfg.output_path) as writer:
                workers = [
                    CrawlerWorker(
                        worker_id=i,
                        config=self._cfg,
                        url_manager=url_manager,
                        robots_manager=robots_manager,
                        http_fetcher=http_fetcher,
                        browser_fetcher=browser_fetcher,
                        detector=detector,
                        extractor=extractor,
                        chunker=chunker,
                        writer=writer,
                        rate_limiter=rate_limiter,
                        session=session,
                        pages_crawled=pages_crawled,
                        lock=lock,
                    )
                    for i in range(self._cfg.concurrency)
                ]
                await asyncio.gather(*[w.run() for w in workers])

        log.info("=== Crawl Complete – %d pages scraped ===", len(pages_crawled))
        log.info("Output written to: %s", self._cfg.output_path)


# ===========================================================================
# 13. Entry Point
# ===========================================================================

def main() -> None:
    """
    Interactive CLI entry point — URL only, everything else is auto-maxed.

    Run directly::

        python spec_crawler.py
    """
    print("\n" + "=" * 60)
    print("   SpecCrawler — Dynamic Web Crawler (JSONL Output)")
    print("=" * 60)
    print("  Workers    : 32 parallel async workers")
    print("  Max pages  : 10,000 (effectively unlimited)")
    print("  Browser    : auto-detected per page (JS-heavy sites upgraded)")
    print("  Retries    : 5x with exponential backoff")
    print("=" * 60 + "\n")

    # Only ask for URL
    while True:
        start_url = input("  Enter target URL to crawl: ").strip()
        if start_url:
            if not start_url.startswith(("http://", "https://")):
                start_url = "https://" + start_url
            break
        print("  [!] URL cannot be empty.")

    # Output filename derived from domain automatically
    import urllib.parse as _up
    domain = _up.urlparse(start_url).netloc.replace(".", "_")
    output_path = f"{domain}_crawl.jsonl"

    print(f"\n  Crawling  : {start_url}")
    print(f"  Output    : {output_path}")
    print("\n  Starting...\n")

    config = CrawlerConfig(
        start_url=start_url,
        output_path=output_path,
    )

    crawler = Crawler(config)
    asyncio.run(crawler.run())

    print("\n" + "=" * 60)
    print(f"  Done! Output saved to: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
