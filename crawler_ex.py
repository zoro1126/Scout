#!/usr/bin/env python3
"""
DEPRECATED – crawler_ex.py
==========================
This module is deprecated and will be removed in a future release.
Please use `crawler.py` (CrawlerConfig + Crawler) instead.
"""
import warnings
warnings.warn(
    "crawler_ex.py is deprecated and will be removed in a future release. "
    "Use 'from crawler import Crawler, CrawlerConfig' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations

import asyncio
import aiofiles
import aiohttp
import csv
import io
import json
import logging
import os
import random
import re
import time
import urllib.robotparser
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from lxml import etree
from lxml import html as lxml_html
import tldextract

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

try:
    import tiktoken
    TOK_ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(TOK_ENCODER.encode(text))
except Exception:
    TOK_ENCODER = None

    def count_tokens(text: str) -> int:
        words = re.findall(r"\w+|\S", text)
        return int(len(words) * 1.33) + 1


START_URL = os.environ.get("START_URL", "https://www.wikipedia.org")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "10"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))
MAX_CTX_TOKENS = int(os.environ.get("MAX_CTX_TOKENS", "4096"))
RESERVED_PROMPT_TOKENS = int(os.environ.get("RESERVED_PROMPT_TOKENS", "128"))
MAX_CHUNK_TOKENS = MAX_CTX_TOKENS - RESERVED_PROMPT_TOKENS
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "64"))
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "output_chunks.jsonl")

USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

PER_HOST_DELAY = float(os.environ.get("PER_HOST_DELAY", "0.4"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "25"))
PAGE_TIMEOUT = int(os.environ.get("PAGE_TIMEOUT", "35"))

USE_PLAYWRIGHT = os.environ.get("USE_PLAYWRIGHT", "0").strip().lower() in {"1", "true", "yes"}
PLAYWRIGHT_NAV_TIMEOUT_MS = int(os.environ.get("PLAYWRIGHT_NAV_TIMEOUT_MS", "12000"))
PLAYWRIGHT_WAIT_MS = int(os.environ.get("PLAYWRIGHT_WAIT_MS", "600"))
PLAYWRIGHT_SCROLLS = int(os.environ.get("PLAYWRIGHT_SCROLLS", "1"))
PROXY_URL = os.environ.get("PROXY_URL")
AUTH_COOKIES = os.environ.get("SCRAPER_COOKIES")

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "yclid", "mc_eid", "mc_cid", "ga_campaign"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("universal-crawler")


def normalize_start_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        scheme, netloc, path, params, query, fragment = parsed
        fragment = ""

        if query:
            q = parse_qsl(query, keep_blank_values=True)
            q_filtered = [(k, v) for (k, v) in q if k not in TRACKING_PARAMS]
            q_filtered.sort()
            query = urlencode(q_filtered, doseq=True)
        else:
            query = ""

        return urlunparse((scheme, netloc, path or "/", params or "", query, fragment))
    except Exception:
        return url


def same_domain(start_url: str, candidate: str) -> bool:
    s = urlparse(start_url)
    c = urlparse(candidate)
    if not c.netloc:
        return True
    s_e = tldextract.extract(s.netloc)
    c_e = tldextract.extract(c.netloc)
    return s_e.domain == c_e.domain and s_e.suffix == c_e.suffix


def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.split("#")[0].strip()
    if href.startswith(("mailto:", "tel:", "javascript:")):
        return None
    try:
        joined = urljoin(base, href)
        parsed = urlparse(joined)
        if parsed.scheme not in ("http", "https"):
            return None
        return canonicalize_url(parsed.geturl())
    except Exception:
        return None


def extract_visible_text(html_bytes: bytes) -> str:
    try:
        doc = lxml_html.fromstring(html_bytes)
    except Exception:
        try:
            return html_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    for bad in doc.xpath('//script|//style|//noscript|//header|//footer|//nav|//aside|//form|//svg|//iframe'):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)

    etree.strip_elements(doc, etree.Comment, with_tail=False)

    text_blocks: List[str] = []
    for selector in ["//article", "//main", "//body"]:
        nodes = doc.xpath(selector)
        if nodes:
            for node in nodes:
                text = node.text_content()
                if text and len(text.strip()) > 20:
                    text_blocks.append(text)
            if text_blocks:
                break

    if not text_blocks:
        text_blocks = [doc.text_content()]

    text = "\n\n".join(tb.strip() for tb in text_blocks if tb and tb.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sentences_like(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: List[str] = []
    for p in paras:
        if len(p.split()) > 200:
            parts = re.split(r"(?<=[\.\?\!])\s+", p)
            out.extend([s.strip() for s in parts if s.strip()])
        else:
            out.append(p)
    return out


def chunk_text_tokenwise(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    token_counter: Callable[[str], int] = count_tokens,
):
    pieces = split_into_sentences_like(text)
    if not pieces:
        return

    cur_chunk: List[str] = []
    cur_tokens = 0
    piece_tokens = [token_counter(p) for p in pieces]
    i = 0

    while i < len(pieces):
        p = pieces[i]
        ptok = piece_tokens[i]

        if ptok >= max_tokens:
            words = p.split()
            w_i = 0
            while w_i < len(words):
                sub = []
                while w_i < len(words):
                    sub.append(words[w_i])
                    if token_counter(" ".join(sub)) > max_tokens:
                        sub.pop()
                        break
                    w_i += 1
                if sub:
                    yield " ".join(sub)
                else:
                    yield words[w_i]
                    w_i += 1
            i += 1
            continue

        if cur_tokens + ptok <= max_tokens:
            cur_chunk.append(p)
            cur_tokens += ptok
            i += 1
        else:
            if cur_chunk:
                yield "\n\n".join(cur_chunk)

            if overlap_tokens > 0 and cur_chunk:
                tail = []
                tail_tokens = 0
                for tpiece in reversed(cur_chunk):
                    t_toks = token_counter(tpiece)
                    if tail_tokens + t_toks > overlap_tokens:
                        break
                    tail.insert(0, tpiece)
                    tail_tokens += t_toks

                if tail_tokens + ptok > max_tokens:
                    cur_chunk = []
                    cur_tokens = 0
                else:
                    cur_chunk = tail.copy()
                    cur_tokens = tail_tokens
            else:
                cur_chunk = []
                cur_tokens = 0

    if cur_chunk:
        yield "\n\n".join(cur_chunk)


class RobotsCache:
    def __init__(self):
        self.cache = {}
        self.lock = asyncio.Lock()

    async def allowed(self, session: aiohttp.ClientSession, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        async with self.lock:
            if origin in self.cache:
                return self.cache[origin].can_fetch(user_agent, url)

        robots_url = urljoin(origin, "/robots.txt")
        try:
            async with session.get(robots_url, timeout=10) as resp:
                text = await resp.text()
        except Exception:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse([])
            async with self.lock:
                self.cache[origin] = rp
            return True

        rp = urllib.robotparser.RobotFileParser()
        rp.parse(text.splitlines())
        async with self.lock:
            self.cache[origin] = rp
        return rp.can_fetch(user_agent, url)


class UniversalCrawler:
    def __init__(self, start_url: str, max_pages: int = MAX_PAGES, concurrency: int = CONCURRENCY, output_file: str = OUTPUT_FILE):
        self.start_url = canonicalize_url(normalize_start_url(start_url))
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.output_file = output_file

        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.seen: Set[str] = set()
        self.scheduled: Set[str] = set()

        self.pages_fetched = 0
        self.pages_lock = asyncio.Lock()

        self.session: Optional[aiohttp.ClientSession] = None
        self.per_host_last = defaultdict(float)
        self.robots = RobotsCache()

        self.playwright = None
        self.browser = None
        self.browser_context = None
        self.playwright_ready = False
        self.pw_semaphore = asyncio.Semaphore(2)

    async def _write_chunk(self, metadata: dict, chunk_text: str):
        row = {"meta": metadata, "chunk": chunk_text}
        async with aiofiles.open(self.output_file, "a", encoding="utf-8") as f:
            await f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _init_playwright(self):
        if not (PLAYWRIGHT_AVAILABLE and USE_PLAYWRIGHT):
            return

        try:
            self.playwright = await async_playwright().start()
            launch_kwargs = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            }
            if PROXY_URL:
                launch_kwargs["proxy"] = {"server": PROXY_URL}

            self.browser = await self.playwright.chromium.launch(**launch_kwargs)
            self.browser_context = await self.browser.new_context(
                user_agent=USER_AGENT,
                ignore_https_errors=True,
            )

            if AUTH_COOKIES:
                try:
                    cookies = json.loads(AUTH_COOKIES)
                    if isinstance(cookies, dict):
                        cookies = [cookies]
                    await self.browser_context.add_cookies(cookies)
                except Exception as exc:
                    logger.warning("Failed to parse SCRAPER_COOKIES: %s", exc)

            self.playwright_ready = True
            logger.info("Playwright initialized successfully.")
        except Exception as exc:
            logger.warning("Failed to initialize Playwright: %s", exc)
            await self._cleanup_playwright()

    async def _cleanup_playwright(self):
        try:
            if self.browser_context:
                await self.browser_context.close()
        except Exception:
            pass
        try:
            if self.browser:
                await self.browser.close()
        except Exception:
            pass
        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception:
            pass

        self.playwright = None
        self.browser = None
        self.browser_context = None
        self.playwright_ready = False

    async def _fetch_http(self, url: str) -> Tuple[Optional[bytes], Optional[str], int, str]:
        parsed_netloc = urlparse(url).netloc
        wait_for = PER_HOST_DELAY - (time.time() - self.per_host_last[parsed_netloc])
        if wait_for > 0:
            await asyncio.sleep(wait_for)

        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with self.session.get(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True) as resp:
                    self.per_host_last[parsed_netloc] = time.time()
                    final_url = str(resp.url)

                    if resp.status == 429:
                        delay = resp.headers.get("Retry-After")
                        try:
                            delay_s = int(delay) if delay is not None else 2 ** attempt
                        except Exception:
                            delay_s = 2 ** attempt
                        logger.warning("429 hit: %s. Sleeping %s seconds.", url, delay_s)
                        await asyncio.sleep(delay_s)
                        continue

                    if resp.status >= 500:
                        last_error = f"server_status_{resp.status}"
                        await asyncio.sleep((2 ** (attempt - 1)) + random.random())
                        continue

                    if resp.status >= 400:
                        body = await resp.read()
                        ctype = resp.headers.get("Content-Type", "")
                        if body:
                            return body, ctype, resp.status, final_url
                        return None, ctype, resp.status, final_url

                    body = await resp.read()
                    ctype = resp.headers.get("Content-Type", "")
                    return body, ctype, resp.status, final_url

            except asyncio.TimeoutError:
                last_error = "timeout"
                await asyncio.sleep((2 ** (attempt - 1)) + random.random())
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep((2 ** (attempt - 1)) + random.random())

        logger.warning("Fetch failed after retries: %s (%s)", url, last_error)
        return None, None, 0, url

    async def _render_with_playwright(self, url: str) -> Tuple[str, List[str]]:
        async with self.pw_semaphore:
            page = None
            try:
                page = await self.browser_context.new_page()
                page.set_default_navigation_timeout(PLAYWRIGHT_NAV_TIMEOUT_MS)
                page.set_default_timeout(PLAYWRIGHT_NAV_TIMEOUT_MS)

                await page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )

                response = await page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_NAV_TIMEOUT_MS)
                if not response or response.status >= 400:
                    return "", []

                for _ in range(max(0, PLAYWRIGHT_SCROLLS)):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(PLAYWRIGHT_WAIT_MS)

                text = await page.evaluate("document.body.innerText")
                links = await page.evaluate("Array.from(document.querySelectorAll('a[href]')).map(a => a.href)")
                return text or "", links or []
            except Exception as exc:
                logger.debug("Playwright render failed for %s: %s", url, exc)
                return "", []
            finally:
                if page is not None:
                    try:
                        await page.close()
                    except Exception:
                        pass

    def _parse_json(self, content: bytes) -> str:
        try:
            obj = json.loads(content.decode("utf-8", errors="ignore"))
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return content.decode("utf-8", errors="ignore")

    def _parse_pdf(self, content: bytes) -> str:
        if not PYPDF_AVAILABLE:
            return ""
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                try:
                    txt = page.extract_text()
                except Exception:
                    txt = ""
                if txt:
                    pages.append(txt)
            return "\n\n".join(pages)
        except Exception:
            return ""

    def _parse_csv(self, content: bytes) -> str:
        try:
            reader = csv.reader(io.StringIO(content.decode("utf-8", errors="ignore")))
            rows = []
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append(" | ".join(row))
            return "\n".join(rows)
        except Exception:
            return ""

    def _parse_xml(self, content: bytes) -> str:
        try:
            root = etree.fromstring(content)
            pieces = []
            for elem in root.iter():
                txt = " ".join((elem.text or "").split())
                if txt:
                    pieces.append(txt)
            return "\n".join(pieces)
        except Exception:
            try:
                return content.decode("utf-8", errors="ignore")
            except Exception:
                return ""

    def _parse_structured(self, content_type: str, content: bytes) -> str:
        ctype = (content_type or "").lower()
        if "application/json" in ctype:
            return self._parse_json(content)
        if "application/pdf" in ctype:
            return self._parse_pdf(content)
        if "text/csv" in ctype or "application/csv" in ctype:
            return self._parse_csv(content)
        if "xml" in ctype:
            return self._parse_xml(content)
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    async def _enqueue_url(self, base_url: str, raw_href: str):
        n = normalize_url(base_url, raw_href)
        if not n or not same_domain(self.start_url, n):
            return

        async with self.pages_lock:
            if n in self.seen or n in self.scheduled or len(self.scheduled) >= self.max_pages:
                return
            self.scheduled.add(n)

        await self.queue.put(n)

    async def fetch_and_process(self, url: str):
        logger.info("Fetching: %s", url)

        allowed = await self.robots.allowed(self.session, url, USER_AGENT)
        if not allowed:
            logger.info("Robots blocked: %s", url)
            return

        content, content_type, status, final_url = await self._fetch_http(url)
        if not content:
            logger.warning("No content: %s", url)
            return

        logger.info("Fetched %s -> status=%s content-type=%s bytes=%d", final_url, status, content_type, len(content))

        ctype = (content_type or "").lower()
        text = ""
        discovered_links: List[str] = []

        if "text/html" in ctype or "application/xhtml+xml" in ctype or ctype == "":
            text = extract_visible_text(content)

            try:
                doc = lxml_html.fromstring(content)
                discovered_links = doc.xpath('//a[@href]/@href')
            except Exception:
                discovered_links = []

            if self.playwright_ready and USE_PLAYWRIGHT and len(text.strip()) < 300:
                logger.info("Playwright fallback: %s", url)
                try:
                    pw_text, pw_links = await asyncio.wait_for(
                        self._render_with_playwright(final_url),
                        timeout=max(5, PLAYWRIGHT_NAV_TIMEOUT_MS // 1000 + 5),
                    )
                    if len(pw_text.strip()) > len(text.strip()):
                        text = pw_text
                    discovered_links.extend(pw_links)
                except asyncio.TimeoutError:
                    logger.warning("Playwright timed out for %s", url)
        else:
            text = self._parse_structured(ctype, content)

        text = text.strip()

        if not text:
            logger.warning("Empty extracted text: %s", url)
            return

        meta = {
            "url": final_url,
            "time": int(time.time()),
            "status": status,
            "content_type": content_type,
        }

        wrote_any = False
        for chunk in chunk_text_tokenwise(
            text,
            max_tokens=MAX_CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            token_counter=count_tokens,
        ):
            wrote_any = True
            await self._write_chunk(meta, chunk)

        if not wrote_any and len(text) > 0:
            await self._write_chunk(meta, text)

        for raw in discovered_links:
            await self._enqueue_url(final_url, raw)

    async def worker(self, wid: int):
        while True:
            url = await self.queue.get()
            try:
                if url is None:
                    return

                async with self.pages_lock:
                    if self.pages_fetched >= self.max_pages or url in self.seen:
                        continue
                    self.seen.add(url)
                    self.pages_fetched += 1
                    current = self.pages_fetched

                logger.info("[%d] fetching (%d/%d) %s", wid, current, self.max_pages, url)

                try:
                    await asyncio.wait_for(self.fetch_and_process(url), timeout=PAGE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("Timed out processing %s", url)
                except Exception as exc:
                    logger.exception("Worker error on %s: %s", url, exc)

            finally:
                self.queue.task_done()

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 15)
        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        headers = {"User-Agent": USER_AGENT}

        await self._init_playwright()

        async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
            self.session = session

            async with aiofiles.open(self.output_file, "w", encoding="utf-8") as f:
                await f.write("")

            await self._enqueue_url(self.start_url, self.start_url)

            workers = [asyncio.create_task(self.worker(i)) for i in range(max(1, self.concurrency))]

            try:
                await self.queue.join()
            finally:
                for _ in workers:
                    await self.queue.put(None)
                await asyncio.gather(*workers, return_exceptions=True)
                await self._cleanup_playwright()


Crawler = UniversalCrawler


def main():
    logger.info("Universal Advanced Crawler initialized.")
    logger.info("START_URL=%s, MAX_PAGES=%d, CONCURRENCY=%d", START_URL, MAX_PAGES, CONCURRENCY)
    logger.info("MAX_CHUNK_TOKENS=%d, OVERLAP_TOKENS=%d, OUTPUT_FILE=%s", MAX_CHUNK_TOKENS, OVERLAP_TOKENS, OUTPUT_FILE)

    crawler = Crawler(
        START_URL,
        max_pages=MAX_PAGES,
        concurrency=CONCURRENCY,
        output_file=OUTPUT_FILE,
    )

    try:
        asyncio.run(crawler.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down.")
    finally:
        logger.info("Done. Chunks written to %s (fetched %d pages)", OUTPUT_FILE, crawler.pages_fetched)


if __name__ == "__main__":
    main()