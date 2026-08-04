import asyncio
import hashlib
import socket
from urllib.parse import urlsplit

from playwright.async_api import Browser, Route
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.core.errors import UpstreamError
from app.utils.urls import is_private_ip


class WebScraper:
    def __init__(self, browser: Browser, settings):
        self.browser = browser
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.scrape_concurrency)
        self.proxy_enforced = bool(settings.scrape_proxy_url)

    async def _validate_url(self, value: str) -> None:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Blocked URL scheme or hostname")
        if parsed.username or parsed.password:
            raise ValueError("URLs with credentials are blocked")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise ValueError("Invalid URL port") from exc
        host = parsed.hostname.rstrip(".")
        if is_private_ip(host) or host.lower() in {
            "localhost",
            "localhost.localdomain",
        }:
            raise ValueError("Private and loopback destinations are blocked")
        # The isolated worker has no direct DNS or internet egress. Squid resolves
        # hostnames and applies destination-IP ACLs, binding validation to the
        # actual proxied connection and avoiding an application DNS-rebinding gap.
        if self.proxy_enforced:
            return
        try:
            addresses = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError("URL hostname could not be resolved") from exc
        if not addresses or any(is_private_ip(address[4][0]) for address in addresses):
            raise ValueError("Private and loopback destinations are blocked")

    async def _route(self, route: Route) -> None:
        if route.request.resource_type in {
            "image",
            "media",
            "font",
            "stylesheet",
            "texttrack",
            "websocket",
            "manifest",
        }:
            await route.abort("blockedbyclient")
            return
        try:
            await self._validate_url(route.request.url)
            await route.continue_()
        except (ValueError, OSError):
            await route.abort("blockedbyclient")

    async def fetch(self, url: str) -> dict:
        await self._validate_url(url)
        requested = urlsplit(url)
        if requested.hostname in {"linkedin.com", "www.linkedin.com"}:
            raise UpstreamError(
                "The website requires authentication",
                502,
                "website_access_restricted",
            )
        async with self.semaphore:
            context = await self.browser.new_context(
                accept_downloads=False, service_workers="block"
            )
            page = await context.new_page()
            await context.route("**/*", self._route)
            try:
                async with asyncio.timeout(self.settings.scrape_timeout_seconds):
                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.settings.scrape_timeout_seconds * 1000,
                    )
                    await self._validate_url(page.url)
                    final_url = urlsplit(page.url)
                    if final_url.hostname in {
                        "linkedin.com",
                        "www.linkedin.com",
                    } and final_url.path.startswith(("/authwall", "/login")):
                        raise UpstreamError(
                            "The website requires authentication",
                            502,
                            "website_access_restricted",
                        )
                    title = await page.title()
                    description_locator = page.locator('meta[name="description"]')
                    description = None
                    if await description_locator.count():
                        description = await description_locator.first.get_attribute(
                            "content"
                        )
                    content_locator = page.locator("main, article, body").first
                    text = await content_locator.inner_text(timeout=5_000)
                    excerpt = " ".join(text.split())[
                        : self.settings.scrape_max_content_chars
                    ]
                    source_id = hashlib.sha256(page.url.encode()).hexdigest()[:16]
                    return {
                        "id": f"web:{source_id}",
                        "url": page.url[:2_048],
                        "type": "website",
                        "title": title[:500],
                        "excerpt": f"{description or ''}\n{excerpt}"[
                            : self.settings.scrape_max_content_chars
                        ],
                    }
            except (PlaywrightTimeoutError, TimeoutError, ValueError, OSError) as exc:
                raise UpstreamError(
                    "Website scraping failed", 502, "website_unavailable"
                ) from exc
            finally:
                await context.close()
