import hashlib
import json
from urllib.parse import urlencode

import httpx

from app.core.errors import UpstreamError
from app.utils.urls import classify_source, normalize_url


class OEmbedClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def fetch(self, url: str) -> list[dict]:
        normalized = normalize_url(url)
        if normalized is None:
            raise ValueError("Invalid media URL")
        source_type, source_kind = classify_source(normalized)
        if source_kind != "video":
            raise ValueError("Unsupported oEmbed URL")
        endpoints = {
            "youtube": "https://www.youtube.com/oembed",
            "vimeo": "https://vimeo.com/api/oembed.json",
        }
        endpoint = endpoints.get(source_type)
        if endpoint is None:
            raise ValueError("Unsupported oEmbed URL")
        query = urlencode({"url": normalized, "format": "json"})
        try:
            response = await self.client.get(
                f"{endpoint}?{query}",
                headers={"Accept": "application/json", "User-Agent": "cv-analyzer"},
            )
            if response.status_code == 404:
                raise UpstreamError(
                    "The public media was not found", 404, "media_not_found"
                )
            if response.status_code in {401, 403}:
                raise UpstreamError(
                    "The media is not publicly accessible",
                    502,
                    "website_access_restricted",
                )
            if response.status_code == 429:
                raise UpstreamError(
                    "The media provider rate limited enrichment",
                    502,
                    "media_rate_limited",
                )
            response.raise_for_status()
            payload = response.json()
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Media metadata enrichment timed out", 504, "media_timeout"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise UpstreamError(
                "Media metadata enrichment failed", 502, "media_unavailable"
            ) from exc
        if not isinstance(payload, dict):
            raise UpstreamError(
                "The media provider returned an unexpected response",
                502,
                "media_invalid_response",
            )
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            raise UpstreamError(
                "The media provider returned an unexpected response",
                502,
                "media_invalid_response",
            )
        excerpt = {
            "author": payload.get("author_name"),
            "provider": payload.get("provider_name"),
            "media_type": payload.get("type"),
        }
        source_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return [
            {
                "id": f"{source_type}:{source_hash}",
                "url": normalized,
                "type": source_type,
                "kind": source_kind,
                "access_status": "public",
                "title": title.strip()[:500],
                "excerpt": json.dumps(excerpt, ensure_ascii=False),
            }
        ]
