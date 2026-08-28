import asyncio
import json

import httpx
import pytest

from app.clients.oembed import OEmbedClient
from app.core.errors import UpstreamError


@pytest.mark.parametrize(
    ("url", "provider_host", "source_type"),
    [
        ("https://youtu.be/video-id", "www.youtube.com", "youtube"),
        ("https://vimeo.com/123456", "vimeo.com", "vimeo"),
    ],
)
def test_public_media_returns_structured_metadata(url, provider_host, source_type):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == provider_host
        assert request.url.params["url"] == url
        return httpx.Response(
            200,
            json={
                "title": "Candidate presentation",
                "author_name": "Candidate",
                "provider_name": source_type.title(),
                "type": "video",
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OEmbedClient(client).fetch(url)

    source = asyncio.run(run())[0]
    excerpt = json.loads(source["excerpt"])

    assert source["type"] == source_type
    assert source["kind"] == "video"
    assert source["access_status"] == "public"
    assert source["title"] == "Candidate presentation"
    assert excerpt["author"] == "Candidate"


def test_private_media_is_reported_as_restricted():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await OEmbedClient(client).fetch("https://vimeo.com/123456")

    with pytest.raises(UpstreamError) as error:
        asyncio.run(run())

    assert error.value.code == "website_access_restricted"
