import asyncio
import json

import httpx

from app.clients.bitbucket import BitbucketClient
from app.core.errors import UpstreamError


def test_profile_fetch_emits_public_repository_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/2.0/repositories/candidate":
            return httpx.Response(
                200,
                json={
                    "size": 2,
                    "values": [
                        {
                            "full_name": "candidate/public-project",
                            "description": "Public project",
                            "language": "Python",
                            "updated_on": "2026-08-01T00:00:00Z",
                            "is_private": False,
                            "mainbranch": {"name": "main"},
                            "links": {
                                "html": {
                                    "href": "https://bitbucket.org/candidate/public-project"
                                }
                            },
                        },
                        {
                            "full_name": "candidate/private-project",
                            "is_private": True,
                        },
                    ],
                },
            )
        if request.url.path.endswith("/src/main/README.md"):
            return httpx.Response(200, text="# Public Project\n\nPython service")
        if request.url.path.endswith("/src/main/package.json"):
            return httpx.Response(404)
        raise AssertionError(f"Unexpected endpoint {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await BitbucketClient(client).fetch(
                "https://bitbucket.org/candidate"
            )

    sources = asyncio.run(run())

    assert [source["id"] for source in sources] == [
        "bitbucket:candidate",
        "bitbucket:candidate/public-project",
    ]
    assert json.loads(sources[0]["excerpt"])["public_repositories"] == 2
    repository = json.loads(sources[1]["excerpt"])
    assert repository["language"] == "Python"
    assert repository["readme"] == "# Public Project\n\nPython service"
    assert repository["package"] is None


def test_repository_fetch_includes_readme_and_manifest():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/2.0/repositories/team/public-project":
            return httpx.Response(
                200,
                json={
                    "full_name": "team/public-project",
                    "description": "Public service",
                    "language": "TypeScript",
                    "scm": "git",
                    "mainbranch": {"name": "main"},
                    "links": {
                        "html": {"href": "https://bitbucket.org/team/public-project"}
                    },
                },
            )
        if path.endswith("/src/main/README.md"):
            return httpx.Response(
                200, text="Public deployment documentation" + "x" * 20_000
            )
        if path.endswith("/src/main/package.json"):
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "name": "public-project",
                        "dependencies": {"next": "1"},
                        "scripts": {"test": "vitest"},
                    }
                ),
            )
        raise AssertionError(f"Unexpected endpoint {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await BitbucketClient(client).fetch(
                "https://bitbucket.org/team/public-project/src/main"
            )

    source = asyncio.run(run())[0]
    excerpt = json.loads(source["excerpt"])

    assert source["id"] == "bitbucket:team/public-project"
    assert source["url"] == "https://bitbucket.org/team/public-project"
    assert excerpt["readme"].startswith("Public deployment documentation")
    assert len(excerpt["readme"].encode()) == 4_000
    assert excerpt["package"]["dependencies"] == ["next"]


def test_private_repository_maps_to_restricted_access():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await BitbucketClient(client).fetch(
                "https://bitbucket.org/team/private-project"
            )

    try:
        asyncio.run(run())
    except UpstreamError as exc:
        assert exc.code == "website_access_restricted"
    else:
        raise AssertionError("expected website_access_restricted")
