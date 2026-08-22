import asyncio
import base64
import json

import httpx

from app.clients.github import PROFILE_REPO_SOURCES, GithubClient
from app.core.errors import UpstreamError


def test_profile_fetch_emits_citable_repository_sources():
    repos = [
        {
            "name": f"project-{index}",
            "full_name": f"octocat/project-{index}",
            "description": "A project",
            "language": "Python",
            "stargazers_count": index,
            "forks_count": 0,
            "updated_at": "2026-01-01T00:00:00Z",
            "fork": False,
        }
        for index in range(7)
    ]
    repos.insert(
        3,
        {
            "name": "someone-elses-work",
            "full_name": "octocat/someone-elses-work",
            "fork": True,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/users/octocat":
            return httpx.Response(
                200,
                json={
                    "login": "octocat",
                    "name": "Octo Cat",
                    "bio": None,
                    "public_repos": 8,
                    "followers": 5,
                    "created_at": "2020-01-01T00:00:00Z",
                },
            )
        if request.url.path == "/users/octocat/repos":
            return httpx.Response(200, json=repos)
        raise AssertionError(f"Unexpected endpoint {request.url.path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GithubClient(client).fetch("https://github.com/octocat")

    sources = asyncio.run(run())

    assert sources[0]["id"] == "github:octocat"
    assert sources[0]["url"] == "https://github.com/octocat"
    ids = [source["id"] for source in sources]
    # Profile + capped repository sources; forks are never emitted.
    assert len(ids) == 1 + PROFILE_REPO_SOURCES
    assert "github:octocat/project-0" in ids
    assert f"github:octocat/project-{PROFILE_REPO_SOURCES}" not in ids
    assert "github:octocat/someone-elses-work" not in ids
    listed = json.loads(sources[1]["excerpt"])
    assert listed["description"] == "A project"
    assert listed["language"] == "Python"


def test_invalid_token_maps_to_auth_failed_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await GithubClient(client, token="expired").fetch(
                "https://github.com/octocat"
            )

    try:
        asyncio.run(run())
    except UpstreamError as exc:
        assert exc.code == "github_auth_failed"
    else:
        raise AssertionError("expected github_auth_failed")


def test_repo_fetch_returns_single_source_including_readme():
    readme_content = "# hello-world\n\nBuilt with Python."
    encoded = base64.b64encode(readme_content.encode()).decode()
    package_content = json.dumps(
        {
            "name": "hello-world",
            "scripts": {"test": "pytest", "hidden": "do not expose"},
            "dependencies": {"next": "1.0", "react": "1.0"},
            "devDependencies": {"typescript": "1.0"},
        }
    )
    encoded_package = base64.b64encode(package_content.encode()).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/octocat/hello-world":
            return httpx.Response(
                200,
                json={
                    "full_name": "octocat/hello-world",
                    "description": "Demo",
                    "html_url": "https://github.com/octocat/hello-world",
                    "stargazers_count": 3,
                },
            )
        if path == "/repos/octocat/hello-world/languages":
            return httpx.Response(200, json={"Python": 1200})
        if path == "/repos/octocat/hello-world/readme":
            return httpx.Response(200, json={"content": encoded})
        if path == "/repos/octocat/hello-world/contents/package.json":
            return httpx.Response(200, json={"content": encoded_package})
        raise AssertionError(f"Unexpected endpoint {path}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GithubClient(client).fetch(
                "https://github.com/octocat/hello-world"
            )

    sources = asyncio.run(run())

    assert len(sources) == 1
    assert sources[0]["id"] == "github:octocat/hello-world"
    payload = json.loads(sources[0]["excerpt"])
    assert payload["readme"] == readme_content
    assert payload["package"] == {
        "name": "hello-world",
        "dependencies": ["next", "react"],
        "devDependencies": ["typescript"],
        "scripts": ["hidden", "test"],
    }
