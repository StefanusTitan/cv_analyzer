import asyncio
import json

import httpx

from app.clients.gitlab import GitlabClient


def test_profile_fetch_emits_citable_public_project_sources():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v4/users":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 7,
                        "username": "candidate",
                        "name": "Candidate",
                        "bio": "Builds reliable systems",
                        "job_title": "Engineer",
                        "organization": "Example",
                        "location": "Jakarta",
                        "website_url": "https://candidate.example",
                    }
                ],
            )
        if path == "/api/v4/users/7/projects":
            return httpx.Response(
                200,
                json=[
                    {
                        "path_with_namespace": "candidate/public-project",
                        "name_with_namespace": "Candidate / Public Project",
                        "web_url": "https://gitlab.com/candidate/public-project",
                        "description": "Public project",
                        "topics": ["python"],
                        "star_count": 2,
                        "forks_count": 1,
                        "last_activity_at": "2026-08-01T00:00:00Z",
                    }
                ],
            )
        if path.endswith("/languages"):
            return httpx.Response(200, json={"Python": 90.0, "Shell": 10.0})
        if "/repository/files/README.md/raw" in path:
            return httpx.Response(200, text="# Public Project\n\nFastAPI service")
        if "/repository/files/package.json/raw" in path:
            return httpx.Response(404)
        raise AssertionError(f"Unexpected endpoint {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GitlabClient(client).fetch("https://gitlab.com/candidate")

    sources = asyncio.run(run())

    assert sources[0]["id"] == "gitlab:candidate"
    assert sources[0]["type"] == "gitlab"
    assert json.loads(sources[0]["excerpt"])["job_title"] == "Engineer"
    assert sources[1]["id"] == "gitlab:candidate/public-project"
    assert sources[1]["url"] == "https://gitlab.com/candidate/public-project"
    project = json.loads(sources[1]["excerpt"])
    assert project["languages"] == {"Python": 90.0, "Shell": 10.0}
    assert project["readme"] == "# Public Project\n\nFastAPI service"
    assert project["package"] is None


def test_repository_fetch_includes_languages_readme_and_manifest():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/languages"):
            return httpx.Response(200, json={"Python": 80.0, "Shell": 20.0})
        if "/repository/files/README.md/raw" in path:
            return httpx.Response(200, text="Public deployment documentation")
        if "/repository/files/package.json/raw" in path:
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "name": "public-project",
                        "dependencies": {"fastapi": "1"},
                        "scripts": {"test": "pytest"},
                    }
                ),
            )
        if "/api/v4/projects/" in path:
            return httpx.Response(
                200,
                json={
                    "path_with_namespace": "team/public-project",
                    "name_with_namespace": "Team / Public Project",
                    "web_url": "https://gitlab.com/team/public-project",
                    "description": "Production service",
                    "created_at": "2025-01-01T00:00:00Z",
                    "last_activity_at": "2026-08-01T00:00:00Z",
                    "star_count": 3,
                    "forks_count": 1,
                    "open_issues_count": 0,
                    "topics": ["api"],
                    "license": {"key": "mit"},
                    "archived": False,
                },
            )
        raise AssertionError(f"Unexpected endpoint {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GitlabClient(client).fetch(
                "https://gitlab.com/team/public-project/-/tree/main"
            )

    source = asyncio.run(run())[0]
    excerpt = json.loads(source["excerpt"])

    assert source["id"] == "gitlab:team/public-project"
    assert source["url"] == "https://gitlab.com/team/public-project"
    assert excerpt["languages"] == {"Python": 80.0, "Shell": 20.0}
    assert excerpt["readme"] == "Public deployment documentation"
    assert excerpt["package"]["dependencies"] == ["fastapi"]
