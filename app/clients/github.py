import asyncio
import base64
import binascii
import json
from urllib.parse import urlsplit

import httpx

from app.core.errors import UpstreamError
from app.utils.urls import is_github_url


class GithubClient:
    def __init__(self, client: httpx.AsyncClient, token: str = ""):
        self.client = client
        self.token = token

    async def _get(self, endpoint: str) -> dict | list:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "cv-analyzer",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = await self.client.get(endpoint, headers=headers)
            if response.status_code == 404:
                raise UpstreamError(
                    "The GitHub resource was not found", 404, "github_not_found"
                )
            if response.status_code in {403, 429}:
                raise UpstreamError(
                    "GitHub API rate limit or access policy prevented enrichment",
                    502,
                    "github_rate_limited",
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, (dict, list)):
                raise TypeError("Unexpected GitHub response shape")
            return payload
        except UpstreamError:
            raise
        except (httpx.TimeoutException, httpx.HTTPError, TypeError, ValueError) as exc:
            raise UpstreamError(
                "GitHub API enrichment failed", 502, "github_unavailable"
            ) from exc

    async def _readme_excerpt(self, owner: str, repo_name: str) -> str:
        try:
            readme = await self._get(
                f"https://api.github.com/repos/{owner}/{repo_name}/readme"
            )
        except UpstreamError as exc:
            if exc.status_code == 404:
                return ""
            raise
        if not isinstance(readme, dict):
            return ""
        encoded = readme.get("content", "")
        if not isinstance(encoded, str) or not encoded:
            return ""
        try:
            return base64.b64decode(encoded).decode("utf-8", errors="replace")[:10_000]
        except (binascii.Error, ValueError) as exc:
            raise UpstreamError(
                "GitHub README could not be decoded", 502, "github_invalid_response"
            ) from exc

    async def _profile_source(self, url: str, owner: str) -> dict:
        profile, repositories = await asyncio.gather(
            self._get(f"https://api.github.com/users/{owner}"),
            self._get(
                f"https://api.github.com/users/{owner}/repos?per_page=20&sort=updated"
            ),
        )
        if not isinstance(profile, dict) or not isinstance(repositories, list):
            raise UpstreamError(
                "GitHub returned an unexpected response", 502, "github_invalid_response"
            )
        excerpt = {
            "bio": profile.get("bio"),
            "public_repos": profile.get("public_repos"),
            "followers": profile.get("followers"),
            "created_at": profile.get("created_at"),
            "repositories": [
                {
                    "name": repo.get("name"),
                    "description": repo.get("description"),
                    "language": repo.get("language"),
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "updated_at": repo.get("updated_at"),
                }
                for repo in repositories
                if isinstance(repo, dict)
            ],
        }
        return {
            "id": f"github:{owner}",
            "url": url,
            "type": "github",
            "title": profile.get("name"),
            "excerpt": json.dumps(excerpt, ensure_ascii=False),
        }

    async def fetch(self, url: str) -> dict:
        if not is_github_url(url):
            raise ValueError("Not a GitHub URL")
        parsed = urlsplit(url)
        parts = [part for part in parsed.path.split("/") if part]
        if not parts:
            raise ValueError("Invalid GitHub URL")

        if (
            parsed.hostname == "api.github.com"
            and parts[0] == "users"
            and len(parts) >= 2
        ):
            return await self._profile_source(url, parts[1])
        if (
            parsed.hostname == "api.github.com"
            and parts[0] == "repos"
            and len(parts) >= 3
        ):
            parts = parts[1:]
        if len(parts) == 1:
            return await self._profile_source(url, parts[0])

        owner, repo_name = parts[0], parts[1]
        repo, languages, readme_excerpt = await asyncio.gather(
            self._get(f"https://api.github.com/repos/{owner}/{repo_name}"),
            self._get(f"https://api.github.com/repos/{owner}/{repo_name}/languages"),
            self._readme_excerpt(owner, repo_name),
        )
        if not isinstance(repo, dict) or not isinstance(languages, dict):
            raise UpstreamError(
                "GitHub returned an unexpected response", 502, "github_invalid_response"
            )

        license_data = repo.get("license")
        topics = repo.get("topics")
        excerpt = {
            "description": repo.get("description"),
            "html_url": repo.get("html_url"),
            "created_at": repo.get("created_at"),
            "pushed_at": repo.get("pushed_at"),
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "open_issues": repo.get("open_issues_count"),
            "language": repo.get("language"),
            "languages": languages,
            "topics": topics if isinstance(topics, list) else [],
            "license": license_data.get("spdx_id")
            if isinstance(license_data, dict)
            else None,
            "archived": repo.get("archived"),
            "fork": repo.get("fork"),
            "readme": readme_excerpt,
        }
        return {
            "id": f"github:{owner}/{repo_name}",
            "url": url,
            "type": "github",
            "title": repo.get("full_name"),
            "excerpt": json.dumps(excerpt, ensure_ascii=False),
        }
