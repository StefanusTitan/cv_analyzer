import asyncio
import json
from urllib.parse import quote, urlsplit

import httpx

from app.core.errors import UpstreamError
from app.utils.urls import is_gitlab_url

PROFILE_PROJECT_SOURCES = 5
README_EXCERPT_CHARS = 4_000


class GitlabClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def _request(self, endpoint: str) -> httpx.Response:
        try:
            response = await self.client.get(
                endpoint,
                headers={"Accept": "application/json", "User-Agent": "cv-analyzer"},
            )
            if response.status_code == 404:
                raise UpstreamError(
                    "The GitLab resource was not found", 404, "gitlab_not_found"
                )
            if response.status_code in {401, 403}:
                raise UpstreamError(
                    "The GitLab resource is not publicly accessible",
                    502,
                    "website_access_restricted",
                )
            if response.status_code == 429:
                raise UpstreamError(
                    "GitLab rate limited enrichment", 502, "gitlab_rate_limited"
                )
            response.raise_for_status()
            return response
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "GitLab enrichment timed out", 504, "gitlab_timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "GitLab API enrichment failed", 502, "gitlab_unavailable"
            ) from exc

    async def _get(self, endpoint: str) -> dict | list:
        response = await self._request(endpoint)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            ) from exc
        if not isinstance(payload, (dict, list)):
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            )
        return payload

    async def _text_file(self, project: str, path: str) -> str:
        project_id = quote(project, safe="")
        file_path = quote(path, safe="")
        try:
            response = await self._request(
                f"https://gitlab.com/api/v4/projects/{project_id}/repository/files/"
                f"{file_path}/raw?ref=HEAD"
            )
        except UpstreamError as exc:
            if exc.status_code == 404:
                return ""
            raise
        return response.text[:10_000]

    async def _package_manifest(self, project: str) -> dict | None:
        raw = await self._text_file(project, "package.json")
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        summary: dict[str, str | list[str]] = {}
        name = payload.get("name")
        if isinstance(name, str) and name:
            summary["name"] = name[:200]
        for key in ("dependencies", "devDependencies"):
            values = payload.get(key)
            if isinstance(values, dict):
                summary[key] = sorted(str(value)[:200] for value in values)[:100]
        scripts = payload.get("scripts")
        if isinstance(scripts, dict):
            summary["scripts"] = sorted(str(value)[:200] for value in scripts)[:50]
        return summary or None

    @staticmethod
    def _listed_project_source(project: dict) -> dict | None:
        path = project.get("path_with_namespace")
        if not isinstance(path, str) or not path:
            return None
        excerpt = {
            "description": project.get("description"),
            "topics": project.get("topics")
            if isinstance(project.get("topics"), list)
            else [],
            "stars": project.get("star_count"),
            "forks": project.get("forks_count"),
            "last_activity_at": project.get("last_activity_at"),
        }
        return {
            "id": f"gitlab:{path}",
            "url": str(project.get("web_url") or f"https://gitlab.com/{path}"),
            "type": "gitlab",
            "title": str(project.get("name_with_namespace") or path),
            "excerpt": json.dumps(excerpt, ensure_ascii=False),
        }

    async def _profile_project_source(self, project: dict) -> dict | None:
        source = self._listed_project_source(project)
        if source is None:
            return None
        project_path = str(project["path_with_namespace"])
        project_id = quote(project_path, safe="")
        languages, readme, package = await asyncio.gather(
            self._get(f"https://gitlab.com/api/v4/projects/{project_id}/languages"),
            self._text_file(project_path, "README.md"),
            self._package_manifest(project_path),
        )
        if not isinstance(languages, dict):
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            )
        excerpt = json.loads(source["excerpt"])
        excerpt.update(
            {
                "languages": languages,
                "readme": readme[:README_EXCERPT_CHARS],
                "package": package,
            }
        )
        source["excerpt"] = json.dumps(excerpt, ensure_ascii=False)
        return source

    async def _profile_sources(self, url: str, username: str) -> list[dict]:
        users = await self._get(
            f"https://gitlab.com/api/v4/users?username={quote(username, safe='')}"
        )
        if not isinstance(users, list) or not users or not isinstance(users[0], dict):
            raise UpstreamError(
                "The GitLab profile was not found", 404, "gitlab_not_found"
            )
        profile = users[0]
        user_id = profile.get("id")
        if not isinstance(user_id, int):
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            )
        projects = await self._get(
            f"https://gitlab.com/api/v4/users/{user_id}/projects?visibility=public&"
            "order_by=last_activity_at&sort=desc&per_page=20&simple=true"
        )
        if not isinstance(projects, list):
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            )
        profile_excerpt = {
            "bio": profile.get("bio"),
            "job_title": profile.get("job_title"),
            "organization": profile.get("organization"),
            "location": profile.get("location"),
            "website_url": profile.get("website_url"),
        }
        sources = [
            {
                "id": f"gitlab:{username}",
                "url": url,
                "type": "gitlab",
                "title": profile.get("name"),
                "excerpt": json.dumps(profile_excerpt, ensure_ascii=False),
            }
        ]
        selected_projects: list[dict] = []
        for project in projects:
            if not isinstance(project, dict):
                continue
            selected_projects.append(project)
            if len(selected_projects) >= PROFILE_PROJECT_SOURCES:
                break
        enriched_projects = await asyncio.gather(
            *(self._profile_project_source(project) for project in selected_projects)
        )
        sources.extend(source for source in enriched_projects if source is not None)
        return sources

    async def _project_source(self, url: str, project: str) -> list[dict]:
        project_id = quote(project, safe="")
        project_info, languages, readme, package = await asyncio.gather(
            self._get(f"https://gitlab.com/api/v4/projects/{project_id}?license=true"),
            self._get(f"https://gitlab.com/api/v4/projects/{project_id}/languages"),
            self._text_file(project, "README.md"),
            self._package_manifest(project),
        )
        if not isinstance(project_info, dict) or not isinstance(languages, dict):
            raise UpstreamError(
                "GitLab returned an unexpected response",
                502,
                "gitlab_invalid_response",
            )
        license_data = project_info.get("license")
        excerpt = {
            "description": project_info.get("description"),
            "created_at": project_info.get("created_at"),
            "last_activity_at": project_info.get("last_activity_at"),
            "stars": project_info.get("star_count"),
            "forks": project_info.get("forks_count"),
            "open_issues": project_info.get("open_issues_count"),
            "languages": languages,
            "topics": project_info.get("topics")
            if isinstance(project_info.get("topics"), list)
            else [],
            "license": license_data.get("key")
            if isinstance(license_data, dict)
            else None,
            "archived": project_info.get("archived"),
            "readme": readme[:README_EXCERPT_CHARS],
            "package": package,
        }
        return [
            {
                "id": f"gitlab:{project}",
                "url": str(project_info.get("web_url") or url),
                "type": "gitlab",
                "title": project_info.get("name_with_namespace"),
                "excerpt": json.dumps(excerpt, ensure_ascii=False),
            }
        ]

    async def fetch(self, url: str) -> list[dict]:
        if not is_gitlab_url(url):
            raise ValueError("Not a GitLab URL")
        parts = [part for part in urlsplit(url).path.split("/") if part]
        if not parts:
            raise ValueError("Invalid GitLab URL")
        project_parts = parts[: parts.index("-")] if "-" in parts else parts
        if len(project_parts) == 1:
            return await self._profile_sources(url, project_parts[0])
        return await self._project_source(url, "/".join(project_parts))
