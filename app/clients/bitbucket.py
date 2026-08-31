import asyncio
import json
from urllib.parse import quote, urlsplit

import httpx

from app.core.errors import UpstreamError
from app.utils.urls import is_bitbucket_url

PROFILE_REPOSITORY_SOURCES = 5
TEXT_FILE_BYTES = 10_000
README_EXCERPT_CHARS = 4_000


class BitbucketClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 404:
            raise UpstreamError(
                "The Bitbucket resource was not found",
                404,
                "bitbucket_not_found",
            )
        if response.status_code in {401, 403}:
            raise UpstreamError(
                "The Bitbucket resource is not publicly accessible",
                502,
                "website_access_restricted",
            )
        if response.status_code == 429:
            raise UpstreamError(
                "Bitbucket rate limited enrichment",
                502,
                "bitbucket_rate_limited",
            )
        response.raise_for_status()

    async def _request(self, endpoint: str) -> httpx.Response:
        try:
            response = await self.client.get(
                endpoint,
                headers={"Accept": "application/json", "User-Agent": "cv-analyzer"},
            )
            self._raise_for_status(response)
            return response
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Bitbucket enrichment timed out", 504, "bitbucket_timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Bitbucket API enrichment failed", 502, "bitbucket_unavailable"
            ) from exc

    async def _read_text(self, endpoint: str) -> str:
        try:
            async with self.client.stream(
                "GET",
                endpoint,
                headers={"Accept": "text/plain", "User-Agent": "cv-analyzer"},
            ) as response:
                self._raise_for_status(response)
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    remaining = TEXT_FILE_BYTES - len(content)
                    if remaining <= 0:
                        break
                    content.extend(chunk[:remaining])
                return bytes(content).decode("utf-8", errors="replace")
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Bitbucket enrichment timed out", 504, "bitbucket_timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Bitbucket API enrichment failed", 502, "bitbucket_unavailable"
            ) from exc

    async def _get(self, endpoint: str) -> dict | list:
        response = await self._request(endpoint)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Bitbucket returned an unexpected response",
                502,
                "bitbucket_invalid_response",
            ) from exc
        if not isinstance(payload, (dict, list)):
            raise UpstreamError(
                "Bitbucket returned an unexpected response",
                502,
                "bitbucket_invalid_response",
            )
        return payload

    async def _text_file(
        self, workspace: str, repository: str, reference: str, path: str
    ) -> str:
        try:
            return await self._read_text(
                "https://api.bitbucket.org/2.0/repositories/"
                f"{quote(workspace, safe='')}/{quote(repository, safe='')}/src/"
                f"{quote(reference, safe='')}/{quote(path, safe='')}"
            )
        except UpstreamError as exc:
            if exc.status_code == 404:
                return ""
            raise

    async def _package_manifest(
        self, workspace: str, repository: str, reference: str
    ) -> dict | None:
        raw = await self._text_file(workspace, repository, reference, "package.json")
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
    def _repository_source(repository: dict) -> dict | None:
        full_name = repository.get("full_name")
        if not isinstance(full_name, str) or "/" not in full_name:
            return None
        links = repository.get("links")
        html_link = links.get("html") if isinstance(links, dict) else None
        url = html_link.get("href") if isinstance(html_link, dict) else None
        excerpt = {
            "description": repository.get("description"),
            "language": repository.get("language"),
            "created_on": repository.get("created_on"),
            "updated_on": repository.get("updated_on"),
            "size": repository.get("size"),
            "is_private": repository.get("is_private"),
        }
        return {
            "id": f"bitbucket:{full_name}",
            "url": str(url or f"https://bitbucket.org/{full_name}"),
            "type": "bitbucket",
            "title": full_name,
            "excerpt": json.dumps(excerpt, ensure_ascii=False),
        }

    async def _profile_repository_source(self, repository: dict) -> dict | None:
        source = self._repository_source(repository)
        if source is None:
            return None
        workspace, repository_name = str(repository["full_name"]).split("/", 1)
        main_branch = repository.get("mainbranch")
        reference = (
            str(main_branch.get("name"))
            if isinstance(main_branch, dict) and main_branch.get("name")
            else "HEAD"
        )
        readme, package = await asyncio.gather(
            self._text_file(workspace, repository_name, reference, "README.md"),
            self._package_manifest(workspace, repository_name, reference),
        )
        excerpt = json.loads(source["excerpt"])
        excerpt.update({"readme": readme[:README_EXCERPT_CHARS], "package": package})
        source["excerpt"] = json.dumps(excerpt, ensure_ascii=False)
        return source

    async def _profile_sources(self, url: str, workspace: str) -> list[dict]:
        repositories = await self._get(
            f"https://api.bitbucket.org/2.0/repositories/"
            f"{quote(workspace, safe='')}?pagelen=10&sort=-updated_on"
        )
        if not isinstance(repositories, dict):
            raise UpstreamError(
                "Bitbucket returned an unexpected response",
                502,
                "bitbucket_invalid_response",
            )
        values = repositories.get("values")
        if not isinstance(values, list):
            raise UpstreamError(
                "Bitbucket returned an unexpected response",
                502,
                "bitbucket_invalid_response",
            )
        profile_source = {
            "id": f"bitbucket:{workspace}",
            "url": url,
            "type": "bitbucket",
            "title": workspace,
            "excerpt": json.dumps(
                {"public_repositories": repositories.get("size")},
                ensure_ascii=False,
            ),
        }
        sources = [profile_source]
        selected_repositories: list[dict] = []
        for repository in values:
            if not isinstance(repository, dict) or repository.get("is_private"):
                continue
            selected_repositories.append(repository)
            if len(selected_repositories) >= PROFILE_REPOSITORY_SOURCES:
                break
        enriched_repositories = await asyncio.gather(
            *(
                self._profile_repository_source(repository)
                for repository in selected_repositories
            )
        )
        sources.extend(source for source in enriched_repositories if source is not None)
        return sources

    async def _direct_repository(
        self, url: str, workspace: str, repository: str
    ) -> list[dict]:
        base = (
            "https://api.bitbucket.org/2.0/repositories/"
            f"{quote(workspace, safe='')}/{quote(repository, safe='')}"
        )
        repository_info = await self._get(base)
        if not isinstance(repository_info, dict):
            raise UpstreamError(
                "Bitbucket returned an unexpected response",
                502,
                "bitbucket_invalid_response",
            )
        if repository_info.get("is_private"):
            raise UpstreamError(
                "The Bitbucket resource is not publicly accessible",
                502,
                "website_access_restricted",
            )
        main_branch = repository_info.get("mainbranch")
        reference = (
            str(main_branch.get("name"))
            if isinstance(main_branch, dict) and main_branch.get("name")
            else "HEAD"
        )
        readme, package = await asyncio.gather(
            self._text_file(workspace, repository, reference, "README.md"),
            self._package_manifest(workspace, repository, reference),
        )
        excerpt = {
            "description": repository_info.get("description"),
            "language": repository_info.get("language"),
            "scm": repository_info.get("scm"),
            "created_on": repository_info.get("created_on"),
            "updated_on": repository_info.get("updated_on"),
            "size": repository_info.get("size"),
            "has_issues": repository_info.get("has_issues"),
            "is_fork": bool(repository_info.get("parent")),
            "readme": readme[:README_EXCERPT_CHARS],
            "package": package,
        }
        full_name = str(repository_info.get("full_name") or f"{workspace}/{repository}")
        links = repository_info.get("links")
        html_link = links.get("html") if isinstance(links, dict) else None
        source_url = html_link.get("href") if isinstance(html_link, dict) else None
        return [
            {
                "id": f"bitbucket:{full_name}",
                "url": str(source_url or url),
                "type": "bitbucket",
                "title": full_name,
                "excerpt": json.dumps(excerpt, ensure_ascii=False),
            }
        ]

    async def fetch(self, url: str) -> list[dict]:
        if not is_bitbucket_url(url):
            raise ValueError("Not a Bitbucket URL")
        parts = [part for part in urlsplit(url).path.split("/") if part]
        if not parts:
            raise ValueError("Invalid Bitbucket URL")
        if len(parts) == 1:
            return await self._profile_sources(url, parts[0])
        return await self._direct_repository(url, parts[0], parts[1])
