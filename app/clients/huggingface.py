import asyncio
import json
from urllib.parse import quote, urlsplit

import httpx

from app.core.errors import UpstreamError
from app.utils.urls import is_huggingface_url

PROFILE_ARTIFACT_SOURCES = 5
README_BYTES = 10_000


class HuggingFaceClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 404:
            raise UpstreamError(
                "The Hugging Face resource was not found",
                404,
                "huggingface_not_found",
            )
        if response.status_code in {401, 403}:
            raise UpstreamError(
                "The Hugging Face resource is not publicly accessible",
                502,
                "website_access_restricted",
            )
        if response.status_code == 429:
            raise UpstreamError(
                "Hugging Face rate limited enrichment",
                502,
                "huggingface_rate_limited",
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
                "Hugging Face enrichment timed out", 504, "huggingface_timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Hugging Face API enrichment failed",
                502,
                "huggingface_unavailable",
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
                    remaining = README_BYTES - len(content)
                    if remaining <= 0:
                        break
                    content.extend(chunk[:remaining])
                return bytes(content).decode("utf-8", errors="replace")
        except UpstreamError:
            raise
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Hugging Face enrichment timed out", 504, "huggingface_timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                "Hugging Face API enrichment failed",
                502,
                "huggingface_unavailable",
            ) from exc

    async def _get(self, endpoint: str) -> dict | list:
        response = await self._request(endpoint)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "Hugging Face returned an unexpected response",
                502,
                "huggingface_invalid_response",
            ) from exc
        if not isinstance(payload, (dict, list)):
            raise UpstreamError(
                "Hugging Face returned an unexpected response",
                502,
                "huggingface_invalid_response",
            )
        return payload

    async def _readme(self, artifact_kind: str, artifact_id: str) -> str:
        prefixes = {"model": "", "dataset": "datasets/", "application": "spaces/"}
        prefix = prefixes[artifact_kind]
        encoded_id = "/".join(quote(part, safe="") for part in artifact_id.split("/"))
        try:
            return await self._read_text(
                f"https://huggingface.co/{prefix}{encoded_id}/raw/main/README.md"
            )
        except UpstreamError as exc:
            if exc.status_code == 404:
                return ""
            raise

    @staticmethod
    def _artifact_source(artifact: dict, artifact_kind: str) -> dict | None:
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or "/" not in artifact_id:
            return None
        path_prefixes = {"model": "", "dataset": "datasets/", "application": "spaces/"}
        tags = artifact.get("tags")
        excerpt = {
            "pipeline_tag": artifact.get("pipeline_tag"),
            "library_name": artifact.get("library_name"),
            "tags": [str(tag)[:100] for tag in tags[:50]]
            if isinstance(tags, list)
            else [],
            "downloads": artifact.get("downloads"),
            "likes": artifact.get("likes"),
            "last_modified": artifact.get("lastModified"),
        }
        return {
            "id": f"huggingface:{artifact_kind}:{artifact_id}",
            "url": f"https://huggingface.co/{path_prefixes[artifact_kind]}{artifact_id}",
            "type": "huggingface",
            "kind": artifact_kind,
            "title": artifact_id,
            "excerpt": json.dumps(excerpt, ensure_ascii=False),
        }

    async def _profile_sources(self, url: str, username: str) -> list[dict]:
        encoded_username = quote(username, safe="")
        models, datasets, spaces = await asyncio.gather(
            self._get(
                f"https://huggingface.co/api/models?author={encoded_username}&"
                "limit=10&sort=lastModified&direction=-1"
            ),
            self._get(
                f"https://huggingface.co/api/datasets?author={encoded_username}&"
                "limit=10&sort=lastModified&direction=-1"
            ),
            self._get(
                f"https://huggingface.co/api/spaces?author={encoded_username}&"
                "limit=10&sort=lastModified&direction=-1"
            ),
        )
        if not all(isinstance(items, list) for items in (models, datasets, spaces)):
            raise UpstreamError(
                "Hugging Face returned an unexpected response",
                502,
                "huggingface_invalid_response",
            )
        typed_artifacts = [
            *((item, "model") for item in models if isinstance(item, dict)),
            *((item, "dataset") for item in datasets if isinstance(item, dict)),
            *((item, "application") for item in spaces if isinstance(item, dict)),
        ]
        if not typed_artifacts:
            return []
        typed_artifacts.sort(
            key=lambda item: str(item[0].get("lastModified") or ""), reverse=True
        )
        profile_source = {
            "id": f"huggingface:profile:{username}",
            "url": url,
            "type": "huggingface",
            "kind": "profile",
            "title": username,
            "excerpt": json.dumps(
                {
                    "listed_models": len(models),
                    "listed_datasets": len(datasets),
                    "listed_spaces": len(spaces),
                },
                ensure_ascii=False,
            ),
        }
        sources = [profile_source]
        for artifact, artifact_kind in typed_artifacts:
            source = self._artifact_source(artifact, artifact_kind)
            if source is not None:
                sources.append(source)
            if len(sources) > PROFILE_ARTIFACT_SOURCES:
                break
        return sources

    async def _direct_artifact(
        self, artifact_kind: str, artifact_id: str
    ) -> list[dict]:
        api_kinds = {"model": "models", "dataset": "datasets", "application": "spaces"}
        encoded_id = "/".join(quote(part, safe="") for part in artifact_id.split("/"))
        artifact, readme = await asyncio.gather(
            self._get(
                f"https://huggingface.co/api/{api_kinds[artifact_kind]}/{encoded_id}"
            ),
            self._readme(artifact_kind, artifact_id),
        )
        if not isinstance(artifact, dict):
            raise UpstreamError(
                "Hugging Face returned an unexpected response",
                502,
                "huggingface_invalid_response",
            )
        source = self._artifact_source(artifact, artifact_kind)
        if source is None:
            raise UpstreamError(
                "Hugging Face returned an unexpected response",
                502,
                "huggingface_invalid_response",
            )
        excerpt = json.loads(source["excerpt"])
        excerpt["readme"] = readme
        source["excerpt"] = json.dumps(excerpt, ensure_ascii=False)
        return [source]

    async def fetch(self, url: str) -> list[dict]:
        if not is_huggingface_url(url):
            raise ValueError("Not a Hugging Face URL")
        parts = [part for part in urlsplit(url).path.split("/") if part]
        if not parts:
            raise ValueError("Invalid Hugging Face URL")
        prefixes = {"datasets": "dataset", "spaces": "application"}
        if parts[0] in prefixes:
            if len(parts) < 3:
                raise ValueError("Invalid Hugging Face artifact URL")
            return await self._direct_artifact(prefixes[parts[0]], "/".join(parts[1:3]))
        if len(parts) == 1:
            return await self._profile_sources(url, parts[0])
        return await self._direct_artifact("model", "/".join(parts[:2]))
