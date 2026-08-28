import asyncio
import json

import httpx

from app.clients.huggingface import HuggingFaceClient
from app.core.errors import UpstreamError


def test_profile_fetch_emits_recent_public_artifacts():
    def handler(request: httpx.Request) -> httpx.Response:
        artifacts = {
            "/api/models": [
                {
                    "id": "candidate/model",
                    "pipeline_tag": "text-classification",
                    "lastModified": "2026-08-03T00:00:00Z",
                }
            ],
            "/api/datasets": [
                {
                    "id": "candidate/dataset",
                    "tags": ["language:en"],
                    "lastModified": "2026-08-02T00:00:00Z",
                }
            ],
            "/api/spaces": [
                {
                    "id": "candidate/application",
                    "lastModified": "2026-08-01T00:00:00Z",
                }
            ],
        }
        return httpx.Response(200, json=artifacts[request.url.path])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HuggingFaceClient(client).fetch(
                "https://huggingface.co/candidate"
            )

    sources = asyncio.run(run())

    assert [source["kind"] for source in sources] == [
        "profile",
        "model",
        "dataset",
        "application",
    ]
    profile = json.loads(sources[0]["excerpt"])
    assert profile == {
        "listed_models": 1,
        "listed_datasets": 1,
        "listed_spaces": 1,
    }
    assert sources[2]["url"] == "https://huggingface.co/datasets/candidate/dataset"


def test_profile_without_public_artifacts_does_not_create_evidence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HuggingFaceClient(client).fetch(
                "https://huggingface.co/candidate"
            )

    assert asyncio.run(run()) == []


def test_dataset_fetch_includes_metadata_and_readme():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/datasets/candidate/example":
            return httpx.Response(
                200,
                json={
                    "id": "candidate/example",
                    "tags": ["task_categories:text-classification"],
                    "downloads": 12,
                    "likes": 3,
                    "lastModified": "2026-08-01T00:00:00Z",
                },
            )
        if request.url.path == "/datasets/candidate/example/raw/main/README.md":
            return httpx.Response(
                200, text="Dataset card with collection details" + "x" * 20_000
            )
        raise AssertionError(f"Unexpected endpoint {request.url}")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HuggingFaceClient(client).fetch(
                "https://huggingface.co/datasets/candidate/example"
            )

    source = asyncio.run(run())[0]
    excerpt = json.loads(source["excerpt"])

    assert source["id"] == "huggingface:dataset:candidate/example"
    assert source["kind"] == "dataset"
    assert excerpt["downloads"] == 12
    assert excerpt["readme"].startswith("Dataset card with collection details")
    assert len(excerpt["readme"].encode()) == 10_000


def test_gated_artifact_maps_to_restricted_access():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await HuggingFaceClient(client).fetch(
                "https://huggingface.co/candidate/gated-model"
            )

    try:
        asyncio.run(run())
    except UpstreamError as exc:
        assert exc.code == "website_access_restricted"
    else:
        raise AssertionError("expected website_access_restricted")
