import pytest

from app.core.config import Settings


def settings_for(gateway_url: str, *, allow_local: bool = False) -> Settings:
    return Settings(
        _env_file=None,
        dashscope_api_key="test-key",
        scraper_worker_token="test-token",
        gateway_api_url=gateway_url,
        allow_insecure_local_gateway=allow_local,
    )


def test_runtime_accepts_https_gateway():
    settings_for("https://gateway.test/api").validate_runtime()


def test_runtime_rejects_http_gateway_by_default():
    with pytest.raises(RuntimeError, match="must use HTTPS"):
        settings_for("http://127.0.0.1:9996/api").validate_runtime()


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "[::1]", "host.docker.internal"],
)
def test_runtime_accepts_explicit_local_http_gateway(host):
    settings_for(
        f"http://{host}:9996/api",
        allow_local=True,
    ).validate_runtime()


def test_runtime_never_allows_arbitrary_insecure_gateway():
    with pytest.raises(RuntimeError, match="must use HTTPS"):
        settings_for(
            "http://gateway.example.test/api",
            allow_local=True,
        ).validate_runtime()
