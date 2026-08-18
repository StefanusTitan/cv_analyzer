import pytest

from app.core.config import Settings


def settings_for(job_posting_url: str) -> Settings:
    return Settings(
        _env_file=None,
        dashscope_api_key="test-key",
        scraper_worker_token="test-token",
        job_posting_api_url=job_posting_url,
    )


def test_default_dashscope_model_is_qwen_flash_release():
    assert (
        Settings.model_fields["dashscope_model"].default
        == "qwen3.7-flash-2026-07-15"
    )


def test_runtime_accepts_https_job_posting_api():
    settings_for("https://gateway.test/api/proxy/v3/employees/job-posting").validate_runtime()


def test_runtime_rejects_nonlocal_http_job_posting_api():
    with pytest.raises(RuntimeError, match="must use HTTPS"):
        settings_for("http://employees.example.test/api/v3/job-posting").validate_runtime()


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "[::1]", "host.docker.internal"],
)
def test_runtime_accepts_local_http_job_posting_api(host):
    settings_for(f"http://{host}:5555/api/v3/job-posting").validate_runtime()


def test_runtime_never_allows_arbitrary_insecure_job_posting_api():
    with pytest.raises(RuntimeError, match="must use HTTPS"):
        settings_for("http://employees.example.test/api/v3/job-posting").validate_runtime()
