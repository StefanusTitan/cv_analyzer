# Duluin Machine Learning

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (Astral's Python package and project manager)

## Installation

Create the Python 3.12 virtual environment and install the locked dependencies:

```bash
uv sync --python 3.12
```

The project includes `PyMuPDF` for PDF processing and the OpenAI SDK. The
`uv sync` command installs the dependencies listed in `pyproject.toml` and uses
`uv.lock` for reproducible installs.

## Configuration

Set application options in the appropriate file under `environment/`, then run
with that file loaded locally. Docker Compose uses `environment/.env.staging`.

CV analysis configuration is read from environment variables. The required values are `DASHSCOPE_API_KEY` and optionally `GITHUB_ACCESS_TOKEN`; the default DashScope endpoint is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` with model `qwen3.7-flash` and thinking disabled. The API accepts PDF and DOCX uploads at `POST /cv/analyze` as multipart fields `job_posting_id` and `files`. It retrieves the job title and HTML description from `${GATEWAY_API_URL}/proxy/v3/employees/job-posting/{job_posting_id}`, converts the description to plain text, and returns a one-paragraph hiring assessment with trusted evidence sources and warnings.

Set the reverse proxy or API gateway request-body limit at or below `REQUEST_MAX_SIZE_BYTES` (25 MB by default).

Browser scraping runs in a second container built from this same project image. The worker has no direct internet or product-network connection: Chromium must use the Squid egress proxy, which rejects private, loopback, link-local, carrier-NAT, multicast, reserved, and cloud-metadata destinations. The worker endpoint is authenticated with `SCRAPER_WORKER_TOKEN` and is reachable only on the internal `scraper_control` network.

Before deployment, export secrets without committing them:

```bash
export DASHSCOPE_API_KEY='...'
export GITHUB_ACCESS_TOKEN='...'
export SCRAPER_WORKER_TOKEN="$(openssl rand -hex 32)"
```

CORS is intentionally fixed in code to `https://workin-dev.duluin.id`; it is not configurable through environment variables. CORS is enabled only on the analyzer, while the scraper worker remains inaccessible to browsers.

`X-Forwarded-Host` is accepted as a request header for preflight compatibility, but it is not trusted for CORS or authorization decisions. The gateway should remove any client-supplied value and set it itself. A standards-compliant value is a host such as `workin-dev.duluin.id`, not a complete URL; the browser automatically sends `Origin: https://workin-dev.duluin.id` for CORS validation.

The Squid image is pinned to an immutable multi-architecture digest. Update that digest deliberately as part of dependency maintenance and vulnerability patching.

Logging options include:

```text
LOG_MASKING_ENABLED=true
LOG_MASK_FIELDS=nik,token,password,secret
LOG_UNMASK_FIELDS=company_id
LOG_MAX_VALUE_LENGTH=2000
LOG_MAX_COLLECTION_ITEMS=30
LOG_MAX_BODY_PREVIEW_LENGTH=1000
LOG_PRETTY_JSON=false
```

## Run

For normal development and staging, run the complete isolated stack:

```bash
docker compose up --build
```

The analyzer listens on container port `8003` and is exposed only to `apigateway_internal_network`; the scraper worker and proxy publish no host ports. The gateway should call `http://cv_analyzer:8003`.

For Python-only tests:

```bash
uv run pytest -q
uv run ruff check main.py app tests
```
