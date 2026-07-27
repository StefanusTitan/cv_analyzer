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

CV analysis configuration is read from environment variables. The required values are `DASHSCOPE_API_KEY` and optionally `GITHUB_ACCESS_TOKEN`; the default DashScope endpoint is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` with model `qwen3.7-flash`. The API accepts PDF and DOCX uploads at `POST /cv/analyze` as multipart fields `job_title` and `files`, and returns a narrative hiring assessment with trusted evidence sources and warnings.

Set the reverse proxy or API gateway request-body limit at or below `REQUEST_MAX_SIZE_BYTES` (25 MB by default). Generic browser scraping must also run behind outbound network controls that block private, loopback, link-local, and cloud metadata destinations to close DNS-rebinding risks that application-level DNS checks cannot eliminate.

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

```bash
uv run uvicorn main:app --reload
```

The API listens on port 8000 locally. To run the containerized service:

```bash
docker compose up --build
```

Docker exposes the service on port 8001.
