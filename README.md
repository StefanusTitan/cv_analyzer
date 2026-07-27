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
