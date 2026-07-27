# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8003

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8003 --timeout-keep-alive 300 --workers ${UVICORN_WORKERS:-2}"]

