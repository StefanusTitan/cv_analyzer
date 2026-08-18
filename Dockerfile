# syntax=docker/dockerfile:1
FROM python:3.14-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HOME=/tmp
RUN playwright install --with-deps chromium \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /tmp --no-create-home app \
    && chmod -R a+rX /ms-playwright
COPY --chown=app:app . .
USER app

EXPOSE 8003 8010
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8003 --timeout-keep-alive 300 --workers ${UVICORN_WORKERS:-1} --no-access-log"]
