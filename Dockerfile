FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8003

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port 8003 --timeout-keep-alive 300 --workers ${UVICORN_WORKERS:-2}"]