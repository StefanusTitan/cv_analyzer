import json
import os
import time
import uuid
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.utils.log import logger


class LogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = logger
        self.max_body_preview_length = self._get_int_env(
            "LOG_MAX_BODY_PREVIEW_LENGTH", 1000
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            body = b""
            request_payload = {
                "content_type": content_type,
                "size_bytes": request.headers.get("content-length"),
                "multipart": True,
            }
        else:
            body = await request.body()
            request_payload = await self._extract_request_payload(request, body)
            await self._restore_request_body(request, body)
        response = await call_next(request)

        if request.url.path == "/cv/analyze":
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            self.logger.bind(
                request_id=request_id,
                method=request.method,
                url=request.url.path,
                client_ip=request.client.host if request.client else None,
                query_params=dict(request.query_params),
                request_payload=request_payload,
                status_code=response.status_code,
                response_body={"omitted": True, "reason": "contains_candidate_data"},
                duration_ms=duration_ms,
                user_id=None,
            ).log(
                "INFO" if response.status_code < 400 else "ERROR",
                "Success" if response.status_code < 400 else "Fail",
            )
            request.state.response_logged = True
            response.headers["x-request-id"] = request_id
            return response

        # Capture response body for non-sensitive endpoints.
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # Reconstruct response since body_iterator was consumed
        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        endpoint = (
            f"{request.url.path}?{request.query_params}"
            if request.query_params
            else request.url.path
        )
        client_ip = request.client.host if request.client else None
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        try:
            resp_json = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            resp_json = (
                response_body.decode("utf-8", errors="replace")
                if response_body
                else None
            )

        log_level = "INFO" if response.status_code < 400 else "ERROR"
        message = "Success" if response.status_code < 400 else "Fail"

        self.logger.bind(
            request_id=request_id,
            method=request.method,
            url=endpoint,
            client_ip=client_ip,
            query_params=dict(request.query_params),
            request_payload=request_payload,
            status_code=response.status_code,
            response_body=resp_json,
            duration_ms=duration_ms,
            user_id=None,
        ).log(log_level, message)
        request.state.response_logged = True

        response.headers["x-request-id"] = request_id

        return response

    async def _extract_request_payload(self, request: Request, body: bytes):
        content_type = request.headers.get("content-type", "")

        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return None

        if "application/json" in content_type:
            if not body:
                return None

            try:
                return json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return body.decode("utf-8", errors="replace")

        if "application/x-www-form-urlencoded" in content_type:
            if not body:
                return None
            parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
            return {
                key: values[0] if len(values) == 1 else values
                for key, values in parsed.items()
            }

        if not body:
            return None

        return {
            "content_type": content_type or None,
            "size_bytes": len(body),
            "preview": body[: self.max_body_preview_length].decode(
                "utf-8", errors="replace"
            ),
        }

    async def _restore_request_body(self, request: Request, body: bytes):
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = receive

    def _get_int_env(self, env_name: str, default: int):
        raw_value = os.getenv(env_name)
        if raw_value is None:
            return default
        try:
            return int(raw_value)
        except ValueError:
            return default
