from collections.abc import Callable
from typing import Any

from starlette.responses import JSONResponse


class RequestTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app: Callable, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable):
        if (
            scope["type"] != "http"
            or scope.get("path") != "/cv/analyze"
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        try:
            if int(headers.get(b"content-length", b"0")) > self.max_bytes:
                response = JSONResponse(
                    {
                        "message": "Request body too large",
                        "result": None,
                        "errors": ["request_size_exceeded"],
                    },
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        except ValueError:
            pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge
            return message

        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestTooLarge:
            if response_started:
                raise
            response = JSONResponse(
                {
                    "message": "Request body too large",
                    "result": None,
                    "errors": ["request_size_exceeded"],
                },
                status_code=413,
            )
            await response(scope, receive, send)
