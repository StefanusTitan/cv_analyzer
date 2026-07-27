import asyncio

from app.middlewares.size_limit import RequestSizeLimitMiddleware


def run_request(messages, headers=None):
    sent = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    pending = list(messages)

    async def receive():
        return pending.pop(0)

    async def send(message):
        sent.append(message)

    middleware = RequestSizeLimitMiddleware(app, max_bytes=10)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/cv/analyze",
        "headers": headers or [],
    }
    asyncio.run(middleware(scope, receive, send))
    return sent


def response_status(messages):
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def test_rejects_declared_oversized_request():
    sent = run_request([], [(b"content-length", b"11")])
    assert response_status(sent) == 413


def test_rejects_streamed_oversized_request():
    sent = run_request(
        [
            {"type": "http.request", "body": b"123456", "more_body": True},
            {"type": "http.request", "body": b"78901", "more_body": False},
        ]
    )
    assert response_status(sent) == 413
