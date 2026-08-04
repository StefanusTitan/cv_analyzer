import traceback

from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from app.utils.log import logger


class LogError:
    async def request_validation_exception_handler(
        self, request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error_details = []
        for error in exc.errors():
            error_detail = {
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"],
            }
            error_details.append(error_detail)

        self._log_exception(
            request=request,
            status_code=400,
            message="Validation error",
            error={
                "type": "RequestValidationError",
                "details": error_details,
            },
            level="ERROR",
        )

        response = JSONResponse(
            status_code=400,
            content={
                "message": "Validation error",
                "result": None,
                "errors": error_details,
            },
        )
        response.headers["x-request-id"] = self._get_request_id(request)
        return response

    async def http_exception_handler(
        self, request: Request, exc: HTTPException
    ) -> JSONResponse:
        self._log_exception(
            request=request,
            status_code=exc.status_code,
            message=str(exc.detail),
            error={
                "type": "HTTPException",
                "detail": exc.detail,
            },
            level="ERROR" if exc.status_code >= 400 else "INFO",
        )

        response = JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.detail, "result": None, "errors": None},
        )
        response.headers["x-request-id"] = self._get_request_id(request)
        return response

    async def unhandled_exception_handler(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        self._log_exception(
            request=request,
            status_code=500,
            message="Unhandled exception",
            error={
                "type": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            },
            level="ERROR",
        )

        response = JSONResponse(
            status_code=500,
            content={
                "message": "Internal server error",
                "result": None,
                "errors": ["internal_server_error"],
            },
        )
        response.headers["x-request-id"] = self._get_request_id(request)
        return response

    def _log_exception(
        self, request: Request, status_code: int, message: str, error, level: str
    ):
        if getattr(request.state, "response_logged", False):
            return

        endpoint = (
            f"{request.url.path}?{request.query_params}"
            if request.query_params
            else request.url.path
        )
        client_ip = request.client.host if request.client else None

        logger.bind(
            request_id=self._get_request_id(request),
            method=request.method,
            url=endpoint,
            client_ip=client_ip,
            query_params=dict(request.query_params),
            request_payload=None,
            status_code=status_code,
            response_body=None,
            duration_ms=None,
            user_id=None,
            error=error,
        ).log(level, message)
        request.state.response_logged = True

    def _get_request_id(self, request: Request):
        return getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )
