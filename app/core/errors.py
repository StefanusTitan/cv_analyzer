class AnalysisError(Exception):
    def __init__(self, message: str, status_code: int, code: str):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class UploadError(AnalysisError):
    def __init__(
        self, message: str, status_code: int = 400, code: str = "invalid_upload"
    ):
        super().__init__(message, status_code, code)


class UpstreamError(AnalysisError):
    def __init__(
        self, message: str, status_code: int = 502, code: str = "upstream_error"
    ):
        super().__init__(message, status_code, code)
