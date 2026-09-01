from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.errors import AnalysisError
from app.schemas.cv import AnalyzeResponse

router = APIRouter(prefix="/cv", tags=["CV"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_cv(
    request: Request,
    # Accepted as a raw string and validated by the JobPostingClient so invalid IDs
    # produce the stable ``invalid_job_posting_id`` contract code instead of a
    # framework validation error shape.
    job_posting_id: Annotated[str, Form(...)],
    # Optional so a missing ``files`` field surfaces as the ``missing_files``
    # contract code rather than a framework validation error shape.
    files: Annotated[list[UploadFile] | None, File()] = None,
    links: Annotated[list[str] | None, Form()] = None,
    file_titles: Annotated[list[str] | None, Form()] = None,
    language: Annotated[str, Form()] = "id",
):
    try:
        result = await request.app.state.cv_analyzer.analyze(
            job_posting_id,
            files,
            request_id=getattr(request.state, "request_id", None),
            links=links,
            file_titles=file_titles,
            language=language,
        )
        return {"message": "CV analyzed successfully", "result": result}
    except AnalysisError as exc:
        request.state.analysis_error = {"code": exc.code, "message": exc.message}
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.message, "result": None, "errors": [exc.code]},
        )
