from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.errors import AnalysisError
from app.schemas.cv import AnalyzeResponse

router = APIRouter(prefix="/cv", tags=["CV"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_cv(
    request: Request,
    job_title: Annotated[str, Form(max_length=200)],
    files: Annotated[list[UploadFile], File(...)],
):
    normalized_job_title = job_title.strip()
    if not normalized_job_title:
        return JSONResponse(
            status_code=400,
            content={
                "message": "job_title must not be blank",
                "result": None,
                "errors": ["invalid_job_title"],
            },
        )
    try:
        result = await request.app.state.cv_analyzer.analyze(
            normalized_job_title,
            files,
            request_id=getattr(request.state, "request_id", None),
        )
        return {"message": "CV analyzed successfully", "result": result}
    except AnalysisError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.message, "result": None, "errors": [exc.code]},
        )
