from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.errors import AnalysisError
from app.schemas.cv import AnalyzeResponse

router = APIRouter(prefix="/cv", tags=["CV"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_cv(
    request: Request,
    job_posting_id: Annotated[UUID, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
):
    try:
        result = await request.app.state.cv_analyzer.analyze(
            str(job_posting_id),
            files,
            request_id=getattr(request.state, "request_id", None),
        )
        return {"message": "CV analyzed successfully", "result": result}
    except AnalysisError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": exc.message, "result": None, "errors": [exc.code]},
        )
