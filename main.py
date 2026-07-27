from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from app.exceptions.log import LogError
from app.middlewares.log import LogMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources here
    print("Application starting...")
    yield
    # Clean up resources here
    print("Application shutting down...")

app = FastAPI(lifespan=lifespan)
log = LogError()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging middleware
app.add_middleware(LogMiddleware)

# Exception handlers
app.add_exception_handler(RequestValidationError, log.request_validation_exception_handler)
app.add_exception_handler(HTTPException, log.http_exception_handler)
app.add_exception_handler(Exception, log.unhandled_exception_handler)

@app.get("/")
async def root():
    return {"message": "Server is UP ✅"}
