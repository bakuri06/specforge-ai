import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import models, session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

app = FastAPI(title="SpecForge AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(models.router)


@app.exception_handler(httpx.TimeoutException)
async def ollama_timeout_handler(request: Request, exc: httpx.TimeoutException):
    return JSONResponse(
        status_code=504,
        content={
            "detail": (
                "The local model took too long to respond (timed out after "
                f"{settings.ollama_timeout_seconds:.0f}s). This is common for "
                "large models on limited hardware. Check that Ollama is still "
                "running, or raise OLLAMA_TIMEOUT_SECONDS in backend/.env."
            )
        },
    )


@app.exception_handler(httpx.ConnectError)
async def ollama_connect_error_handler(request: Request, exc: httpx.ConnectError):
    return JSONResponse(
        status_code=502,
        content={
            "detail": (
                f"Could not reach Ollama at {settings.ollama_base_url}. Make "
                "sure the Ollama daemon is running."
            )
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
