import httpx
from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def list_models():
    """List models actually pulled in the local Ollama daemon, plus the
    server's configured defaults for each role, so the frontend can offer a
    selector without hardcoding model names."""
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=10.0) as client:
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()

    names = sorted(model["name"] for model in data.get("models", []))
    return {
        "models": names,
        "defaults": {
            "vision_model": settings.vision_model,
            "reasoning_model": settings.reasoning_model,
            "formatter_model": settings.formatter_model,
        },
    }
