import base64
import json

import httpx

from app.config import settings


async def ollama_chat(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    expect_json: bool = False,
) -> dict | str:
    """Call Ollama's /api/chat for a single-turn completion.

    When expect_json is True, the model is asked to respond with strict JSON and
    the result is repaired/parsed into a dict. Ollama isn't reachable from this
    sandbox, so this has not been exercised against a live daemon yet.
    """
    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = [_encode_image(path) for path in images]

    payload: dict = {
        "model": model,
        "messages": [message],
        "stream": False,
    }
    if expect_json:
        payload["format"] = "json"

    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=180.0) as client:
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"]

    if expect_json:
        return _parse_json_with_repair(content)
    return content


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_with_repair(content: str) -> dict:
    """Best-effort recovery when the model wraps JSON in prose or code fences."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise
