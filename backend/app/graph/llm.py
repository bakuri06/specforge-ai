import base64
import json
import logging
import time
from typing import Optional, Union

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def ollama_chat(
    model: str,
    prompt: str,
    images: Optional[list[str]] = None,
    expect_json: bool = False,
) -> Union[dict, str]:
    """Call Ollama's /api/chat for a single-turn completion.

    When expect_json is True, the model is asked to respond with strict JSON. If
    parsing still fails (local models under format: json occasionally wrap output
    in prose or fences anyway), one corrective retry is issued before giving up.
    """
    content = await _raw_chat(model, prompt, images, expect_json)
    if not expect_json:
        return content

    try:
        return _parse_json_with_repair(content)
    except json.JSONDecodeError:
        logger.warning(
            "ollama_chat: model=%s returned invalid JSON, retrying once", model
        )
        corrected_prompt = (
            f"{prompt}\n\n"
            "Your previous response was not valid JSON:\n"
            f"{content}\n\n"
            "Respond again with ONLY the JSON object described above. No prose, "
            "no markdown code fences, no explanation."
        )
        retry_content = await _raw_chat(model, corrected_prompt, images, expect_json)
        return _parse_json_with_repair(retry_content)


async def _raw_chat(
    model: str,
    prompt: str,
    images: Optional[list[str]],
    expect_json: bool,
) -> str:
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

    logger.info(
        "ollama_chat: calling model=%s images=%d prompt_chars=%d (this can take a "
        "while on CPU/limited RAM)",
        model,
        len(images or []),
        len(prompt),
    )
    started = time.monotonic()
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=180.0) as client:
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        content = response.json()["message"]["content"]
    elapsed = time.monotonic() - started
    logger.info(
        "ollama_chat: model=%s responded in %.1fs (%d chars)",
        model,
        elapsed,
        len(content),
    )
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
