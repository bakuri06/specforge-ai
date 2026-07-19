import base64
import json
import logging
import re
import time
from typing import Optional, Union

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


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
            "no markdown code fences, no <think> reasoning blocks, no "
            "explanation. Escape every newline inside a string value as \\n "
            "rather than a literal line break, escape every literal "
            "backslash as \\\\ (e.g. a regex pattern like \\d must be "
            "written as \\\\d), and do not leave a trailing comma before a "
            "closing } or ]."
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
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds
    ) as client:
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


def _fix_invalid_escapes(text: str) -> str:
    """Escape any backslash that isn't already part of a valid JSON escape
    sequence (\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t, \\u). Local models
    routinely emit a literal backslash that was never meant as a JSON escape
    at all - e.g. a regex-style pattern in prose ("must match \\d{6}") or a
    Windows path - without doubling it, which json.loads rejects outright as
    "Invalid \\escape" regardless of strict=False (that flag only relaxes
    control characters, not malformed escape sequences)."""
    return _INVALID_ESCAPE_RE.sub(r"\\\\", text)


def _remove_trailing_commas(text: str) -> str:
    """Strip a comma immediately before a closing }/] - valid in JS object/
    array literals (which these models clearly draw on), invalid in strict
    JSON, and surfaces as a confusing "Expecting value" error rather than
    anything mentioning a comma."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _parse_json_with_repair(content: str) -> dict:
    """Best-effort recovery for common local-model JSON quirks:

    - Reasoning models (DeepSeek-R1) prefixing the answer with a
      <think>...</think> block even under format: json.
    - Prose or markdown code fences wrapped around the JSON object.
    - Literal, unescaped newlines/control characters inside string values
      (e.g. a multi-paragraph markdown spec) instead of an escaped \\n —
      json.loads(strict=False) allows these rather than rejecting them.
    - A literal backslash inside a string value that isn't a valid JSON
      escape (see _fix_invalid_escapes).
    - A trailing comma before a closing }/] (see _remove_trailing_commas).

    The last two are only ever applied as repairs on top of a failed parse,
    never on the happy path, since rewriting content always carries some risk
    of changing meaning (however small) — the straightforward parse is always
    tried first, per candidate, before any repair.
    """
    content = _THINK_BLOCK_RE.sub("", content).strip()

    start, end = content.find("{"), content.rfind("}")
    candidates = [content]
    if start != -1 and end != -1 and end > start:
        candidates.append(content[start : end + 1])

    repairs = (
        lambda text: text,
        _fix_invalid_escapes,
        _remove_trailing_commas,
        lambda text: _remove_trailing_commas(_fix_invalid_escapes(text)),
    )

    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        for repair in repairs:
            try:
                return json.loads(repair(candidate), strict=False)
            except json.JSONDecodeError as exc:
                last_error = exc
    raise last_error
