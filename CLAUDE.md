# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpecForge AI: a LangGraph-orchestrated pipeline that turns raw requirements
(text/PDF/CSV/screenshots) into a polished spec, a QA test strategy matrix, and
export-ready test artifacts (TestRail Markdown, qTest CSV, Playwright TS), running
entirely against local Ollama models. Backend is FastAPI + LangGraph
(`backend/`); frontend is a React/Vite/Tailwind wizard (`frontend/`).

## Commands

**Backend** (from `backend/`):
```
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux — use .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
`pip`/`python` alone often don't exist on macOS outside an activated venv —
use `python3`, then activate the venv before calling plain `pip`.
Run tests (needs dev deps, not in the base `requirements.txt`):
```
pip install -r requirements-dev.txt
pytest                                        # single test: pytest tests/test_graph.py::test_straight_through_when_spec_and_matrix_are_clean
```
`pytest.ini` sets `asyncio_mode = auto` so async graph tests need no `@pytest.mark.asyncio`.
No lint/format tooling is configured yet.

**Frontend** (from `frontend/`):
```
npm install
npm run dev        # Vite dev server on :5173
npm run build
```
No test runner is configured for the frontend yet.

There is no Docker setup — the project runs natively (backend via `uvicorn`,
frontend via Vite's dev server), deliberately, since there's no need to host
or containerize this for the hackathon.

**Required local dependency**: Ollama running natively with at least
`qwen2.5vl:7b` (vision), `deepseek-r1:7b` (reasoning), `qwen2.5:7b`
(formatting) pulled — these are just the `app/config.py` defaults, not a hard
requirement; any pulled model can be selected per-session (see Model
selection below). The Ollama base URL and defaults are configurable via env
vars (see `.env.example`).

## Architecture

### Pipeline

```
[Raw Inputs] --> qwen2.5vl:7b       Visual element/layout extraction (images only)
             --> deepseek-r1:14b    Agent 1: BA Requirements Refiner (+ clarification loop)
             --> deepseek-r1:14b    Agent 2: QA Test Matrix Builder (+ gap clarifier loop)
             --> qwen2.5-coder:14b  Agent 3: Formatter Router (TestRail / qTest / Playwright)
```

This is implemented as a single LangGraph `StateGraph` in
`backend/app/graph/build.py`, over the shared state schema in
`backend/app/graph/state.py`. Node logic lives in `backend/app/graph/nodes.py`;
all three agents call Ollama through the thin wrapper in
`backend/app/graph/llm.py` (`ollama_chat`, which POSTs to `/api/chat`, supports
image attachments for the vision model, and for `expect_json=True` calls does
best-effort repair plus one corrective retry before giving up. The repair in
`_parse_json_with_repair` specifically targets two DeepSeek-R1 quirks hit live
(worse on the smaller `deepseek-r1:7b` than `:14b`, and worse the longer/more
structured the requested string value is): it strips any `<think>...</think>`
reasoning block before brace-extracting (R1 emits these even under
`format: json`, and stray braces inside the reasoning text would otherwise
confuse naive brace-slicing), and parses with `json.loads(..., strict=False)`
so a literal unescaped newline inside a string value (e.g. a multi-paragraph
markdown spec) doesn't raise `Invalid control character` instead of just
being treated as part of the string. See `test_llm_parsing.py` for the exact
repro of both.

The httpx timeout for every Ollama call is `settings.ollama_timeout_seconds`
(600s default, env `OLLAMA_TIMEOUT_SECONDS`) — a live run on modest hardware
hit the old hardcoded 180s ceiling on a 14B model with a large prompt. If this
resurfaces, raise the setting rather than re-hardcoding a number; hardware
speed varies a lot across contributors' machines. `main.py` registers
exception handlers for `httpx.TimeoutException` (504) and `httpx.ConnectError`
(502) so these surface as a readable `detail` message to the frontend instead
of a bare "Internal Server Error".

Routing/loop correctness (`route_ambiguity`, `route_gaps`, the interrupt/resume
cycle, legacy-test-case forwarding) is covered by
`backend/tests/test_graph.py`, which monkeypatches `nodes.ollama_chat` — no live
Ollama needed to verify the state machine itself. What isn't covered by those
tests: real model output quality (prompt wording, actual ambiguity judgment,
actual delta analysis) — that still needs a live model pass.

Graph flow: `ingest_visual -> ba_refiner -> (conditional) -> qa_matrix_builder ->
(conditional) -> checklist_signoff -> formatter`. The two conditional edges
(`route_ambiguity`, `route_gaps`) loop back to `ba_clarification` /
`gap_clarification` respectively when the corresponding LLM call reports
unresolved ambiguity/gaps, re-entering `ba_refiner`/`qa_matrix_builder` once
answered.

Both loops are capped at `MAX_CLARIFICATION_ROUNDS` (3, in `nodes.py`) — a live
run hit round 3 with DeepSeek-R1 still finding new ambiguity each pass, with no
way for the user to force progress. Once `len(qa_history)` /
`len(gap_qa_history)` reaches the cap, `ba_refiner_node`/`qa_matrix_builder_node`
switch to a "force resolve" system prompt (`BA_REFINER_FORCE_RESOLVE_SYSTEM` /
`QA_MATRIX_FORCE_RESOLVE_SYSTEM`) instructing the model to proceed with
reasonable, explicitly-labeled assumptions instead of asking again, and the
node treats the result as resolved regardless of what the model reports back
(`result.get("ambiguous")`/`result.get("gaps_found")` is ignored once
`force_resolve` is true). If the model still returns an empty `polished_spec`
on the forced call, `_fallback_spec()` synthesizes one from the raw
requirements + gathered Q&A rather than leaving `polished_spec` empty.

### Human-in-the-loop via LangGraph interrupts

There are three pause points, each implemented with `langgraph.types.interrupt()`
inside a dedicated node: `ba_clarification_node`, `gap_clarification_node`, and
`checklist_signoff_node`. These three are `async def` (even though none of them
`await` anything), which was a first guess at fixing a real crash — it turned
out not to be the actual cause, but there's no harm in leaving them async.

**The real cause, and why `requirements.txt` pins exact versions**: with a
loose `langgraph>=0.2.60,<0.3` range, `pip install` resolved a
`langgraph`/`langchain-core`/`langgraph-checkpoint` combination on one
teammate's machine where `interrupt()`'s internal `get_config()` call raised
`RuntimeError: Called get_config outside of a runnable context` — even when
the node was awaited directly via the correct async path (`await
self.afunc(...)`, no thread executor involved). The identical code worked
under `langgraph==0.2.76` / `langchain-core==0.3.86` /
`langgraph-checkpoint==2.1.2`. `requirements.txt` now pins these four
LangGraph-ecosystem packages exactly instead of as a range, since this is a
cross-package compatibility issue, not something a single-package version
bump can safely paper over. If this resurfaces, the fix is to find another
mutually-compatible set of exact versions and pin those — not to widen the
range back out.

The graph is compiled with a `MemorySaver` checkpointer (process-local, not
persisted across restarts), keyed by `thread_id == session_id`. Resuming a
paused graph is done by invoking with `Command(resume=<value>)` against the
same `thread_id` — see `backend/app/routers/session.py`. This means **a
session only survives as long as the backend process stays up**; there's no
durable session store yet.

Because FastAPI is stateless across requests, `session.py`'s `_to_response()`
helper reconstructs "what is the frontend waiting for" purely by calling
`graph.aget_state()` and inspecting `snapshot.next` (the node LangGraph is about
to run) rather than tracking status separately — `snapshot.next` containing
`ba_clarification`/`gap_clarification`/`checklist_signoff` is what drives the
`awaiting_input` field the frontend switches on.

### Ingestion

`POST /api/sessions/` accepts multipart form data with two separate upload
channels — `files[]` for requirements-side attachments and `legacy_files[]` for
an uploaded legacy test suite — plus `text` and `legacy_test_cases` as plain
form fields for pasted content. Files in `files[]` are appended to
`requirements_draft`; files in `legacy_files[]` are appended to
`legacy_test_cases`. These are intentionally separate fields, not a single
upload routed by content — an early version conflated them, which meant an
uploaded legacy CSV silently ended up in the requirements text instead (see
`test_session_router.py` for the regression tests pinning this).

File routing by MIME/extension happens in the router itself, not in the graph:
PDFs and CSVs are extracted to text synchronously via
`app/services/file_parser.py` before the graph ever runs; images (only valid
in `files[]`) are saved to `storage/<session_id>/` and passed as `image_paths`
into the initial state, to be read and base64-encoded by the vision node at
graph-execution time.

### Model selection

Each session can override `vision_model`/`reasoning_model`/`formatter_model`
independently instead of being stuck with `app/config.py`'s defaults. The
override is a plain value flowing through the system, not a separate
mechanism: `start_session` accepts three optional form fields, writes
whatever was chosen (or the setting default, if blank) directly into
`SpecForgeState`, and each node resolves its own model with
`state.get("<role>_model") or settings.<role>_model` rather than reading
`settings` directly — see the `model = ...` line at the top of
`ingest_visual_node`/`ba_refiner_node`/`qa_matrix_builder_node`/
`formatter_node`. `_to_response` echoes the resolved models back on every
response so the frontend (and logs, via `llm.py`'s existing per-call model
logging) always reflect what's actually running for that session, not just
what's configured.

`GET /api/models` (`app/routers/models.py`) proxies Ollama's own `/api/tags`
to list whatever's actually pulled locally, plus the configured defaults —
this is what populates the Upload step's three model dropdowns
(`UploadStep.jsx`). If Ollama isn't reachable when this loads, the frontend
catches the failure and simply doesn't render the selector, falling back
silently to the backend's configured defaults rather than blocking the form.

### Frontend state machine

`frontend/src/App.jsx` holds the entire session as one object returned verbatim
from the backend's `SessionStateResponse` and derives the active wizard step
from it (`stepKeyFor`) — there is no separate client-side state machine mirroring
the backend one. Every user action (`clarify-requirements`, `clarify-gaps`,
`checklist-signoff`) POSTs and replaces the whole session object with the
response, which is why the checklist editor keeps its own local `matrix` copy
(`ChecklistEditor.jsx`) until sign-off is submitted.

See README.md's Troubleshooting section for real environment issues hit
during setup (macOS system-Python contamination causing a LangGraph
`get_config` crash, port-mismatch CORS/404s, etc.) before assuming a new bug
report is something novel in the code.

### Known gaps (intentional, not yet built)

- No persistent checkpointer (sessions lost on backend restart).
- JSON repair is one corrective retry, not a full validation/repair loop —
  malformed output after the retry propagates as an unhandled exception
  (surfaces to the caller as a 500).
- Prompts in `nodes.py` have never been run against a live model; expect to
  iterate on wording once real DeepSeek-R1/Qwen output comes back.
- Playwright export is prompt-only — no generated file is executed/validated.
- No automated frontend tests. Backend tests cover the graph's routing/loop
  logic and the health check, but not the FastAPI routes themselves
  (`routers/session.py`) or the file parsers.
