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

**Required local dependency**: Ollama running natively with three models pulled —
`qwen2.5vl:7b` (vision), `deepseek-r1:14b` (reasoning), `qwen2.5-coder:14b`
(formatting). Model names and the Ollama base URL are configurable via env vars
(see `backend/app/config.py` / `.env.example`).

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
best-effort brace-extraction parsing plus one corrective retry — re-prompting
with the malformed output included — before giving up, since local models under
`format: json` still occasionally wrap output in prose/fences).

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

### Human-in-the-loop via LangGraph interrupts

There are three pause points, each implemented with `langgraph.types.interrupt()`
inside a dedicated node: `ba_clarification_node`, `gap_clarification_node`, and
`checklist_signoff_node`. These three are deliberately `async def`, even though
none of them `await` anything — a plain `def` node gets dispatched through
`langchain_core`'s thread-pool executor to avoid blocking the event loop, and on
at least one observed `langchain-core`/`langgraph` version combination that
thread hop drops the contextvar `interrupt()` needs, raising `RuntimeError:
Called get_config outside of a runnable context`. Keep any future
interrupt-calling node `async def` to avoid reintroducing this.

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

### Frontend state machine

`frontend/src/App.jsx` holds the entire session as one object returned verbatim
from the backend's `SessionStateResponse` and derives the active wizard step
from it (`stepKeyFor`) — there is no separate client-side state machine mirroring
the backend one. Every user action (`clarify-requirements`, `clarify-gaps`,
`checklist-signoff`) POSTs and replaces the whole session object with the
response, which is why the checklist editor keeps its own local `matrix` copy
(`ChecklistEditor.jsx`) until sign-off is submitted.

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
