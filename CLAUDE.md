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
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Run tests: `pytest` (single test: `pytest tests/test_health.py::test_health`).
No lint/format tooling is configured yet.

**Frontend** (from `frontend/`):
```
npm install
npm run dev        # Vite dev server on :5173
npm run build
```
No test runner is configured for the frontend yet.

**Full stack via Docker Compose** (from repo root): `docker compose up --build`
— backend on `:8000`, frontend on `:5173`. The backend container reaches the
*host's* native Ollama daemon via `http://host.docker.internal:11434` (set in
`docker-compose.yml`), since Ollama itself is not containerized.

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
image attachments for the vision model, and does best-effort JSON repair for
`expect_json=True` calls since local models under `format: json` still
occasionally wrap output in prose/fences).

Graph flow: `ingest_visual -> ba_refiner -> (conditional) -> qa_matrix_builder ->
(conditional) -> checklist_signoff -> formatter`. The two conditional edges
(`route_ambiguity`, `route_gaps`) loop back to `ba_clarification` /
`gap_clarification` respectively when the corresponding LLM call reports
unresolved ambiguity/gaps, re-entering `ba_refiner`/`qa_matrix_builder` once
answered.

### Human-in-the-loop via LangGraph interrupts

There are three pause points, each implemented with `langgraph.types.interrupt()`
inside a dedicated node: `ba_clarification_node`, `gap_clarification_node`, and
`checklist_signoff_node`. The graph is compiled with a `MemorySaver` checkpointer
(process-local, not persisted across restarts), keyed by `thread_id ==
session_id`. Resuming a paused graph is done by invoking with
`Command(resume=<value>)` against the same `thread_id` — see
`backend/app/routers/session.py`. This means **a session only survives as long
as the backend process stays up**; there's no durable session store yet.

Because FastAPI is stateless across requests, `session.py`'s `_to_response()`
helper reconstructs "what is the frontend waiting for" purely by calling
`graph.aget_state()` and inspecting `snapshot.next` (the node LangGraph is about
to run) rather than tracking status separately — `snapshot.next` containing
`ba_clarification`/`gap_clarification`/`checklist_signoff` is what drives the
`awaiting_input` field the frontend switches on.

### Ingestion

`POST /api/sessions/` accepts multipart form data (`text`, `legacy_test_cases`,
`files[]`). File routing by MIME/extension happens in the router itself, not in
the graph: PDFs and CSVs are extracted to text synchronously via
`app/services/file_parser.py` before the graph ever runs; images are saved to
`storage/<session_id>/` and passed as `image_paths` into the initial state, to be
read and base64-encoded by the vision node at graph-execution time.

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
- No structural-repair/retry loop around malformed LLM JSON beyond the
  best-effort brace-extraction in `llm.py`.
- Playwright export is prompt-only — no generated file is executed/validated.
- No automated frontend tests; backend has only a health-check test.
