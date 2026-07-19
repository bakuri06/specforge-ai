# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpecForge AI: a LangGraph-orchestrated pipeline that turns raw requirements
(text/PDF/CSV/screenshots) into a polished spec, a QA test strategy matrix, and
export-ready test artifacts (BDD/Gherkin, TestRail Markdown, qTest CSV,
Jira/Xray JSON, Azure DevOps CSV), running entirely against local Ollama
models. Backend is FastAPI + LangGraph (`backend/`); frontend is a
React/Vite/Tailwind wizard (`frontend/`), including the requirement-evaluation
step and hierarchical test-step editing (see "Known gaps" for what's still
Flow-A-only in the UI: no `workflow_mode`/`out_of_scope_details` selection
yet, so Flows B/C can only be driven via the API directly).

The graph supports three independent entry points (`workflow_mode`), not just
one linear pipeline:
- **Flow A ("full")**: raw requirements -> BA refiner -> QA matrix builder ->
  formatter. The only flow with a requirement-evaluation gate and BA
  clarification loop.
- **Flow B ("qa_direct")**: caller supplies already-refined requirements
  directly into `polished_spec`, bypassing the BA refiner entirely, and enters
  at the QA matrix builder.
- **Flow C ("format_only")**: caller supplies an existing/legacy test
  document + a target format upfront; enters directly at the formatter, which
  translates/reformats that document with no BA/QA agents and no interrupts
  at all — a single-pass, non-interactive request.

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
Flow A (full):
[Raw Inputs] --> qwen2.5vl:7b       Visual element/layout extraction (images only)
             --> deepseek-r1:7b     Requirement Evaluator (readiness score + interrupt)
             --> deepseek-r1:7b     Agent 1: BA Requirements Refiner (+ clarification loop)
             --> deepseek-r1:7b     Agent 2: QA Test Matrix Builder (+ gap clarifier loop)
             --> qwen2.5:7b         Agent 3: Formatter Router (BDD/TestRail/qTest/Jira-Xray/Azure DevOps)

Flow B (qa_direct):  [Pre-refined text] --> polished_spec --> Agent 2 --> Agent 3
Flow C (format_only): [Legacy test cases] --> Agent 3 (translate mode, no BA/QA, no interrupts)
```

This is implemented as a single LangGraph `StateGraph` in
`backend/app/graph/build.py`, over the shared state schema in
`backend/app/graph/state.py`. Node logic lives in `backend/app/graph/nodes.py`;
all four agents (evaluator, BA refiner, QA matrix builder, formatter) call
Ollama through the thin wrapper in
`backend/app/graph/llm.py` (`ollama_chat`, which POSTs to `/api/chat`, supports
image attachments for the vision model, and for `expect_json=True` calls does
best-effort repair plus one corrective retry before giving up. The repair in
`_parse_json_with_repair` specifically targets four DeepSeek-R1 quirks hit
live (worse on the smaller `deepseek-r1:7b` than `:14b`, and worse the
longer/more structured the requested string value is): it strips any
`<think>...</think>` reasoning block before brace-extracting (R1 emits these
even under `format: json`, and stray braces inside the reasoning text would
otherwise confuse naive brace-slicing); it parses with
`json.loads(..., strict=False)` so a literal unescaped newline inside a
string value (e.g. a multi-paragraph markdown spec) doesn't raise
`Invalid control character` instead of just being treated as part of the
string; it falls back to `_fix_invalid_escapes` (escaping any backslash that
isn't already a valid JSON escape char) for a literal backslash inside a
string value — e.g. a regex-style pattern like `\d{6}` written without
doubling it — which raises `Invalid \escape` regardless of `strict=False`
(that flag only relaxes control characters, not malformed escape sequences);
and it falls back to `_remove_trailing_commas` for a comma left before a
closing `}`/`]` (valid in a JS object/array literal, invalid in strict JSON,
and surfaces as a confusing `Expecting value` error that never mentions a
comma at all). The escape and trailing-comma repairs are genuinely
last-resort — tried only after a plain parse of every candidate fails — since
rewriting content always carries some risk of changing meaning; the escape
one specifically hit on *both* the original call and the corrective retry in
a live run, confirming re-prompting alone isn't reliable for this class of
error and the parser itself has to repair it. `_parse_json_with_repair` tries
each repair (identity, escape-fix, trailing-comma-fix, both together) against
both the raw and brace-sliced content before giving up. See
`test_llm_parsing.py` for the exact repro of each.

Getting valid JSON back isn't the same as getting the right *shape* back —
`_parse_json_with_repair` only guarantees parseable JSON, not that
`polished_spec` is a string or `category`/`status` are one of the allowed
enum values. A live run on `deepseek-r1:7b` returned `polished_spec` as a
JSON object with each section as a key instead of one markdown string, and
separately returned every `category` as the literal string
`"sunny_day|rainy_day|boundary|edge_case"` — copying the prompt's own
shape-example notation verbatim instead of picking one value — which crashed
`SessionStateResponse(...)`'s pydantic validation in `_to_response`. Both
`ba_refiner_node` and `qa_matrix_builder_node` now run their result through
`_coerce_polished_spec`/`_coerce_test_matrix` (with `_coerce_enum` splitting
on `|`/`,`/`/` to recover a valid token, falling back to a safe default
otherwise) before returning, at every place the model's result feeds into
state — not just the `force_resolve` path, since either failure mode can
happen on a normal resolved pass too. See
`test_malformed_model_output_does_not_crash_response_construction` in
`test_session_router.py` for the full end-to-end repro (it runs the real
graph, not the `_FakeGraph` used by the rest of that file, since the crash
was specifically in response construction downstream of the graph).

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

Graph flow (see `backend/app/graph/build.py`):

```
START --route_entry--> ingest_visual        [workflow_mode in (full, qa_direct)]
                     -> formatter            [workflow_mode == format_only, translate mode]

ingest_visual --route_after_ingest--> requirement_evaluator   [full]
                                    -> qa_matrix_builder        [qa_direct]

requirement_evaluator -> evaluation_review (interrupt)
evaluation_review --route_after_evaluation--> ba_refiner   [proceed]
                                            -> END          [abort]

ba_refiner --route_ambiguity--> qa_matrix_builder | ba_clarification
ba_clarification -> ba_refiner   (loop back)

qa_matrix_builder --route_gaps--> checklist_signoff | gap_clarification
gap_clarification -> qa_matrix_builder   (loop back)

checklist_signoff -> formatter -> END
```

Both entry routers (`route_entry`, `route_after_ingest`) default defensively
to the "full"/non-`format_only` path for any missing or unrecognized
`workflow_mode`, matching the codebase's `_coerce_enum` convention of never
trusting a decision-point input. Confirmed via direct inspection of the
installed `langgraph==0.2.76` source that `add_conditional_edges(START, ...)`
is genuinely supported as a conditional entry point (not just something that
happens to pass `validate()`) — traced through `StateGraph.compile()` and
`Pregel`'s resume path; `Command(resume=...)` never re-touches the START
channel, so a conditional entry point cannot be accidentally re-triggered
mid-session on resume.

Flow B (`qa_direct`) still routes through `ingest_visual` before diverging to
`qa_matrix_builder`, even though it skips the BA refiner entirely — an
earlier draft of this routing sent `qa_direct` straight from START, which
would have silently dropped any uploaded screenshot's `visual_context` since
nothing else populates it. `qa_matrix_builder_node` doesn't currently read
`visual_context` either way, but the ingestion step at least isn't silently
lost.

Both loops are capped at `MAX_CLARIFICATION_ROUNDS` (currently 1, in
`nodes.py` — originally 3, lowered since the smaller 7B models rarely
resolve ambiguity even by round 3 and the wait per round is long) — a live
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

There are four pause points, each implemented with `langgraph.types.interrupt()`
inside a dedicated node: `evaluation_review_node`, `ba_clarification_node`,
`gap_clarification_node`, and `checklist_signoff_node`. These are all `async def`
(even though none of them `await` anything), which was a first guess at fixing a
real crash — it turned out not to be the actual cause, but there's no harm in
leaving them async.

**Every interrupt-calling node must do NO real work before its `interrupt()`
call — put the work in a separate preceding node instead.** LangGraph replays
a node's code from the top on every resume, up to (and re-executing) whatever
runs before the `interrupt()` call in that same node. `requirement_evaluator_node`
originally called the evaluation LLM *and then* called `interrupt()` in one
node; a test caught that every resume silently re-ran the (expensive) LLM
call. Fixed by splitting it into `requirement_evaluator_node` (does the LLM
call, writes `readiness_score`/`evaluation_feedback`/`recommended_clarification_rounds`
to state, no interrupt) followed by a plain edge into `evaluation_review_node`
(interrupt-only, reads those values back out of state) — mirroring the
existing `ba_refiner_node` -> `ba_clarification_node` split. Any new
interrupt point should follow this same two-node shape from the start.

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
`evaluation_review`/`ba_clarification`/`gap_clarification`/`checklist_signoff`
(via `_AWAITING_BY_NEXT_NODE`, keyed by node name) is what drives the
`awaiting_input` field the frontend switches on. Any new pause node must be
added to that dict or the frontend has no way to learn it needs to render a
step for it.

An aborted Flow A session (user chose "abort" at the evaluation gate) also
ends with `snapshot.next == ()`, structurally identical to a normally
*completed* session's empty `next` — so `_to_response` does **not** rely on
`next` emptiness to detect this. `evaluation_review_node` sets an explicit
`workflow_aborted: bool` in state instead, echoed straight through in
`SessionStateResponse.workflow_aborted`, which is the only reliable signal.

### Requirement Evaluation Gate (Flow A only)

`requirement_evaluator_node` scores raw-requirements completeness (0-100),
lists qualitative gaps, and recommends a clarification-round count — explicitly
instructed to ignore anything the user listed in `out_of_scope_details` (no
score penalty, no questions about it). `evaluation_review_node` then pauses so
the user can either override the round count (including `0`, which skips the
BA clarification loop entirely on the very next `ba_refiner_node` call) or
abort. `ba_refiner_node`'s force-resolve check reads
`state.get("max_clarification_rounds", MAX_CLARIFICATION_ROUNDS)` — a
per-session value set by `evaluation_review_node` — instead of the module
constant directly; Flows B/C never populate that key (they skip the evaluator
entirely) and correctly fall back to the module constant. The QA matrix
builder's gap-loop cap is untouched by any of this — it's out of scope for the
evaluator, which is specifically "immediately after ingestion in System 1."

### Hierarchical test steps

`test_matrix` items no longer have a flat `description: str` — each item has
`steps: list[{step_number, action, expected_result}]`. `_coerce_test_matrix`
is backward-compatible with a model that still returns the old flat
`description` field (falls back to a single synthesized step from that text)
so an under-instructed smaller model degrades gracefully instead of losing
the scenario entirely.

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
graph-execution time. `start_session` also accepts `workflow_mode`,
`out_of_scope_details`, and (Flow C only) an upfront `output_format` field —
the last one means Flow C's very first response already has `output_format`
populated, unlike Flows A/B where it stays `None` until checklist sign-off.

### Formatter Router (5 formats)

`FORMATTER_FORMAT_RULES` holds just the shape/column rules per format
(`bdd`, `testrail`, `qtest`, `jira_xray`, `azure_devops`); a `COMPILE_INSTRUCTION`
or `TRANSLATE_INSTRUCTION` wrapper is combined with those rules at call time
depending on `workflow_mode == "format_only"`, so the rules text isn't
duplicated between "compile this structured test_matrix" (Flows A/B) and
"translate/reformat this existing document, don't invent new cases" (Flow C).

Only the two formats with a hard machine-parseable contract get post-generation
validation: `jira_xray` is re-parsed with the same `<think>`-stripping /
`strict=False` tolerance as other structured LLM calls
(`app.graph.llm._parse_json_with_repair`, reused directly) and re-serialized
via `json.dumps` for a canonical string; `qtest`/`azure_devops` are checked
with `csv.reader` for a consistent column count across every row. Both get one
corrective retry on failure (`_validate_and_repair_json`/`_validate_and_repair_csv`),
mirroring `ollama_chat`'s own JSON retry pattern. `bdd`/`testrail` get no extra
validation — same precedent as the original 3-format router, lower structural
risk since there's no hard parse contract to violate. `output_format`'s
5-value Literal is duplicated across `state.py`, `schemas.py` (x2),
`nodes.py`'s rules/sample-file dicts, and `session.py`'s download extension
map — pre-existing duplication (was 3 formats across the same set of places),
not something this change introduced or fixed.

### Few-shot samples (`backend/app/samples/`, `backend/app/services/samples.py`)

Gold-standard example files (`sample_raw_spec.md`/`expected_refined_spec.md`
pair, `sample_test_matrix.json`, and one `expected_<format>.*` file per
formatter format) are loaded via `samples.load_sample()` (`lru_cache`'d file
read) and spliced into the relevant prompt via `samples.few_shot_block()` at
**module import time** — `BA_REFINER_SYSTEM`/`QA_MATRIX_SYSTEM`/etc. are
computed once when `nodes.py` is imported, not per-request. This means a
missing/corrupt sample file fails the entire app at startup, not per-request —
intentional fail-fast for what are meant to be permanent, version-controlled
files, not something dynamic. Do not call `few_shot_block()` inside an f-string
combined with a later `.format()` call on the same constant — the literal JSON
braces in these prompts collide with `str.format()`'s placeholder syntax
(hit this exact bug on `QA_MATRIX_SYSTEM`; fixed by using plain string
concatenation instead of `.format()` to splice the example in).

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

Since the multi-entry refactor, `stepKeyFor` also handles two states that
have no `snapshot.next`-derived equivalent: `session.workflow_aborted` (routes
to a terminal "aborted" step — deliberately checked as an explicit field, not
inferred from an empty `next`, for the same reason `_to_response` does it
that way server-side) and `awaiting_input === "requirement_evaluation"`
(routes to `EvaluationStep.jsx`, which posts to the new
`POST /{session_id}/evaluation-decision` endpoint with `{action, max_clarification_rounds}`).
`ChecklistEditor.jsx` was updated in lockstep with the backend's `steps`
shape (per-scenario add/edit/remove step rows instead of one flat
`description` textarea) and the 5-format dropdown.

**Still Flow-A-only in the UI** — Flows B/C exist and are fully functional
via the API (see `test_session_router.py`'s `qa_direct`/`format_only` tests),
but nothing in `UploadStep.jsx` lets a user pick `workflow_mode`, set
`out_of_scope_details`, or (Flow C) choose the target format upfront. A
session started from the current UI is always `workflow_mode: "full"`.

See README.md's Troubleshooting section for real environment issues hit
during setup (macOS system-Python contamination causing a LangGraph
`get_config` crash, port-mismatch CORS/404s, etc.) before assuming a new bug
report is something novel in the code.

### Known gaps (intentional, not yet built)

- No persistent checkpointer (sessions lost on backend restart).
- JSON repair is one corrective retry, not a full validation/repair loop —
  malformed output after the retry propagates as an unhandled exception
  (surfaces to the caller as a 500). The formatter's `jira_xray`/`qtest`/
  `azure_devops` validate-and-repair helpers follow the same one-retry
  ceiling, not a full loop either.
- Prompts in `nodes.py` have never been run against a live model; expect to
  iterate on wording once real DeepSeek-R1/Qwen output comes back — this is
  now also true of the requirement-evaluation prompt and all 5 formatter
  prompts, none of which have been validated against a live model yet either.
- No automated frontend tests. Backend tests cover the graph's routing/loop
  logic and the health check, but not the FastAPI routes themselves
  (`routers/session.py`) or the file parsers.
- No UI for Flows B/C's entry parameters (`workflow_mode`, `out_of_scope_details`,
  Flow C's upfront format picker) — see "Frontend state machine" above.
