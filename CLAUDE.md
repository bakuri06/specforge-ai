# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpecForge AI: a LangGraph-orchestrated pipeline that turns raw requirements
(text/PDF/CSV/screenshots) into a polished spec, a QA test strategy matrix, and
export-ready test artifacts (BDD/Gherkin, TestRail Markdown, qTest CSV,
Jira/Xray JSON, Azure DevOps CSV), running entirely against local Ollama
models. Backend is FastAPI + LangGraph (`backend/`); frontend is a
React/Vite/Tailwind wizard (`frontend/`), including the requirement-evaluation
step, hierarchical test-step editing, and a `workflow_mode`/`out_of_scope_details`
picker on the Upload step (see "Known gaps" for the one flow still API-only).

The graph supports four independent entry points (`workflow_mode`), not just
one linear pipeline:
- **Flow A ("full")**: raw requirements -> BA refiner -> QA matrix builder ->
  formatter. The only flows with a requirement-evaluation gate and BA
  clarification loop are this one and "refine_only" below (they share the
  same path up through the BA refiner).
- **"refine_only"**: identical path to Flow A through ingestion, the
  evaluation gate, and the BA refiner (including its clarification loop) —
  but stops there instead of continuing into the QA matrix builder. The
  polished spec itself is the deliverable; there's no test matrix, checklist
  sign-off, or formatter step for this flow at all.
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

refine_only: same path as Flow A through the BA refiner, then stops — no
             Agent 2, no Agent 3, no checklist sign-off.
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

ba_refiner --route_ambiguity--> qa_matrix_builder | ba_clarification | END
                                [resolved, full]    [clarify]           [resolved, refine_only]
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

### Vision OCR fail-safe (`ingest_visual_node`, `app/services/vision_ocr.py`)

`ingest_visual_node` no longer calls `ollama_chat` directly for each
screenshot — it calls `vision_ocr.extract_text_from_screenshot(path,
model=model, prompt=VISION_PROMPT)`, a dual-layer fail-safe: try the vision
LLM first, and fall back to deterministic OCR (`pytesseract`) if it throws or
its output looks like garbage. Added because `qwen2.5vl:7b` occasionally
returns degraded token-loop/gibberish output on scaled-down or low-contrast
screenshots, with no way to detect or recover from that today.

Both paths run against the same preprocessed image (`preprocess_image`):
resize the long side into `[768, 1344]px` preserving aspect ratio, add a 20px
solid white border (prevents text clipping at the edges), then grayscale +
Gaussian blur + Otsu's binarization. Order matters — resize happens before
padding so the size target reflects actual content, not border pixels. The
binarized array is used directly for the `pytesseract` fallback and written
to a throwaway temp PNG for the vision-LLM call (`ollama_chat`'s `images`
param only accepts file paths, not in-memory arrays); the temp file is
removed in a `finally` block regardless of outcome.

`_looks_like_gibberish` flags the vision output as a failure on any of: zero
alphabetic characters, a run of 10+ identical characters (Qwen-VL's
degenerate token-loop failure mode), a cluster of block/shade/geometric
Unicode characters or the `U+FFFD` replacement character (garbled-decode
artifacts), no word-like token at all, or an alpha-character ratio below 15%.
Any of these — or the vision call raising an exception at all — triggers the
`pytesseract.image_to_string()` fallback on the same preprocessed array.

`pytesseract` is only a Python wrapper around the Tesseract binary, which
must be installed natively (see README Prerequisites) — the same kind of
system-level dependency this project already has with Ollama. `settings.tesseract_cmd`
(env `TESSERACT_CMD`) is only needed when Tesseract isn't on `PATH` (common on
Windows), same override pattern as every other `app/config.py` setting.

Tests (`backend/tests/test_vision_ocr.py`) always monkeypatch both
`vision_ocr.ollama_chat` and `vision_ocr.pytesseract.image_to_string` — no
live Ollama or a real Tesseract install needed to verify the fail-safe logic
itself, matching this project's existing "mock the LLM call, not the
mechanism" testing convention. One non-obvious wrinkle this caused: existing
graph tests that monkeypatch `nodes_module.ollama_chat` to fake image
analysis (`test_flow_b_qa_direct_still_processes_uploaded_images`) had to
also patch `vision_ocr.ollama_chat` — `nodes.py` and `vision_ocr.py` each do
their own `from app.graph.llm import ollama_chat`, so these are two separate
name bindings in two separate module namespaces (both call the bare name
`ollama_chat(...)`, resolved via their *own* module's globals at call time),
even though they originally point at the same function object. Patching one
does not affect the other. That test's fake image path also had to become a
real, readable image file (via `cv2.imwrite` against `tmp_path`) rather than
a nonexistent placeholder string, since `preprocess_image` now actually opens
the file with `cv2.imread` before the mocked `ollama_chat` is ever reached.

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

### Requirement Evaluation Gate (Flow A and refine_only)

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

`evaluation_feedback` is a categorized dict, not a flat list of strings:
`state.py`'s `QualityGateFeedback`/`schemas.py`'s mirrored Pydantic model both
fix the same four keys — `data_and_boundaries` (input validation/regex/field
limits), `integration_and_async_behavior` (background jobs, push protocols,
messaging hooks, payload schemas), `network_and_resiliency` (timeouts,
backoff/retries, network fault states), `state_and_lifecycle` (status
mutations, retention, expiration) — each a `list[str]` of gaps found in that
specific gate. `readiness_score` stays one holistic 0-100 LLM judgment across
all four, not a per-category sub-score averaged deterministically; there was
no way to compute a *reliable* aggregate from four independent per-category
numbers that wouldn't just reintroduce the same "trust the model's own
number" problem this evaluator already has. `_coerce_quality_gate_feedback`
(`nodes.py`) replaces the old `_coerce_feedback_list`: if a smaller model
ignores the categorized shape and returns the old flat list (or a bare
string) instead, there's no reliable way to guess which category its items
belong to, so the safe fallback is all-four-categories-empty — same
never-crash-on-wrong-shape precedent as `_coerce_test_matrix`'s non-list
fallback. `EvaluationStep.jsx` renders one labeled group per category instead
of a single flat bullet list.

### "refine_only": stopping after the polished spec

`route_ambiguity` (`nodes.py`) is the only place `refine_only` diverges from
Flow A: once `ambiguity_resolved` is true, it returns `"stop"` instead of
`"resolved"` when `state.get("workflow_mode") == "refine_only"`, which
`build.py` maps to `END` in `ba_refiner`'s conditional-edge `path_map`
(`{"resolved": "qa_matrix_builder", "clarify": "ba_clarification", "stop": END}`).
Everything upstream of that check — ingestion, the evaluation gate, the BA
clarification loop itself — is unchanged Flow A code; `refine_only` isn't a
shortcut around clarification, it just declines to continue past the refiner
once resolved. `qa_matrix_builder` is never invoked for this flow (see
`test_refine_only_stops_after_ba_refiner_resolves` in `test_graph.py`, which
asserts on a call counter rather than just on `snapshot.next`). On the
frontend, `App.jsx`'s `stepKeyFor` routes to a terminal `RefineOnlyDoneStep.jsx`
when `workflow_mode === "refine_only"` and `polished_spec` is set with no
`awaiting_input` pending; that component reuses the existing
`PolishedSpecPanel.jsx` display (already rendered unconditionally whenever
`polished_spec` is present) and adds a client-side Blob download of the spec
as `.md` — no backend endpoint needed since the spec is already in the
session response.

Adding `refine_only` to `state.py`'s/`nodes.py`'s/`build.py`'s
`workflow_mode` Literal wasn't sufficient by itself — a live smoke test (fake
Ollama server + real uvicorn, not just the mocked-`ollama_chat` graph tests)
caught `SessionStateResponse.workflow_mode` in `schemas.py` still pinned to
the old 3-value Literal, which made pydantic reject the very first response
for any `refine_only` session with a `literal_error` inside `_to_response`.
`test_graph.py` never catches this class of bug since it drives the graph
directly and never constructs a `SessionStateResponse`; the regression test
is `test_refine_only_flow_reaches_polished_spec_through_the_real_router` in
`test_session_router.py`, which goes through the real FastAPI app (like
`test_malformed_model_output_does_not_crash_response_construction` does) for
exactly this reason. Any future `workflow_mode` value must be added in all
three places (`state.py`, `session.py`'s `VALID_WORKFLOW_MODES`, and
`schemas.py`'s `SessionStateResponse.workflow_mode`) — there's no single
source of truth for this Literal today.

### Prompts module (`app/graph/prompts.py`)

Every prompt constant (`VISION_PROMPT`, `REQUIREMENT_EVALUATOR_SYSTEM`,
`BA_REFINER_SYSTEM`/`BA_REFINER_FORCE_RESOLVE_SYSTEM`,
`QA_MATRIX_SYSTEM`/`QA_MATRIX_FORCE_RESOLVE_SYSTEM`,
`FORMATTER_FORMAT_RULES`/`FORMAT_SAMPLE_FILES`/`COMPILE_INSTRUCTION`/
`TRANSLATE_INSTRUCTION`) lives here now, not inline in `nodes.py` — `nodes.py`
imports them and stays focused on graph/node logic. The
`few_shot_block(...)` splicing that happens at module-import time (see
"Few-shot samples" below) moved with them; `prompts.py` importing from
`app.services.samples` has no circular-import risk since `samples.py` never
imports from `app.graph`. This was a pure move — no prompt constant's *name*
changed, and every marker phrase existing tests key on to disambiguate mocked
`ollama_chat` calls (`"Core Calculation Framework"` for the BA refiner,
`"senior QA Engineer"` for the QA matrix builder, `"recommended_clarification_rounds"`
for the evaluator) is still present verbatim, so no test needed to change
because of the move itself — only where a prompt's *text* was deliberately
rewritten (see below) did anything else need touching.

`QA_MATRIX_SYSTEM`/`QA_MATRIX_FORCE_RESOLVE_SYSTEM` both got a "CONTEXT
LOCKING" instruction block (`QA_MATRIX_CONTEXT_LOCKING`, shared between the
two so a future edit can't update one and silently forget the other — the
same mistake this codebase already hit once with `BA_REFINER_SPEC_SECTIONS`)
forbidding generic template scenarios (login, money-transfer) unless the
polished blueprint actually describes that functionality, instructing the
model to trace the blueprint's actual named components instead. This is
prompt-only, not paired with a deterministic check — unlike the BDD guardrail
below, "did the model actually anchor to the input" isn't something a regex
can verify.

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
graph-execution time. `_save_upload` prefixes every saved filename with a
fresh `uuid4().hex` rather than writing directly to `upload.filename` — two
screenshots sharing a name (routine for clipboard-pasted images, e.g. both
literally `image.png`) used to silently overwrite each other on disk. This
mattered specifically for images and not PDFs/CSVs/legacy files: those are
read back via `_extract_text` synchronously, in the same loop iteration as
the save, so a same-named *later* upload in the batch can't clobber an
*earlier* one before it's been read; `image_paths` is only read much later at
graph-execution time in `ingest_visual_node`, by which point every upload in
the request has already been saved, so a collision meant both `image_paths`
entries pointed at whichever image happened to be written last — the vision
model would then analyze the same (wrong) screenshot twice, surfacing as
plausible-looking but incorrect per-image analysis rather than a crash. See
`test_same_named_image_uploads_do_not_overwrite_each_other_on_disk` in
`test_session_router.py`. `start_session` also accepts `workflow_mode`,
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

The two formats with a hard machine-parseable contract get post-generation
validation via a corrective LLM re-prompt: `jira_xray` is re-parsed with the
same `<think>`-stripping / `strict=False` tolerance as other structured LLM
calls (`app.graph.llm._parse_json_with_repair`, reused directly) and
re-serialized via `json.dumps` for a canonical string; `qtest`/`azure_devops`
are checked with `csv.reader` for a consistent column count across every row.
Both get one corrective retry on failure
(`_validate_and_repair_json`/`_validate_and_repair_csv`), mirroring
`ollama_chat`'s own JSON retry pattern.

`bdd` gets a different kind of post-generation guardrail: `_merge_multiple_bdd_features`
(`nodes.py`) deterministically collapses N `Feature:` lines into 1 by deleting
every `Feature:` line after the first, letting their `Scenario`/`Scenario Outline`
blocks fall through unchanged so they end up nested under the single
remaining Feature. A local model asked to compile several test cases at once
sometimes emits a separate `Feature:` per case instead of nesting them all
under one — invalid Gherkin (a `.feature` file has exactly one root Feature).
Unlike the JSON/CSV cases, this doesn't retry via the LLM at all: a
multi-Feature merge is always mechanically fixable by deleting lines, so
there's no reason to gamble on a re-prompt the way JSON/CSV validity
sometimes has to. `FORMATTER_FORMAT_RULES["bdd"]`'s prompt text was also
updated to explicitly demand exactly one `Feature:` declaration — the
deterministic merge is the backstop for when that instruction alone doesn't
hold, not a replacement for it. `testrail` still gets no extra validation —
lower structural risk since there's no hard parse contract to violate.

`output_format`'s 5-value Literal is duplicated across `state.py`,
`schemas.py` (x2), `prompts.py`'s rules/sample-file dicts, and `session.py`'s
download extension map — pre-existing duplication (was 3 formats across the
same set of places), not something this change introduced or fixed.

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

### Rewinding to a previous step

`POST /{session_id}/rewind` (body: `RewindRequest{target}`, `target` one of
`evaluation_review`/`ba_clarification`/`gap_clarification`/`checklist_signoff`
— the same four node names as `_AWAITING_BY_NEXT_NODE`'s keys) lets a session
go back to an earlier pause point and discard everything computed after it,
so the user can resubmit with different input — confirmed via clarifying
questions to cover all four pause points with discard-and-rerun-downstream
semantics, not just patch a field in place.

This relies on LangGraph checkpoint history, not just replaying an endpoint —
validated directly against this app's real compiled graph in
`backend/scripts/repro_time_travel.py` before relying on it here (same
"verify the tricky LangGraph mechanic experimentally first" discipline as the
conditional-entry-point and interrupt/`get_config` investigations above).
Two things aren't obvious from LangGraph's docs alone:

- `graph.aget_state(config)` with an explicit `checkpoint_id` in `config`
  always returns that one frozen historical snapshot — it never reflects
  anything computed later. Only the bare `{"configurable": {"thread_id":
  ...}}` config (no `checkpoint_id`) resolves to the thread's actual
  current/latest state. Passing a pinned historical config to a *continuation*
  call (`Command(resume=...)`) just re-runs from that same frozen point again
  instead of advancing — this bit the first draft of the diagnostic script.
- To rewind without touching any of the target checkpoint's own already-computed
  values, fork with `graph.aupdate_state(target.config, None, as_node="__copy__")`.
  This clones that checkpoint as the thread's new tip. The existing resume
  endpoints (`/evaluation-decision`, `/clarify-requirements`, `/clarify-gaps`,
  `/checklist-signoff`) need **zero changes** for this to work, since they
  already call `Command(resume=...)` against the bare thread config
  (`_config(session_id)`), which naturally picks up the rewound checkpoint as
  "current" on the very next call.

`rewind_session` (`session.py`) scans `graph.aget_state_history(config)`
newest-first and forks the **first** (i.e. most recent) snapshot whose `next`
matches `target` — for a multi-round clarification loop (`max_clarification_rounds`
up to 5 via the evaluation-gate override) this means "redo my last answer,"
not picking an arbitrary earlier round, so no `round` parameter is needed.
`_to_response` needs no changes at all: it already derives `awaiting_input`
purely from `snapshot.next`, which the rewind naturally restores. Rewinding
`evaluation_review` after an abort also correctly clears `workflow_aborted`
for free, since that field is only set by `evaluation_review_node`'s abort
branch — a checkpoint from *before* that node resolves never has it set. A
target never reached in this session's history (e.g. `ba_clarification` on a
session that resolved with 0 clarification rounds) 404s.

On the frontend, everything needed to decide whether a step's "go back" is
valid already lives on the existing `SessionStateResponse` — no new
client-side tracking, since `ambiguity_round`/`gap_round` stay meaningful
after that step resolves (`_to_response` always derives them from
`qa_history`/`gap_qa_history` length regardless of current `awaiting_input`).
The one exception is Upload: the backend never echoes `requirements_draft`/
`out_of_scope_details`/etc. back, so `App.jsx` keeps a client-side-only
`uploadDraft` (the raw payload last passed to `startSession`) purely to
pre-fill the form if the user goes back to it — going back to Upload doesn't
call `/rewind` at all, it just clears `session` and lets the user submit a
brand-new session, identical to how every session already starts. `File`
objects in that draft (attachments/legacy files) intentionally aren't
restored — not meaningfully re-creatable, and a disclosed limitation rather
than a bug. `WizardStepper.jsx` makes a step's circle/label clickable only
when it's strictly before the current step (`index < activeIndex`) **and**
the derived `canGoBack[step.key]` is true; clicking calls
`rewindSession`/resets `session`, and `stepKeyFor`'s existing render
branches pick up the restored `awaiting_input` with no changes needed there.

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

`UploadStep.jsx` now has a 3-option `workflow_mode` picker (`full`/
`refine_only`/`qa_direct`, each a radio-styled card) plus an
`out_of_scope_details` textarea (hidden for `qa_direct`, since that flow skips
the evaluator entirely) — `client.js`'s `startSession` forwards both as
`workflow_mode`/`out_of_scope_details` form fields. Field visibility/labels
adapt per mode: the requirements textarea is relabeled "Already-refined
requirements" for `qa_direct`, and the legacy-test-cases section is hidden
entirely for `refine_only` (irrelevant to a flow that never reaches the QA
matrix builder). `stepKeyFor` gains a `refined` branch — checked before the
`requirement_evaluation`/`ba_clarification` branches since a completed
`refine_only` session also has `polished_spec` set with no `awaiting_input`,
which would otherwise fall through to the same default `'matrix'` bucket a
normal Flow A completion does — rendering `RefineOnlyDoneStep.jsx` instead.
`WizardStepper.jsx`'s `STEPS` array doesn't have a `refined` entry, so (like
the existing `aborted` step) it renders with no step highlighted — an
accepted, pre-existing pattern for terminal states that aren't part of the
main A/B/C linear stepper.

**Still API-only**: Flow C (`format_only`) — it's fully functional via the
API (see `test_session_router.py`'s `format_only` tests), but nothing in
`UploadStep.jsx` lets a user pick it or choose its upfront target format; it
wasn't part of the ask that added the `workflow_mode` picker, only
`full`/`refine_only`/`qa_direct` were.

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
- No UI for Flow C's entry parameters (`workflow_mode: "format_only"` and its
  upfront format picker) — see "Frontend state machine" above. Full,
  refine_only, and qa_direct all have UI now.
