import logging
import re

from langgraph.types import interrupt

from app.config import settings
from app.graph.llm import ollama_chat
from app.graph.state import SpecForgeState

logger = logging.getLogger(__name__)

# Cap on how many rounds of clarifying questions either agent can ask before
# being forced to proceed with best-effort assumptions instead of looping
# forever if the model keeps finding new ambiguity.
MAX_CLARIFICATION_ROUNDS = 1

# --- Phase 1: Visual Context Engine -----------------------------------------


async def ingest_visual_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    image_paths = state.get("image_paths") or []
    if not image_paths:
        logger.info("[%s] ingest_visual: no images attached, skipping", session_id)
        return {"visual_context": "", "stage": "ingest_visual"}

    logger.info("[%s] ingest_visual: analyzing %d image(s)", session_id, len(image_paths))
    model = state.get("vision_model") or settings.vision_model
    contexts = []
    for path in image_paths:
        content = await ollama_chat(
            model,
            "Analyze this UI screenshot. Produce a markdown map of every visible "
            "element: accessibility label, element type, input constraints, and "
            "layout order (top to bottom, left to right).",
            images=[path],
        )
        contexts.append(content)
    logger.info("[%s] ingest_visual: done", session_id)
    return {"visual_context": "\n\n---\n\n".join(contexts), "stage": "ingest_visual"}


# --- Phase 2: Agent 1 - BA Requirements Refiner ------------------------------

BA_REFINER_SPEC_SECTIONS = """Use exactly these top-level sections, in this order
(omit a section only if there is truly nothing to put in it):

## Overview
## User Flow
## Business Rules
## Error Handling & Edge Cases
## Data Retention & Validation
## Out of Scope"""

BA_REFINER_SYSTEM = f"""You are a senior Business Analyst turning raw input into a
technical requirements blueprint precise enough to write test cases directly
against. Given raw requirements text (optionally with a visual UI element map
appended) and any prior clarification Q&A, evaluate whether the requirements are
complete enough to design test cases against.

Check specifically for: missing error boundaries, undefined network timeout behavior,
missing data retention/validation rules, and undefined edge-case business rules.

Respond with ONLY a JSON object, no prose, matching this shape:
{{"ambiguous": true|false, "questions": ["...", "..."], "polished_spec": "..."}}

- If ambiguous is true: include exactly 2-3 targeted clarifying questions in
  "questions" and leave "polished_spec" empty.
- If ambiguous is false: leave "questions" empty and write the full requirements
  blueprint in "polished_spec" as structured markdown. It must MERGE the original
  requirements with every answer from the clarification Q&A below — fold each
  answer into the section it belongs to as a concrete requirement, do not just
  append the raw Q&A at the end. Preserve every specific detail from the original
  input (numbers, thresholds, field names, business rules) — do not drop or
  generalize them away. Be thorough: several sentences of concrete behavior per
  section, not a one-line restatement of the section title.

IMPORTANT: "polished_spec" must be ONE markdown STRING containing all the
section headings below inside that single string (e.g. "## Overview\\nSome
text\\n\\n## User Flow\\n..."). It must NOT be a JSON object/dictionary with
each section name as a separate key.

{BA_REFINER_SPEC_SECTIONS}
"""

BA_REFINER_FORCE_RESOLVE_SYSTEM = f"""You are a senior Business Analyst. You have
already used up the maximum number of clarification rounds allowed. You must now
produce a final requirements blueprint using the information gathered so far, even
if some ambiguity remains.

Merge the original requirements with every answer from the clarification Q&A below
into the sections below, preserving every specific detail from the original input.
For anything still unresolved, make a reasonable assumption within the relevant
section instead of asking another question, and also list it under
"## Assumptions" at the end.

Respond with ONLY a JSON object, no prose, matching this shape:
{{"ambiguous": false, "questions": [], "polished_spec": "..."}}

IMPORTANT: "polished_spec" must be ONE markdown STRING containing all the
section headings below inside that single string. It must NOT be a JSON
object/dictionary with each section name as a separate key.

{BA_REFINER_SPEC_SECTIONS}
## Assumptions
"""


def _format_qa_history(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    blocks = []
    for entry in history:
        pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(entry["questions"], entry["answers"])
        )
        blocks.append(pairs)
    return "\n\n".join(blocks)


def _fallback_spec(state: SpecForgeState) -> str:
    """Last-resort spec if a forced-resolve call still comes back empty."""
    parts = ["## Overview\n" + state.get("requirements_draft", "")]
    history = state.get("qa_history", [])
    if history:
        parts.append("## Assumptions / Clarifications Gathered\n" + _format_qa_history(history))
    return "\n\n".join(parts)


def _coerce_polished_spec(value) -> str:
    """A smaller model sometimes returns polished_spec as a JSON object with
    each section as a key (e.g. {"Overview": "...", "User Flow": "..."})
    instead of one markdown string, despite the prompt saying not to. Repair
    it into a string rather than letting an invalid shape crash the response."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n\n".join(f"## {key}\n{val}" for key, val in value.items())
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return str(value) if value else ""


async def ba_refiner_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    rounds_used = len(state.get("qa_history", []))
    pass_number = rounds_used + 1
    force_resolve = rounds_used >= MAX_CLARIFICATION_ROUNDS
    logger.info(
        "[%s] ba_refiner: starting (pass %d)%s",
        session_id,
        pass_number,
        " [max rounds reached, forcing resolution]" if force_resolve else "",
    )

    system_prompt = BA_REFINER_FORCE_RESOLVE_SYSTEM if force_resolve else BA_REFINER_SYSTEM
    prompt = (
        f"{system_prompt}\n\n"
        f"# Requirements\n{state.get('requirements_draft', '')}\n\n"
        f"# Visual context\n{state.get('visual_context') or '(none)'}\n\n"
        f"# Prior clarifications\n{_format_qa_history(state.get('qa_history', []))}"
    )
    model = state.get("reasoning_model") or settings.reasoning_model
    result = await ollama_chat(model, prompt, expect_json=True)

    if force_resolve:
        polished_spec = _coerce_polished_spec(result.get("polished_spec")) or _fallback_spec(
            state
        )
        logger.info(
            "[%s] ba_refiner: forced resolution, polished spec ready (%d chars)",
            session_id,
            len(polished_spec),
        )
        return {
            "polished_spec": polished_spec,
            "ambiguity_resolved": True,
            "stage": "ba_refiner",
        }

    if result.get("ambiguous"):
        questions = result.get("questions", [])[:3]
        logger.info(
            "[%s] ba_refiner: ambiguous, %d clarifying question(s)",
            session_id,
            len(questions),
        )
        return {
            "ambiguity_questions": questions,
            "ambiguity_resolved": False,
            "stage": "ba_refiner",
        }
    polished_spec = _coerce_polished_spec(result.get("polished_spec", ""))
    logger.info(
        "[%s] ba_refiner: resolved, polished spec ready (%d chars)",
        session_id,
        len(polished_spec),
    )
    return {
        "polished_spec": polished_spec,
        "ambiguity_resolved": True,
        "stage": "ba_refiner",
    }


async def ba_clarification_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    logger.info(
        "[%s] ba_clarification: paused, awaiting answers to %d question(s)",
        session_id,
        len(state["ambiguity_questions"]),
    )
    answers = interrupt(
        {"type": "ba_clarification", "questions": state["ambiguity_questions"]}
    )
    logger.info("[%s] ba_clarification: resumed with answers", session_id)
    history = state.get("qa_history", []) + [
        {"questions": state["ambiguity_questions"], "answers": answers}
    ]
    return {"qa_history": history, "stage": "ba_clarification"}


def route_ambiguity(state: SpecForgeState) -> str:
    return "resolved" if state.get("ambiguity_resolved") else "clarify"


# --- Phase 3: Agent 2 - QA Test Matrix Builder -------------------------------

QA_MATRIX_SYSTEM = """You are a senior QA Engineer building a test strategy matrix.
Given a polished requirements blueprint and (optionally) a legacy test suite, perform
delta analysis: which legacy cases are now broken by the new requirements, which need
modification, and where new coverage is required. Then produce a full test scenario
checklist.

Respond with ONLY a JSON object, no prose, matching this shape:
{
  "gaps_found": true|false,
  "questions": ["...", "..."],
  "test_matrix": [
    {"id": "TC-1", "category": "sunny_day", "title": "...", "description": "...",
     "status": "new", "included": true}
  ]
}

"category" must be exactly ONE of these four words: sunny_day, rainy_day,
boundary, edge_case. Never combine multiple values and never include a "|"
character in the value.
"status" must be exactly ONE of these four words: new, modified, broken,
unchanged.

- If gaps_found is true: include exactly 2-3 targeted questions about the coverage
  gaps and leave test_matrix empty.
- If gaps_found is false: leave questions empty and return the complete matrix,
  grouped logically across Sunny Day, Rainy Day, Boundaries, and Edge Cases.
"""

QA_MATRIX_FORCE_RESOLVE_SYSTEM = """You are a senior QA Engineer. You have already
used up the maximum number of clarification rounds allowed for coverage gaps. You
must now produce the final test strategy matrix using the information gathered so
far, even if some coverage questions remain unresolved — make reasonable
assumptions for anything still unclear instead of asking again.

Respond with ONLY a JSON object, no prose, matching this shape:
{
  "gaps_found": false,
  "questions": [],
  "test_matrix": [
    {"id": "TC-1", "category": "sunny_day", "title": "...", "description": "...",
     "status": "new", "included": true}
  ]
}

"category" must be exactly ONE of these four words: sunny_day, rainy_day,
boundary, edge_case. Never combine multiple values and never include a "|"
character in the value.
"status" must be exactly ONE of these four words: new, modified, broken,
unchanged.
"""

VALID_CATEGORIES = {"sunny_day", "rainy_day", "boundary", "edge_case"}
VALID_STATUSES = {"new", "modified", "broken", "unchanged"}


def _coerce_enum(value, valid_values: set, fallback: str) -> str:
    """A smaller model sometimes copies the prompt's own placeholder notation
    verbatim (e.g. "sunny_day|rainy_day|boundary|edge_case") instead of
    picking one value. Recover a valid value from that rather than crashing."""
    if value in valid_values:
        return value
    if isinstance(value, str):
        for token in re.split(r"[|,/]", value):
            token = token.strip()
            if token in valid_values:
                return token
    return fallback


def _coerce_test_matrix(raw_matrix) -> list[dict]:
    if not isinstance(raw_matrix, list):
        return []
    fixed = []
    for i, item in enumerate(raw_matrix):
        if not isinstance(item, dict):
            continue
        fixed.append(
            {
                "id": str(item.get("id") or f"TC-{i + 1}"),
                "category": _coerce_enum(item.get("category"), VALID_CATEGORIES, "edge_case"),
                "title": str(item.get("title") or "Untitled scenario"),
                "description": str(item.get("description") or ""),
                "status": _coerce_enum(item.get("status"), VALID_STATUSES, "new"),
                "included": bool(item.get("included", True)),
            }
        )
    return fixed


async def qa_matrix_builder_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    rounds_used = len(state.get("gap_qa_history", []))
    pass_number = rounds_used + 1
    force_resolve = rounds_used >= MAX_CLARIFICATION_ROUNDS
    logger.info(
        "[%s] qa_matrix_builder: starting (pass %d)%s",
        session_id,
        pass_number,
        " [max rounds reached, forcing resolution]" if force_resolve else "",
    )

    system_prompt = (
        QA_MATRIX_FORCE_RESOLVE_SYSTEM if force_resolve else QA_MATRIX_SYSTEM
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"# Polished requirements\n{state.get('polished_spec', '')}\n\n"
        f"# Legacy test cases\n{state.get('legacy_test_cases') or '(none provided)'}\n\n"
        f"# Prior clarifications\n{_format_qa_history(state.get('gap_qa_history', []))}"
    )
    model = state.get("reasoning_model") or settings.reasoning_model
    result = await ollama_chat(model, prompt, expect_json=True)

    if force_resolve:
        test_matrix = _coerce_test_matrix(result.get("test_matrix", []))
        logger.info(
            "[%s] qa_matrix_builder: forced resolution, %d test scenario(s) generated",
            session_id,
            len(test_matrix),
        )
        return {
            "test_matrix": test_matrix,
            "gaps_resolved": True,
            "stage": "qa_matrix_builder",
        }

    if result.get("gaps_found"):
        questions = result.get("questions", [])[:3]
        logger.info(
            "[%s] qa_matrix_builder: gaps found, %d clarifying question(s)",
            session_id,
            len(questions),
        )
        return {
            "gap_questions": questions,
            "gaps_resolved": False,
            "stage": "qa_matrix_builder",
        }
    test_matrix = _coerce_test_matrix(result.get("test_matrix", []))
    logger.info(
        "[%s] qa_matrix_builder: resolved, %d test scenario(s) generated",
        session_id,
        len(test_matrix),
    )
    return {
        "test_matrix": test_matrix,
        "gaps_resolved": True,
        "stage": "qa_matrix_builder",
    }


async def gap_clarification_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    logger.info(
        "[%s] gap_clarification: paused, awaiting answers to %d question(s)",
        session_id,
        len(state["gap_questions"]),
    )
    answers = interrupt(
        {"type": "gap_clarification", "questions": state["gap_questions"]}
    )
    logger.info("[%s] gap_clarification: resumed with answers", session_id)
    history = state.get("gap_qa_history", []) + [
        {"questions": state["gap_questions"], "answers": answers}
    ]
    return {"gap_qa_history": history, "stage": "gap_clarification"}


def route_gaps(state: SpecForgeState) -> str:
    return "resolved" if state.get("gaps_resolved") else "clarify"


# --- Checklist intercept (human sign-off before formatting) ------------------


async def checklist_signoff_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    logger.info(
        "[%s] checklist_signoff: paused, awaiting sign-off on %d scenario(s)",
        session_id,
        len(state.get("test_matrix", [])),
    )
    decision = interrupt(
        {"type": "checklist_signoff", "test_matrix": state.get("test_matrix", [])}
    )
    logger.info(
        "[%s] checklist_signoff: resumed, %d scenario(s) signed off for format=%s",
        session_id,
        len(decision["test_matrix"]),
        decision["output_format"],
    )
    return {
        "test_matrix": decision["test_matrix"],
        "output_format": decision["output_format"],
        "stage": "checklist_signoff",
    }


# --- Phase 4: Agent 3 - Formatter Router -------------------------------------

FORMATTER_PROMPTS = {
    "testrail": (
        "Compile the following signed-off test matrix into TestRail-ready Markdown "
        "(one table per category: Sunny Day, Rainy Day, Boundaries, Edge Cases). "
        "Columns: ID, Title, Preconditions, Steps, Expected Result. Output ONLY the "
        "markdown, no commentary, no code fences."
    ),
    "qtest": (
        "Compile the following signed-off test matrix into a strict comma-delimited "
        "CSV importable into qTest. Columns: Module,Precondition,Step,Step "
        "Description,Expected Result. Output ONLY the raw CSV, no commentary, no "
        "code fences."
    ),
    "playwright": (
        "Compile the following signed-off test matrix into a Playwright TypeScript "
        "test file skeleton, using semantic/text-based locators (getByRole, "
        "getByText, getByLabel). One test() per scenario, grouped into describe() "
        "blocks per category. Output ONLY the TypeScript code, no commentary, no "
        "code fences."
    ),
}


async def formatter_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    fmt = state.get("output_format", "testrail")
    logger.info("[%s] formatter: compiling output as %s", session_id, fmt)

    instructions = FORMATTER_PROMPTS.get(fmt, FORMATTER_PROMPTS["testrail"])
    prompt = f"{instructions}\n\n# Test matrix (JSON)\n{state.get('test_matrix', [])}"
    model = state.get("formatter_model") or settings.formatter_model
    output = await ollama_chat(model, prompt)
    logger.info("[%s] formatter: done (%d chars)", session_id, len(output))
    return {"formatted_output": output, "stage": "formatter"}
