import logging

from langgraph.types import interrupt

from app.config import settings
from app.graph.llm import ollama_chat
from app.graph.state import SpecForgeState

logger = logging.getLogger(__name__)

# --- Phase 1: Visual Context Engine -----------------------------------------


async def ingest_visual_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    image_paths = state.get("image_paths") or []
    if not image_paths:
        logger.info("[%s] ingest_visual: no images attached, skipping", session_id)
        return {"visual_context": "", "stage": "ingest_visual"}

    logger.info("[%s] ingest_visual: analyzing %d image(s)", session_id, len(image_paths))
    contexts = []
    for path in image_paths:
        content = await ollama_chat(
            settings.vision_model,
            "Analyze this UI screenshot. Produce a markdown map of every visible "
            "element: accessibility label, element type, input constraints, and "
            "layout order (top to bottom, left to right).",
            images=[path],
        )
        contexts.append(content)
    logger.info("[%s] ingest_visual: done", session_id)
    return {"visual_context": "\n\n---\n\n".join(contexts), "stage": "ingest_visual"}


# --- Phase 2: Agent 1 - BA Requirements Refiner ------------------------------

BA_REFINER_SYSTEM = """You are a senior Business Analyst. Given raw requirements text
(optionally with a visual UI element map appended) and any prior clarification Q&A,
evaluate whether the requirements are complete enough to design test cases against.

Check specifically for: missing error boundaries, undefined network timeout behavior,
missing data retention/validation rules, and undefined edge-case business rules.

Respond with ONLY a JSON object, no prose, matching this shape:
{"ambiguous": true|false, "questions": ["...", "..."], "polished_spec": "..."}

- If ambiguous is true: include exactly 2-3 targeted clarifying questions in
  "questions" and leave "polished_spec" empty.
- If ambiguous is false: leave "questions" empty and put the full, refined technical
  requirements blueprint in "polished_spec" as clean markdown.
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


async def ba_refiner_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    pass_number = len(state.get("qa_history", [])) + 1
    logger.info("[%s] ba_refiner: starting (pass %d)", session_id, pass_number)

    prompt = (
        f"{BA_REFINER_SYSTEM}\n\n"
        f"# Requirements\n{state.get('requirements_draft', '')}\n\n"
        f"# Visual context\n{state.get('visual_context') or '(none)'}\n\n"
        f"# Prior clarifications\n{_format_qa_history(state.get('qa_history', []))}"
    )
    result = await ollama_chat(settings.reasoning_model, prompt, expect_json=True)

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
    polished_spec = result.get("polished_spec", "")
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


def ba_clarification_node(state: SpecForgeState) -> dict:
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
    {"id": "TC-1", "category": "sunny_day|rainy_day|boundary|edge_case",
     "title": "...", "description": "...",
     "status": "new|modified|broken|unchanged", "included": true}
  ]
}

- If gaps_found is true: include exactly 2-3 targeted questions about the coverage
  gaps and leave test_matrix empty.
- If gaps_found is false: leave questions empty and return the complete matrix,
  grouped logically across Sunny Day, Rainy Day, Boundaries, and Edge Cases.
"""


async def qa_matrix_builder_node(state: SpecForgeState) -> dict:
    session_id = state.get("session_id", "?")
    pass_number = len(state.get("gap_qa_history", [])) + 1
    logger.info("[%s] qa_matrix_builder: starting (pass %d)", session_id, pass_number)

    prompt = (
        f"{QA_MATRIX_SYSTEM}\n\n"
        f"# Polished requirements\n{state.get('polished_spec', '')}\n\n"
        f"# Legacy test cases\n{state.get('legacy_test_cases') or '(none provided)'}\n\n"
        f"# Prior clarifications\n{_format_qa_history(state.get('gap_qa_history', []))}"
    )
    result = await ollama_chat(settings.reasoning_model, prompt, expect_json=True)

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
    test_matrix = result.get("test_matrix", [])
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


def gap_clarification_node(state: SpecForgeState) -> dict:
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


def checklist_signoff_node(state: SpecForgeState) -> dict:
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
    output = await ollama_chat(settings.formatter_model, prompt)
    logger.info("[%s] formatter: done (%d chars)", session_id, len(output))
    return {"formatted_output": output, "stage": "formatter"}
