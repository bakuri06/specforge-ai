import csv
import io
import json
import logging
import re
from typing import Optional

from langgraph.types import interrupt

from app.config import settings
from app.graph.llm import _parse_json_with_repair, ollama_chat
from app.graph.prompts import (
    BA_REFINER_FORCE_RESOLVE_SYSTEM,
    BA_REFINER_SYSTEM,
    COMPILE_INSTRUCTION,
    FORMAT_SAMPLE_FILES,
    FORMATTER_FORMAT_RULES,
    QA_MATRIX_FORCE_RESOLVE_SYSTEM,
    QA_MATRIX_SYSTEM,
    REQUIREMENT_EVALUATOR_SYSTEM,
    TRANSLATE_INSTRUCTION,
    VISION_PROMPT,
)
from app.graph.state import SpecForgeState
from app.services.samples import few_shot_block
from app.services.vision_ocr import extract_text_from_screenshot

logger = logging.getLogger(__name__)

# Cap on how many rounds of clarifying questions either agent can ask before
# being forced to proceed with best-effort assumptions instead of looping
# forever if the model keeps finding new ambiguity. This is the fallback used
# when a session never goes through requirement_evaluator_node (Flows B/C,
# which skip it entirely) - Flow A sessions get a per-session override via
# state["max_clarification_rounds"] instead (see ba_refiner_node).
MAX_CLARIFICATION_ROUNDS = 1

# --- Multi-entry routing (see build.py for how these wire into the graph) ---


def route_entry(state: SpecForgeState) -> str:
    """START's conditional entry point. Defaults to "ingest" (Flows A/B) for
    any missing/unrecognized workflow_mode rather than raising, matching the
    rest of the codebase's stance of never trusting a decision-point input."""
    return "translate" if state.get("workflow_mode") == "format_only" else "ingest"


def route_after_ingest(state: SpecForgeState) -> str:
    """Second entry decision, after ingest_visual: Flow B (qa_direct) skips
    straight to the QA matrix builder with a pre-supplied polished_spec;
    everything else (including any unrecognized value) goes through the full
    BA pipeline starting at requirement_evaluator_node."""
    return "qa_direct" if state.get("workflow_mode") == "qa_direct" else "full"


# --- Phase 1: Visual Context Engine -----------------------------------------


async def ingest_visual_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
    session_id = state.get("session_id", "?")
    image_paths = state.get("image_paths") or []
    if not image_paths:
        logger.info("[%s] ingest_visual: no images attached, skipping", session_id)
        return {"visual_context": "", "stage": "ingest_visual"}

    logger.info("[%s] ingest_visual: analyzing %d image(s)", session_id, len(image_paths))
    model = state.get("vision_model") or settings.vision_model
    contexts = []
    for path in image_paths:
        # extract_text_from_screenshot preprocesses the image (resize/pad/
        # binarize) and falls back to deterministic OCR if the vision model
        # throws or returns degraded/gibberish output - see
        # app/services/vision_ocr.py.
        content = await extract_text_from_screenshot(path, model=model, prompt=VISION_PROMPT)
        contexts.append(content)
    logger.info("[%s] ingest_visual: done", session_id)
    return {"visual_context": "\n\n---\n\n".join(contexts), "stage": "ingest_visual"}


# --- Requirement Evaluation Gate (Flow A only) ------------------------------


def _coerce_score(value) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 50
    return max(0, min(100, score))


QUALITY_GATE_KEYS = (
    "data_and_boundaries",
    "integration_and_async_behavior",
    "network_and_resiliency",
    "state_and_lifecycle",
)


def _coerce_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_quality_gate_feedback(value) -> dict:
    """A smaller model sometimes ignores the categorized shape entirely and
    returns the old flat list (or a bare string) instead of a dict keyed by
    the four quality gates. Rather than guess which category a flat list's
    items belong to, fall back to all-empty-per-category - a safe default,
    same "never crash on wrong shape" precedent as _coerce_test_matrix's
    non-list fallback."""
    if not isinstance(value, dict):
        return {key: [] for key in QUALITY_GATE_KEYS}
    return {key: _coerce_string_list(value.get(key)) for key in QUALITY_GATE_KEYS}


def _coerce_round_count(value, default: int = 1) -> int:
    try:
        rounds = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(0, min(5, rounds))


async def requirement_evaluator_node(
    state: SpecForgeState, config: Optional[dict] = None
) -> dict:
    """Does the actual evaluation LLM call. Deliberately does NOT call
    interrupt() itself - that's evaluation_review_node's job. LangGraph
    replays a node's code from the top on every resume up to its interrupt()
    call, so any real work (like this LLM call) placed before an interrupt()
    in the SAME node gets silently re-executed on every resume. Splitting
    "do the work" from "pause for review" into two nodes (mirroring
    ba_refiner_node -> ba_clarification_node) avoids that entirely."""
    session_id = state.get("session_id", "?")
    logger.info("[%s] requirement_evaluator: starting", session_id)

    prompt = (
        f"{REQUIREMENT_EVALUATOR_SYSTEM}\n\n"
        f"# Requirements\n{state.get('requirements_draft', '')}\n\n"
        f"# Visual context\n{state.get('visual_context') or '(none)'}\n\n"
        f"# Out of scope\n{state.get('out_of_scope_details') or '(none specified)'}"
    )
    model = state.get("reasoning_model") or settings.reasoning_model
    result = await ollama_chat(model, prompt, expect_json=True)

    readiness_score = _coerce_score(result.get("readiness_score"))
    evaluation_feedback = _coerce_quality_gate_feedback(result.get("evaluation_feedback"))
    recommended_rounds = _coerce_round_count(result.get("recommended_clarification_rounds"))

    logger.info(
        "[%s] requirement_evaluator: score=%d, recommended_rounds=%d",
        session_id,
        readiness_score,
        recommended_rounds,
    )
    return {
        "readiness_score": readiness_score,
        "evaluation_feedback": evaluation_feedback,
        "recommended_clarification_rounds": recommended_rounds,
        "stage": "requirement_evaluator",
    }


async def evaluation_review_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
    """Thin interrupt-only node: pauses with the already-computed evaluation
    results and handles the resume decision. No LLM call here, so replay on
    resume is free (matches ba_clarification_node/gap_clarification_node/
    checklist_signoff_node's existing convention)."""
    session_id = state.get("session_id", "?")
    logger.info(
        "[%s] evaluation_review: paused for review (score=%s)",
        session_id,
        state.get("readiness_score"),
    )
    decision = interrupt(
        {
            "type": "requirement_evaluation",
            "readiness_score": state.get("readiness_score"),
            "evaluation_feedback": state.get(
                "evaluation_feedback", {key: [] for key in QUALITY_GATE_KEYS}
            ),
            "recommended_clarification_rounds": state.get("recommended_clarification_rounds"),
        }
    )
    if not isinstance(decision, dict):
        decision = {}

    action = decision.get("action", "proceed")
    logger.info("[%s] evaluation_review: resumed with action=%s", session_id, action)

    if action == "abort":
        return {"workflow_aborted": True, "stage": "aborted"}

    recommended_rounds = state.get("recommended_clarification_rounds", 1)
    chosen_rounds = _coerce_round_count(
        decision.get("max_clarification_rounds"), default=recommended_rounds
    )
    return {
        "workflow_aborted": False,
        "max_clarification_rounds": chosen_rounds,
        "current_clarification_round": 0,
        "stage": "evaluation_review",
    }


def route_after_evaluation(state: SpecForgeState) -> str:
    return "aborted" if state.get("workflow_aborted") else "continue"


# --- Phase 2: Agent 1 - BA Requirements Refiner ------------------------------


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


async def ba_refiner_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
    session_id = state.get("session_id", "?")
    rounds_used = len(state.get("qa_history", []))
    pass_number = rounds_used + 1
    max_rounds = state.get("max_clarification_rounds", MAX_CLARIFICATION_ROUNDS)
    force_resolve = rounds_used >= max_rounds
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
        f"# Out of scope\n{state.get('out_of_scope_details') or '(none specified)'}\n\n"
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


async def ba_clarification_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
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
    return {
        "qa_history": history,
        "current_clarification_round": len(history),
        "stage": "ba_clarification",
    }


def route_ambiguity(state: SpecForgeState) -> str:
    if not state.get("ambiguity_resolved"):
        return "clarify"
    # "refine_only" stops here instead of continuing into the QA matrix
    # builder - the polished_spec itself is the deliverable for this flow.
    return "stop" if state.get("workflow_mode") == "refine_only" else "resolved"


# --- Phase 3: Agent 2 - QA Test Matrix Builder -------------------------------

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


def _coerce_steps(raw_steps, fallback_text: str = "") -> list[dict]:
    """Repair a test case's steps into the hierarchical shape, falling back
    to a single synthesized step if the model returned a flat string/missing
    steps entirely instead of a list."""
    if isinstance(raw_steps, list) and raw_steps:
        fixed = []
        for i, step in enumerate(raw_steps):
            if isinstance(step, dict):
                fixed.append(
                    {
                        "step_number": int(step.get("step_number") or i + 1),
                        "action": str(step.get("action") or ""),
                        "expected_result": str(step.get("expected_result") or ""),
                    }
                )
            elif step:
                fixed.append(
                    {"step_number": i + 1, "action": str(step), "expected_result": ""}
                )
        if fixed:
            return fixed
    if isinstance(raw_steps, str) and raw_steps.strip():
        return [{"step_number": 1, "action": raw_steps.strip(), "expected_result": ""}]
    if fallback_text:
        return [{"step_number": 1, "action": fallback_text, "expected_result": ""}]
    return []


def _coerce_test_matrix(raw_matrix) -> list[dict]:
    if not isinstance(raw_matrix, list):
        return []
    fixed = []
    for i, item in enumerate(raw_matrix):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Untitled scenario")
        steps = _coerce_steps(item.get("steps"), fallback_text=str(item.get("description") or ""))
        fixed.append(
            {
                "id": str(item.get("id") or f"TC-{i + 1}"),
                "category": _coerce_enum(item.get("category"), VALID_CATEGORIES, "edge_case"),
                "title": title,
                "steps": steps,
                "status": _coerce_enum(item.get("status"), VALID_STATUSES, "new"),
                "included": bool(item.get("included", True)),
            }
        )
    return fixed


async def qa_matrix_builder_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
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


async def gap_clarification_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
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


async def checklist_signoff_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
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


async def _validate_and_repair_json(output: str, model: str, original_prompt: str) -> str:
    """jira_xray output must be valid JSON. Reuses the same <think>-stripping /
    brace-extraction / strict=False tolerance already proven for structured
    model calls elsewhere (app.graph.llm._parse_json_with_repair), then
    re-serializes for a canonical, guaranteed-valid string instead of storing
    the model's raw (possibly still slightly malformed) text. One corrective
    retry on failure, mirroring ollama_chat's own JSON retry pattern."""
    try:
        parsed = _parse_json_with_repair(output)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        logger.warning("formatter: jira_xray output was not valid JSON, retrying once")
        corrected_prompt = (
            f"{original_prompt}\n\n"
            "Your previous response was not valid JSON:\n"
            f"{output}\n\n"
            "Respond again with ONLY the JSON object described above. No prose, "
            "no markdown code fences, no <think> reasoning blocks."
        )
        retry_output = await ollama_chat(model, corrected_prompt)
        try:
            parsed = _parse_json_with_repair(retry_output)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return retry_output


def _csv_row_widths(text: str) -> list[int]:
    reader = csv.reader(io.StringIO(text))
    return [len(row) for row in reader if row]


async def _validate_and_repair_csv(output: str, model: str, original_prompt: str) -> str:
    """qtest/azure_devops output must be well-formed CSV with a consistent
    column count across every row. One corrective retry on a shape mismatch,
    same pattern as the JSON path above."""
    widths = _csv_row_widths(output)
    if widths and len(set(widths)) == 1:
        return output

    logger.warning(
        "formatter: CSV output has inconsistent column widths %s, retrying once", widths
    )
    corrected_prompt = (
        f"{original_prompt}\n\n"
        "Your previous response was not valid CSV — every row must have the "
        f"same number of columns as the header row:\n{output}\n\n"
        "Respond again with ONLY the corrected CSV, same column structure "
        "throughout, properly quoting any field that itself contains a comma."
    )
    return await ollama_chat(model, corrected_prompt)


_FEATURE_LINE_RE = re.compile(r"(?m)^\s*Feature:.*$")


def _merge_multiple_bdd_features(output: str) -> str:
    """A generated .feature output must have exactly one root Feature: - a
    local model asked to compile several test cases sometimes emits a
    separate Feature: per case instead of nesting them all as Scenario:
    blocks under one. Unlike JSON/CSV validity, this is always mechanically
    fixable without gambling on a re-prompt: delete every Feature: line after
    the first, so their Scenario/Scenario Outline blocks fall through
    unchanged and end up nested under the single remaining Feature."""
    matches = list(_FEATURE_LINE_RE.finditer(output))
    if len(matches) <= 1:
        return output
    logger.warning(
        "formatter: bdd output had %d 'Feature:' lines, merging into one",
        len(matches),
    )
    pieces, last_end = [], 0
    for match in matches[1:]:
        pieces.append(output[last_end : match.start()])
        last_end = match.end()
    pieces.append(output[last_end:])
    return "".join(pieces)


async def formatter_node(state: SpecForgeState, config: Optional[dict] = None) -> dict:
    session_id = state.get("session_id", "?")
    fmt = state.get("output_format", "testrail")
    is_translation = state.get("workflow_mode") == "format_only"
    logger.info(
        "[%s] formatter: compiling output as %s (%s mode)",
        session_id,
        fmt,
        "translate" if is_translation else "compile",
    )

    rules = FORMATTER_FORMAT_RULES.get(fmt, FORMATTER_FORMAT_RULES["testrail"])
    example = few_shot_block(FORMAT_SAMPLE_FILES.get(fmt, FORMAT_SAMPLE_FILES["testrail"]))

    if is_translation:
        instructions = TRANSLATE_INSTRUCTION.format(fmt_name=fmt, rules=rules)
        source = state.get("legacy_test_cases", "")
        prompt = f"{instructions}\n\n{example}\n\n# Existing test cases\n{source}"
    else:
        instructions = COMPILE_INSTRUCTION.format(fmt_name=fmt, rules=rules)
        prompt = (
            f"{instructions}\n\n{example}\n\n"
            f"# Test matrix (JSON)\n{state.get('test_matrix', [])}"
        )

    model = state.get("formatter_model") or settings.formatter_model
    output = await ollama_chat(model, prompt)

    if fmt == "jira_xray":
        output = await _validate_and_repair_json(output, model, prompt)
    elif fmt in ("qtest", "azure_devops"):
        output = await _validate_and_repair_csv(output, model, prompt)
    elif fmt == "bdd":
        output = _merge_multiple_bdd_features(output)

    logger.info("[%s] formatter: done (%d chars)", session_id, len(output))
    return {"formatted_output": output, "stage": "formatter"}
