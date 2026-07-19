import cv2
import numpy as np
from langgraph.types import Command

from app.graph import nodes as nodes_module
from app.graph.build import graph
from app.services import vision_ocr


def _initial_state(
    session_id: str,
    requirements: str,
    legacy_test_cases: str = "",
    workflow_mode: str = "full",
    **extra,
) -> dict:
    state = {
        "session_id": session_id,
        "workflow_mode": workflow_mode,
        "requirements_draft": requirements,
        "image_paths": [],
        "legacy_test_cases": legacy_test_cases,
        "qa_history": [],
        "gap_qa_history": [],
    }
    state.update(extra)
    return state


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _is_evaluator_prompt(prompt: str) -> bool:
    return "recommended_clarification_rounds" in prompt


def _is_ba_refiner_prompt(prompt: str) -> bool:
    # Both the evaluator and BA refiner prompts mention "senior Business
    # Analyst", so this must be checked after _is_evaluator_prompt, not
    # instead of it. "Core Calculation Framework" only appears in the BA
    # refiner's section-template constant.
    return "Core Calculation Framework" in prompt


def _is_qa_matrix_prompt(prompt: str) -> bool:
    return "senior QA Engineer" in prompt


async def _resume_evaluator(config, action="proceed", max_clarification_rounds=None):
    payload = {"action": action}
    if max_clarification_rounds is not None:
        payload["max_clarification_rounds"] = max_clarification_rounds
    return await graph.ainvoke(Command(resume=payload), config=config)


async def test_ambiguity_and_gap_clarification_loops(monkeypatch):
    """Both agents ask a clarifying question once, then resolve on the next pass."""
    ba_calls = {"count": 0}
    qa_calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 40,
                "evaluation_feedback": {"network_and_resiliency": ["Timeout behavior undefined"]},
                "recommended_clarification_rounds": 1,
            }
        if _is_ba_refiner_prompt(prompt):
            ba_calls["count"] += 1
            if ba_calls["count"] == 1:
                return {
                    "ambiguous": True,
                    "questions": ["What happens on request timeout?", "How long is data retained?"],
                    "polished_spec": "",
                }
            return {"ambiguous": False, "questions": [], "polished_spec": "# Polished Spec"}

        if _is_qa_matrix_prompt(prompt):
            qa_calls["count"] += 1
            if qa_calls["count"] == 1:
                return {
                    "gaps_found": True,
                    "questions": ["Should concurrent submissions be tested?"],
                    "test_matrix": [],
                }
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day",
                        "title": "Happy path submission",
                        "steps": [
                            {
                                "step_number": 1,
                                "action": "Submit the form with valid data",
                                "expected_result": "Submission succeeds",
                            }
                        ],
                        "status": "new",
                        "included": True,
                    }
                ],
            }

        return "FORMATTED OUTPUT"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-loops"
    config = _config(session_id)

    await graph.ainvoke(
        _initial_state(session_id, "Users can submit a form."), config=config
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("evaluation_review",)

    await _resume_evaluator(config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("ba_clarification",)
    assert snapshot.values["ambiguity_questions"] == [
        "What happens on request timeout?",
        "How long is data retained?",
    ]

    await graph.ainvoke(
        Command(resume=["Times out after 30s.", "Retained for 90 days."]),
        config=config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("gap_clarification",)
    assert snapshot.values["polished_spec"] == "# Polished Spec"
    assert ba_calls["count"] == 2

    await graph.ainvoke(
        Command(resume=["Yes, add a concurrency test."]), config=config
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert len(snapshot.values["test_matrix"]) == 1
    assert qa_calls["count"] == 2

    edited_matrix = snapshot.values["test_matrix"] + [
        {
            "id": "TC-2",
            "category": "edge_case",
            "title": "Concurrent submissions",
            "steps": [
                {
                    "step_number": 1,
                    "action": "Submit the same form twice at once",
                    "expected_result": "Only one submission is accepted",
                }
            ],
            "status": "new",
            "included": True,
        }
    ]
    await graph.ainvoke(
        Command(resume={"test_matrix": edited_matrix, "output_format": "testrail"}),
        config=config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["formatted_output"] == "FORMATTED OUTPUT"
    assert snapshot.values["output_format"] == "testrail"
    assert len(snapshot.values["test_matrix"]) == 2


async def test_straight_through_when_spec_and_matrix_are_clean(monkeypatch):
    """No ambiguity, no gaps: both agents should resolve on their first call."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 95,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {"ambiguous": False, "questions": [], "polished_spec": "Clean spec"}
        if _is_qa_matrix_prompt(prompt):
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day",
                        "title": "Happy path",
                        "steps": [
                            {"step_number": 1, "action": "a", "expected_result": "r"}
                        ],
                        "status": "new",
                        "included": True,
                    }
                ],
            }
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-straight-through"
    config = _config(session_id)

    await graph.ainvoke(
        _initial_state(session_id, "Fully specified requirements."), config=config
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("evaluation_review",)

    await _resume_evaluator(config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert snapshot.values["polished_spec"] == "Clean spec"

    await graph.ainvoke(
        Command(
            resume={
                "test_matrix": snapshot.values["test_matrix"],
                "output_format": "qtest",
            }
        ),
        config=config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["formatted_output"] == "FORMATTED"


async def test_requirement_evaluation_abort_stops_pipeline(monkeypatch):
    """The user can abort right after seeing the readiness evaluation instead
    of proceeding into the BA clarification loop."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 20,
                "evaluation_feedback": {"data_and_boundaries": ["Way too vague"]},
                "recommended_clarification_rounds": 3,
            }
        raise AssertionError("no further model calls expected after abort")

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-abort"
    config = _config(session_id)

    await graph.ainvoke(_initial_state(session_id, "Vague."), config=config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("evaluation_review",)

    await _resume_evaluator(config, action="abort")
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["workflow_aborted"] is True
    assert snapshot.values["stage"] == "aborted"
    assert snapshot.values["readiness_score"] == 20


async def test_max_clarification_rounds_zero_skips_ba_loop_entirely(monkeypatch):
    """Overriding max_clarification_rounds to 0 at the evaluation gate must
    force-resolve on the very first ba_refiner call, even though the model
    itself reports the input as ambiguous."""
    ba_calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 50,
                "evaluation_feedback": {"data_and_boundaries": ["Some gaps"]},
                "recommended_clarification_rounds": 2,
            }
        if _is_ba_refiner_prompt(prompt):
            ba_calls["count"] += 1
            return {
                "ambiguous": True,
                "questions": ["Would ask this if allowed"],
                "polished_spec": "",
            }
        if _is_qa_matrix_prompt(prompt):
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-zero-rounds"
    config = _config(session_id)

    await graph.ainvoke(_initial_state(session_id, "Some requirements."), config=config)
    await _resume_evaluator(config, max_clarification_rounds=0)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert snapshot.values["ambiguity_resolved"] is True
    assert ba_calls["count"] == 1
    assert snapshot.values["max_clarification_rounds"] == 0


async def test_ba_refiner_forces_resolution_after_max_rounds(monkeypatch):
    """If the model keeps finding new ambiguity forever, the app must not get
    stuck in an infinite clarification loop — it must force a resolution once
    the session's max_clarification_rounds have been used, regardless of what
    the model itself reports."""
    ba_calls = {"count": 0}
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 50,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 1,
            }
        if _is_ba_refiner_prompt(prompt):
            ba_calls["count"] += 1
            seen_prompts.append(prompt)
            return {
                "ambiguous": True,
                "questions": [f"Question round {ba_calls['count']}"],
                "polished_spec": "",
            }
        if _is_qa_matrix_prompt(prompt):
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-ba-cap"
    config = _config(session_id)

    await graph.ainvoke(_initial_state(session_id, "Vague requirements."), config=config)
    await _resume_evaluator(config)

    max_rounds = 1
    for _ in range(max_rounds):
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("ba_clarification",)
        await graph.ainvoke(Command(resume=["some answer"]), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert snapshot.values["ambiguity_resolved"] is True
    assert ba_calls["count"] == max_rounds + 1
    assert "maximum number of clarification rounds" in seen_prompts[-1]
    # The mock never provides a polished_spec even when forced, so the
    # fallback synthesis (raw requirements + gathered Q&A) must kick in.
    assert "Vague requirements." in snapshot.values["polished_spec"]


async def test_qa_matrix_builder_forces_resolution_after_max_rounds(monkeypatch):
    """Same safety net as the BA loop, for Agent 2's gap-clarification loop.
    This loop is unaffected by the evaluator's max_clarification_rounds
    override — it stays on the module-level MAX_CLARIFICATION_ROUNDS."""
    qa_calls = {"count": 0}
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if _is_qa_matrix_prompt(prompt):
            qa_calls["count"] += 1
            seen_prompts.append(prompt)
            return {
                "gaps_found": True,
                "questions": [f"Gap question round {qa_calls['count']}"],
                "test_matrix": [],
            }
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-qa-cap"
    config = _config(session_id)

    await graph.ainvoke(_initial_state(session_id, "Some requirements."), config=config)
    await _resume_evaluator(config)

    for _ in range(nodes_module.MAX_CLARIFICATION_ROUNDS):
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("gap_clarification",)
        await graph.ainvoke(Command(resume=["some answer"]), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert snapshot.values["gaps_resolved"] is True
    assert qa_calls["count"] == nodes_module.MAX_CLARIFICATION_ROUNDS + 1
    assert "maximum number of clarification rounds" in seen_prompts[-1]


async def test_per_session_model_override_is_used_over_settings_default(monkeypatch):
    """Regression guard for the model selector: a session-level override must
    actually reach ollama_chat, not just get stored and ignored."""
    seen_models = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        seen_models.append(model)
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if _is_qa_matrix_prompt(prompt):
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-model-override"
    config = _config(session_id)
    initial_state = _initial_state(session_id, "Some requirements.")
    initial_state["reasoning_model"] = "deepseek-r1:7b"
    initial_state["formatter_model"] = "qwen2.5:7b"

    await graph.ainvoke(initial_state, config=config)
    await _resume_evaluator(config)
    await graph.ainvoke(
        Command(resume={"test_matrix": [], "output_format": "testrail"}), config=config
    )

    # evaluator, ba_refiner, qa_matrix_builder all use reasoning_model; formatter uses formatter_model
    assert seen_models == [
        "deepseek-r1:7b",
        "deepseek-r1:7b",
        "deepseek-r1:7b",
        "qwen2.5:7b",
    ]


async def test_polished_spec_returned_as_dict_is_coerced_to_string(monkeypatch):
    """Live bug: a smaller model returned polished_spec as a JSON object
    ({"Overview": "...", "Type": "Database"}) instead of a markdown string,
    which crashed SessionStateResponse's pydantic validation downstream."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {
                "ambiguous": False,
                "questions": [],
                "polished_spec": {"Overview": "A feature description", "Type": "Database"},
            }
        if _is_qa_matrix_prompt(prompt):
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-dict-spec"
    config = _config(session_id)
    await graph.ainvoke(_initial_state(session_id, "Some requirements."), config=config)
    await _resume_evaluator(config)

    snapshot = await graph.aget_state(config)
    assert isinstance(snapshot.values["polished_spec"], str)
    assert "## Overview" in snapshot.values["polished_spec"]
    assert "A feature description" in snapshot.values["polished_spec"]


async def test_category_placeholder_string_is_coerced_to_valid_enum_value(monkeypatch):
    """Live bug: the model copied the prompt's own shape-example notation
    ("sunny_day|rainy_day|boundary|edge_case") verbatim as the actual value
    for every scenario, instead of picking one."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if _is_qa_matrix_prompt(prompt):
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day|rainy_day|boundary|edge_case",
                        "title": "Happy path",
                        "steps": [
                            {"step_number": 1, "action": "a", "expected_result": "r"}
                        ],
                        "status": "new|modified|broken|unchanged",
                        "included": True,
                    }
                ],
            }
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-category-placeholder"
    config = _config(session_id)
    await graph.ainvoke(_initial_state(session_id, "Some requirements."), config=config)
    await _resume_evaluator(config)

    snapshot = await graph.aget_state(config)
    matrix = snapshot.values["test_matrix"]
    assert matrix[0]["category"] == "sunny_day"
    assert matrix[0]["status"] == "new"


async def test_legacy_test_cases_are_forwarded_into_qa_prompt(monkeypatch):
    """Regression guard: Agent 2 must actually see the legacy suite for delta analysis."""
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if _is_qa_matrix_prompt(prompt):
            seen_prompts.append(prompt)
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [],
            }
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-legacy"
    config = _config(session_id)
    legacy = "TC-OLD-1: Login with valid credentials -> redirects to dashboard"

    await graph.ainvoke(
        _initial_state(session_id, "Login flow now requires MFA.", legacy_test_cases=legacy),
        config=config,
    )
    await _resume_evaluator(config)

    assert len(seen_prompts) == 1
    assert legacy in seen_prompts[0]


async def test_flow_b_qa_direct_still_processes_uploaded_images(monkeypatch, tmp_path):
    """Flow B (qa_direct) must still route through ingest_visual so uploaded
    screenshots aren't silently dropped, even though it skips the BA refiner
    entirely (start_session pre-populates polished_spec directly). Since
    ingest_visual_node now goes through vision_ocr.extract_text_from_screenshot
    (preprocess + fail-safe OCR fallback) rather than calling ollama_chat
    directly, the fake image path must be a real, readable image (preprocess_image
    opens it with cv2 before the mocked ollama_chat is ever reached), and
    ollama_chat must be patched on both nodes_module and vision_ocr - they're
    separate name bindings from the same `from ... import ollama_chat` shape,
    each resolved independently at call time within its own module."""
    vision_calls = {"count": 0}
    image_path = str(tmp_path / "screenshot.png")
    cv2.imwrite(image_path, np.full((400, 800, 3), 255, dtype=np.uint8))

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if images:
            vision_calls["count"] += 1
            return "Login form with username/password fields"
        if _is_qa_matrix_prompt(prompt):
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(vision_ocr, "ollama_chat", fake_ollama_chat)

    session_id = "test-flow-b-images"
    config = _config(session_id)
    initial_state = _initial_state(
        session_id,
        "",
        workflow_mode="qa_direct",
        polished_spec="Already-refined spec text",
        image_paths=[image_path],
    )

    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)

    assert vision_calls["count"] == 1
    assert "Login form" in snapshot.values["visual_context"]
    assert snapshot.next == ("checklist_signoff",)


async def test_flow_c_format_only_reaches_formatter_directly(monkeypatch):
    """Flow C (format_only) must skip the BA/QA agents and interrupts
    entirely, translating legacy_test_cases straight into the target format
    in a single pass."""
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        seen_prompts.append(prompt)
        return "Feature: translated\n  Scenario: existing case"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-flow-c"
    config = _config(session_id)
    initial_state = _initial_state(
        session_id,
        "",
        workflow_mode="format_only",
        legacy_test_cases="TC-1: Old login test case",
        output_format="bdd",
    )

    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)

    assert snapshot.next == ()
    assert snapshot.values.get("test_matrix", []) == []
    assert "Feature: translated" in snapshot.values["formatted_output"]
    assert len(seen_prompts) == 1
    assert "Translate/reformat" in seen_prompts[0]
    assert "TC-1: Old login test case" in seen_prompts[0]


async def test_flow_c_bdd_output_with_multiple_features_gets_merged_into_one(monkeypatch):
    """formatter_node's post-generation guardrail must collapse a model's
    multi-Feature bdd output into exactly one Feature:, even though this runs
    through the real graph (not just a direct unit test of the merge
    helper) - end-to-end proof the wiring in formatter_node actually fires."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        return (
            "Feature: Login\n\n"
            "  Scenario: Happy path\n"
            "    Given a valid login\n\n"
            "Feature: Login - Failures\n\n"
            "  Scenario: Bad password\n"
            "    Given an invalid password\n"
        )

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-flow-c-bdd-merge"
    config = _config(session_id)
    initial_state = _initial_state(
        session_id,
        "",
        workflow_mode="format_only",
        legacy_test_cases="TC-1: Old login test case",
        output_format="bdd",
    )

    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)

    output = snapshot.values["formatted_output"]
    assert output.count("Feature:") == 1
    assert "Scenario: Happy path" in output
    assert "Scenario: Bad password" in output


async def test_refine_only_stops_after_ba_refiner_resolves(monkeypatch):
    """refine_only shares the full path through requirement_evaluator and
    ba_refiner (including the clarification loop), then must stop at the
    polished spec instead of continuing into qa_matrix_builder."""
    qa_calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 80,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if _is_ba_refiner_prompt(prompt):
            return {
                "ambiguous": False,
                "questions": [],
                "polished_spec": "## Input Validation\nDone.",
            }
        if _is_qa_matrix_prompt(prompt):
            qa_calls["count"] += 1
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-refine-only"
    config = _config(session_id)

    await graph.ainvoke(
        _initial_state(session_id, "Some requirements.", workflow_mode="refine_only"),
        config=config,
    )
    await _resume_evaluator(config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["ambiguity_resolved"] is True
    assert snapshot.values["polished_spec"] == "## Input Validation\nDone."
    assert snapshot.values.get("test_matrix", []) == []
    assert snapshot.values.get("formatted_output") is None
    assert qa_calls["count"] == 0  # qa_matrix_builder must never run


async def test_refine_only_still_runs_the_ba_clarification_loop(monkeypatch):
    """refine_only isn't a shortcut around clarification - it just stops
    afterward instead of continuing to the QA matrix builder."""
    ba_calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if _is_evaluator_prompt(prompt):
            return {
                "readiness_score": 40,
                "evaluation_feedback": {"data_and_boundaries": ["Too vague"]},
                "recommended_clarification_rounds": 1,
            }
        if _is_ba_refiner_prompt(prompt):
            ba_calls["count"] += 1
            if ba_calls["count"] == 1:
                return {
                    "ambiguous": True,
                    "questions": ["What is the retry policy?"],
                    "polished_spec": "",
                }
            return {"ambiguous": False, "questions": [], "polished_spec": "Resolved spec"}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-refine-only-loop"
    config = _config(session_id)

    await graph.ainvoke(
        _initial_state(session_id, "Vague.", workflow_mode="refine_only"), config=config
    )
    await _resume_evaluator(config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("ba_clarification",)

    await graph.ainvoke(Command(resume=["Retries twice with backoff."]), config=config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert snapshot.values["polished_spec"] == "Resolved spec"
    assert ba_calls["count"] == 2


def test_route_entry_and_route_after_ingest_default_to_full_on_bad_workflow_mode():
    assert nodes_module.route_entry({}) == "ingest"
    assert nodes_module.route_entry({"workflow_mode": "bogus"}) == "ingest"
    assert nodes_module.route_entry({"workflow_mode": "format_only"}) == "translate"

    assert nodes_module.route_after_ingest({}) == "full"
    assert nodes_module.route_after_ingest({"workflow_mode": "bogus"}) == "full"
    assert nodes_module.route_after_ingest({"workflow_mode": "qa_direct"}) == "qa_direct"


def test_route_ambiguity_stops_only_for_refine_only():
    assert nodes_module.route_ambiguity({"ambiguity_resolved": False}) == "clarify"
    assert (
        nodes_module.route_ambiguity({"ambiguity_resolved": False, "workflow_mode": "refine_only"})
        == "clarify"
    )
    assert nodes_module.route_ambiguity({"ambiguity_resolved": True}) == "resolved"
    assert (
        nodes_module.route_ambiguity({"ambiguity_resolved": True, "workflow_mode": "full"})
        == "resolved"
    )
    assert (
        nodes_module.route_ambiguity({"ambiguity_resolved": True, "workflow_mode": "refine_only"})
        == "stop"
    )
