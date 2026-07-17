from langgraph.types import Command

from app.graph import nodes as nodes_module
from app.graph.build import graph


def _initial_state(session_id: str, requirements: str, legacy_test_cases: str = "") -> dict:
    return {
        "session_id": session_id,
        "requirements_draft": requirements,
        "image_paths": [],
        "legacy_test_cases": legacy_test_cases,
        "qa_history": [],
        "gap_qa_history": [],
    }


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def test_ambiguity_and_gap_clarification_loops(monkeypatch):
    """Both agents ask a clarifying question once, then resolve on the next pass."""
    ba_calls = {"count": 0}
    qa_calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            ba_calls["count"] += 1
            if ba_calls["count"] == 1:
                return {
                    "ambiguous": True,
                    "questions": ["What happens on request timeout?", "How long is data retained?"],
                    "polished_spec": "",
                }
            return {"ambiguous": False, "questions": [], "polished_spec": "# Polished Spec"}

        if "senior QA Engineer" in prompt:
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
                        "description": "Submit the form with valid data.",
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
            "description": "Submit the same form twice at once.",
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
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Clean spec"}
        if "senior QA Engineer" in prompt:
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day",
                        "title": "Happy path",
                        "description": "d",
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


async def test_ba_refiner_forces_resolution_after_max_rounds(monkeypatch):
    """If the model keeps finding new ambiguity forever, the app must not get
    stuck in an infinite clarification loop — it must force a resolution once
    MAX_CLARIFICATION_ROUNDS rounds have been used, regardless of what the
    model itself reports."""
    ba_calls = {"count": 0}
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            ba_calls["count"] += 1
            seen_prompts.append(prompt)
            return {
                "ambiguous": True,
                "questions": [f"Question round {ba_calls['count']}"],
                "polished_spec": "",
            }
        if "senior QA Engineer" in prompt:
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-ba-cap"
    config = _config(session_id)

    await graph.ainvoke(_initial_state(session_id, "Vague requirements."), config=config)

    for _ in range(nodes_module.MAX_CLARIFICATION_ROUNDS):
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("ba_clarification",)
        await graph.ainvoke(Command(resume=["some answer"]), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("checklist_signoff",)
    assert snapshot.values["ambiguity_resolved"] is True
    assert ba_calls["count"] == nodes_module.MAX_CLARIFICATION_ROUNDS + 1
    assert "maximum number of clarification rounds" in seen_prompts[-1]
    # The mock never provides a polished_spec even when forced, so the
    # fallback synthesis (raw requirements + gathered Q&A) must kick in.
    assert "Vague requirements." in snapshot.values["polished_spec"]


async def test_qa_matrix_builder_forces_resolution_after_max_rounds(monkeypatch):
    """Same safety net as the BA loop, for Agent 2's gap-clarification loop."""
    qa_calls = {"count": 0}
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if "senior QA Engineer" in prompt:
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
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if "senior QA Engineer" in prompt:
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-model-override"
    config = _config(session_id)
    initial_state = _initial_state(session_id, "Some requirements.")
    initial_state["reasoning_model"] = "deepseek-r1:7b"
    initial_state["formatter_model"] = "qwen2.5:7b"

    await graph.ainvoke(initial_state, config=config)
    await graph.ainvoke(
        Command(resume={"test_matrix": [], "output_format": "testrail"}), config=config
    )

    assert seen_models == ["deepseek-r1:7b", "deepseek-r1:7b", "qwen2.5:7b"]


async def test_polished_spec_returned_as_dict_is_coerced_to_string(monkeypatch):
    """Live bug: a smaller model returned polished_spec as a JSON object
    ({"Overview": "...", "Type": "Database"}) instead of a markdown string,
    which crashed SessionStateResponse's pydantic validation downstream."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            return {
                "ambiguous": False,
                "questions": [],
                "polished_spec": {"Overview": "A feature description", "Type": "Database"},
            }
        if "senior QA Engineer" in prompt:
            return {"gaps_found": False, "questions": [], "test_matrix": []}
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    session_id = "test-dict-spec"
    config = _config(session_id)
    await graph.ainvoke(_initial_state(session_id, "Some requirements."), config=config)

    snapshot = await graph.aget_state(config)
    assert isinstance(snapshot.values["polished_spec"], str)
    assert "## Overview" in snapshot.values["polished_spec"]
    assert "A feature description" in snapshot.values["polished_spec"]


async def test_category_placeholder_string_is_coerced_to_valid_enum_value(monkeypatch):
    """Live bug: the model copied the prompt's own shape-example notation
    ("sunny_day|rainy_day|boundary|edge_case") verbatim as the actual value
    for every scenario, instead of picking one."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if "senior QA Engineer" in prompt:
            return {
                "gaps_found": False,
                "questions": [],
                "test_matrix": [
                    {
                        "id": "TC-1",
                        "category": "sunny_day|rainy_day|boundary|edge_case",
                        "title": "Happy path",
                        "description": "d",
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

    snapshot = await graph.aget_state(config)
    matrix = snapshot.values["test_matrix"]
    assert matrix[0]["category"] == "sunny_day"
    assert matrix[0]["status"] == "new"


async def test_legacy_test_cases_are_forwarded_into_qa_prompt(monkeypatch):
    """Regression guard: Agent 2 must actually see the legacy suite for delta analysis."""
    seen_prompts = []

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Spec"}
        if "senior QA Engineer" in prompt:
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

    assert len(seen_prompts) == 1
    assert legacy in seen_prompts[0]
