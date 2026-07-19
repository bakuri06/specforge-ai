import io
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app.graph import nodes as nodes_module
from app.main import app
from app.routers import session as session_module

client = TestClient(app)


class _FakeGraph:
    """Stands in for the real LangGraph graph so these tests don't need Ollama."""

    def __init__(self):
        self.state = None
        self.next = ()

    async def ainvoke(self, payload, config):
        if isinstance(payload, dict):
            self.state = payload
        return self.state

    async def aget_state(self, config):
        return SimpleNamespace(values=self.state or {}, next=self.next)


def test_uploaded_legacy_csv_routes_to_legacy_test_cases_not_requirements(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    csv_content = b"Case ID,Title\nTC-1,Old happy path\n"
    response = client.post(
        "/api/sessions/",
        data={"text": "New requirements text"},
        files={"legacy_files": ("legacy.csv", io.BytesIO(csv_content), "text/csv")},
    )

    assert response.status_code == 200
    assert fake_graph.state["requirements_draft"] == "New requirements text"
    assert "Old happy path" in fake_graph.state["legacy_test_cases"]
    assert "Old happy path" not in fake_graph.state["requirements_draft"]


def test_uploaded_general_csv_still_routes_to_requirements(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    csv_content = b"Field,Value\nExport Format,CSV\n"
    response = client.post(
        "/api/sessions/",
        data={"text": "Base text"},
        files={"files": ("data.csv", io.BytesIO(csv_content), "text/csv")},
    )

    assert response.status_code == 200
    assert "Export Format" in fake_graph.state["requirements_draft"]
    assert fake_graph.state["legacy_test_cases"] == ""


def test_pasted_and_uploaded_legacy_cases_are_both_captured(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    csv_content = b"Case ID,Title\nTC-2,Uploaded case\n"
    response = client.post(
        "/api/sessions/",
        data={"text": "Requirements", "legacy_test_cases": "Pasted legacy case"},
        files={"legacy_files": ("legacy.csv", io.BytesIO(csv_content), "text/csv")},
    )

    assert response.status_code == 200
    legacy = fake_graph.state["legacy_test_cases"]
    assert "Pasted legacy case" in legacy
    assert "Uploaded case" in legacy


def test_qa_direct_rejects_blank_pre_refined_text(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"workflow_mode": "qa_direct", "text": ""},
    )

    assert response.status_code == 400
    assert fake_graph.state is None  # graph must never have been invoked


def test_format_only_rejects_missing_legacy_test_cases(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"workflow_mode": "format_only", "output_format": "bdd"},
    )

    assert response.status_code == 400
    assert fake_graph.state is None


def test_same_named_image_uploads_do_not_overwrite_each_other_on_disk(monkeypatch):
    """Two screenshots sharing a filename (common for clipboard-pasted
    images, e.g. both called "image.png") must not collide on disk. image_paths
    is only read later at graph-execution time (ingest_visual_node), by which
    point every upload in the batch has already been saved - a filename
    collision would silently make both image_paths entries point at whichever
    image was written last, so one screenshot's "analysis" would actually
    describe the other, wrong screenshot."""
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"text": "Some requirements"},
        files=[
            ("files", ("image.png", io.BytesIO(b"FIRST-IMAGE-BYTES"), "image/png")),
            ("files", ("image.png", io.BytesIO(b"SECOND-IMAGE-BYTES"), "image/png")),
        ],
    )

    assert response.status_code == 200
    image_paths = fake_graph.state["image_paths"]
    assert len(image_paths) == 2
    assert image_paths[0] != image_paths[1]
    with open(image_paths[0], "rb") as f:
        assert f.read() == b"FIRST-IMAGE-BYTES"
    with open(image_paths[1], "rb") as f:
        assert f.read() == b"SECOND-IMAGE-BYTES"


def test_start_session_rejects_invalid_workflow_mode(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"workflow_mode": "not-a-real-mode", "text": "Some text"},
    )

    assert response.status_code == 400
    assert fake_graph.state is None


def test_start_session_rejects_invalid_output_format(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"output_format": "not-a-real-format", "text": "Some text"},
    )

    assert response.status_code == 400
    assert fake_graph.state is None


def test_format_only_populates_output_format_in_initial_state(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={
            "workflow_mode": "format_only",
            "legacy_test_cases": "TC-1: an existing case",
            "output_format": "jira_xray",
        },
    )

    assert response.status_code == 200
    assert fake_graph.state["workflow_mode"] == "format_only"
    assert fake_graph.state["output_format"] == "jira_xray"
    assert fake_graph.state["legacy_test_cases"] == "TC-1: an existing case"


def test_qa_direct_populates_polished_spec_from_uploaded_text(monkeypatch):
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    response = client.post(
        "/api/sessions/",
        data={"workflow_mode": "qa_direct", "text": "Already-refined spec text"},
    )

    assert response.status_code == 200
    assert fake_graph.state["workflow_mode"] == "qa_direct"
    assert fake_graph.state["polished_spec"] == "Already-refined spec text"


def test_malformed_model_output_does_not_crash_response_construction(monkeypatch):
    """True end-to-end regression for the live crash: runs the REAL graph
    (not the FakeGraph used elsewhere in this file), since the failure was
    specifically pydantic rejecting SessionStateResponse(...) when the model
    returned polished_spec as a dict and category as a pipe-delimited
    placeholder string copied from the prompt's own shape example."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        # The evaluator's prompt also mentions "senior Business Analyst", so
        # it must be checked first via a marker unique to it.
        if "recommended_clarification_rounds" in prompt:
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if "senior Business Analyst" in prompt:
            return {
                "ambiguous": False,
                "questions": [],
                "polished_spec": {"Overview": "desc", "Type": "Database"},
            }
        if "senior QA Engineer" in prompt:
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

    start_response = client.post("/api/sessions/", data={"text": "Some requirements"})
    assert start_response.status_code == 200
    start_body = start_response.json()
    assert start_body["awaiting_input"] == "requirement_evaluation"

    response = client.post(
        f"/api/sessions/{start_body['session_id']}/evaluation-decision",
        json={"action": "proceed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["awaiting_input"] == "checklist_signoff"
    assert isinstance(body["polished_spec"], str)
    assert body["test_matrix"][0]["category"] == "sunny_day"
    assert body["test_matrix"][0]["status"] == "new"


def test_refine_only_flow_reaches_polished_spec_through_the_real_router(monkeypatch):
    """End-to-end regression for a real bug caught during manual smoke
    testing: SessionStateResponse.workflow_mode's Literal was never widened
    to include "refine_only" when the 4th workflow_mode was added to
    state.py/nodes.py/build.py, so pydantic raised a ValidationError on the
    very first response for any refine_only session - the graph-level tests
    in test_graph.py never catch this since they never construct a
    SessionStateResponse at all."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "recommended_clarification_rounds" in prompt:
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if "senior Business Analyst" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Refined spec"}
        return "UNUSED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    start_response = client.post(
        "/api/sessions/",
        data={"text": "Some requirements", "workflow_mode": "refine_only"},
    )
    assert start_response.status_code == 200
    start_body = start_response.json()
    assert start_body["workflow_mode"] == "refine_only"
    assert start_body["awaiting_input"] == "requirement_evaluation"

    response = client.post(
        f"/api/sessions/{start_body['session_id']}/evaluation-decision",
        json={"action": "proceed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["awaiting_input"] is None
    assert body["polished_spec"] == "Refined spec"
    assert body["test_matrix"] == []
    assert body["formatted_output"] is None


def test_rewind_to_ba_clarification_lets_user_resubmit_a_different_answer(monkeypatch):
    """Rewinding must actually discard downstream work and let a fresh answer
    drive a genuinely different result - not just flip awaiting_input back
    without rerunning anything."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "recommended_clarification_rounds" in prompt:
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 1,
            }
        if "Core Calculation Framework" in prompt:
            if "Retries twice" in prompt:
                return {"ambiguous": False, "questions": [], "polished_spec": "Spec: retries twice"}
            if "Retries three times" in prompt:
                return {
                    "ambiguous": False,
                    "questions": [],
                    "polished_spec": "Spec: retries three times",
                }
            return {
                "ambiguous": True,
                "questions": ["What is the retry policy?"],
                "polished_spec": "",
            }
        if "senior QA Engineer" in prompt:
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

    start = client.post("/api/sessions/", data={"text": "Some requirements"})
    session_id = start.json()["session_id"]
    client.post(f"/api/sessions/{session_id}/evaluation-decision", json={"action": "proceed"})

    original = client.post(
        f"/api/sessions/{session_id}/clarify-requirements",
        json={"answers": ["Retries twice with backoff."]},
    )
    assert original.json()["awaiting_input"] == "checklist_signoff"
    assert original.json()["polished_spec"] == "Spec: retries twice"

    rewound = client.post(
        f"/api/sessions/{session_id}/rewind", json={"target": "ba_clarification"}
    )
    assert rewound.status_code == 200
    assert rewound.json()["awaiting_input"] == "ba_clarification"

    resubmitted = client.post(
        f"/api/sessions/{session_id}/clarify-requirements",
        json={"answers": ["Retries three times with backoff."]},
    )
    assert resubmitted.json()["awaiting_input"] == "checklist_signoff"
    assert resubmitted.json()["polished_spec"] == "Spec: retries three times"
    assert len(resubmitted.json()["test_matrix"]) == 1


def test_rewind_to_evaluation_review_after_abort_allows_proceeding_again(monkeypatch):
    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "recommended_clarification_rounds" in prompt:
            return {
                "readiness_score": 90,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if "Core Calculation Framework" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Resolved spec"}
        return "UNUSED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    start = client.post(
        "/api/sessions/", data={"text": "Some reqs", "workflow_mode": "refine_only"}
    )
    session_id = start.json()["session_id"]

    aborted = client.post(
        f"/api/sessions/{session_id}/evaluation-decision", json={"action": "abort"}
    )
    assert aborted.json()["workflow_aborted"] is True
    assert aborted.json()["awaiting_input"] is None

    rewound = client.post(
        f"/api/sessions/{session_id}/rewind", json={"target": "evaluation_review"}
    )
    assert rewound.status_code == 200
    assert rewound.json()["workflow_aborted"] is False
    assert rewound.json()["awaiting_input"] == "requirement_evaluation"

    proceeded = client.post(
        f"/api/sessions/{session_id}/evaluation-decision", json={"action": "proceed"}
    )
    assert proceeded.json()["workflow_aborted"] is False
    assert proceeded.json()["polished_spec"] == "Resolved spec"


def test_rewind_to_unreached_target_returns_404(monkeypatch):
    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        if "recommended_clarification_rounds" in prompt:
            return {
                "readiness_score": 95,
                "evaluation_feedback": {},
                "recommended_clarification_rounds": 0,
            }
        if "Core Calculation Framework" in prompt:
            return {"ambiguous": False, "questions": [], "polished_spec": "Resolved spec"}
        return "UNUSED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    start = client.post(
        "/api/sessions/", data={"text": "Clean reqs", "workflow_mode": "refine_only"}
    )
    session_id = start.json()["session_id"]
    client.post(f"/api/sessions/{session_id}/evaluation-decision", json={"action": "proceed"})
    # Resolved immediately with 0 rounds -> no ba_clarification checkpoint ever existed.

    response = client.post(
        f"/api/sessions/{session_id}/rewind", json={"target": "ba_clarification"}
    )
    assert response.status_code == 404


def test_ollama_timeout_returns_helpful_504_not_bare_500(monkeypatch):
    class _TimeoutGraph(_FakeGraph):
        async def ainvoke(self, payload, config):
            raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(session_module, "graph", _TimeoutGraph())

    response = client.post(
        "/api/sessions/",
        data={"text": "Some requirements"},
    )

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"] or "too long" in response.json()["detail"]


def test_ollama_unreachable_returns_helpful_502_not_bare_500(monkeypatch):
    class _UnreachableGraph(_FakeGraph):
        async def ainvoke(self, payload, config):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(session_module, "graph", _UnreachableGraph())

    response = client.post(
        "/api/sessions/",
        data={"text": "Some requirements"},
    )

    assert response.status_code == 502
    assert "Ollama" in response.json()["detail"]


def test_ambiguity_round_increments_with_qa_history_and_questions_change(monkeypatch):
    """Regression test: the frontend was showing no visible change between
    clarification rounds because it couldn't tell a new round had started.
    ambiguity_round must increment, and the new questions must differ from
    round to round, so the UI has something to key a remount off of."""
    fake_graph = _FakeGraph()
    monkeypatch.setattr(session_module, "graph", fake_graph)

    fake_graph.state = {
        "stage": "ba_refiner",
        "ambiguity_questions": ["What is the SMS timeout?"],
        "qa_history": [],
    }
    fake_graph.next = ("ba_clarification",)
    first = client.get("/api/sessions/some-id").json()
    assert first["ambiguity_round"] == 1
    assert first["ambiguity_questions"] == ["What is the SMS timeout?"]

    fake_graph.state = {
        "stage": "ba_refiner",
        "ambiguity_questions": ["What happens if the phone number is missing?"],
        "qa_history": [{"questions": ["What is the SMS timeout?"], "answers": ["30s"]}],
    }
    second = client.get("/api/sessions/some-id").json()
    assert second["ambiguity_round"] == 2
    assert second["ambiguity_questions"] == ["What happens if the phone number is missing?"]
    assert second["ambiguity_questions"] != first["ambiguity_questions"]
