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


def test_malformed_model_output_does_not_crash_response_construction(monkeypatch):
    """True end-to-end regression for the live crash: runs the REAL graph
    (not the FakeGraph used elsewhere in this file), since the failure was
    specifically pydantic rejecting SessionStateResponse(...) when the model
    returned polished_spec as a dict and category as a pipe-delimited
    placeholder string copied from the prompt's own shape example."""

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
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
                        "description": "d",
                        "status": "new|modified|broken|unchanged",
                        "included": True,
                    }
                ],
            }
        return "FORMATTED"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    response = client.post("/api/sessions/", data={"text": "Some requirements"})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["polished_spec"], str)
    assert body["test_matrix"][0]["category"] == "sunny_day"
    assert body["test_matrix"][0]["status"] == "new"


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
