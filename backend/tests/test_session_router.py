import io
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.routers import session as session_module

client = TestClient(app)


class _FakeGraph:
    """Stands in for the real LangGraph graph so these tests don't need Ollama."""

    def __init__(self):
        self.state = None

    async def ainvoke(self, payload, config):
        if isinstance(payload, dict):
            self.state = payload
        return self.state

    async def aget_state(self, config):
        return SimpleNamespace(values=self.state or {}, next=())


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
