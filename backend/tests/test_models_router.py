from fastapi.testclient import TestClient

import app.routers.models as models_module
from app.main import app

client = TestClient(app)


class _FakeOllamaResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeOllamaClient:
    def __init__(self, data):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, path):
        return _FakeOllamaResponse(self._data)


def test_list_models_returns_sorted_names_and_defaults(monkeypatch):
    fake_data = {"models": [{"name": "qwen2.5:7b"}, {"name": "deepseek-r1:7b"}]}

    monkeypatch.setattr(
        models_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _FakeOllamaClient(fake_data),
    )

    response = client.get("/api/models")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == ["deepseek-r1:7b", "qwen2.5:7b"]
    assert body["defaults"]["reasoning_model"] == "deepseek-r1:7b"
    assert body["defaults"]["formatter_model"] == "qwen2.5:7b"
    assert body["defaults"]["vision_model"] == "qwen2.5vl:7b"
