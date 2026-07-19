from app.graph import nodes as nodes_module


def test_coerce_score_clamps_and_defaults():
    assert nodes_module._coerce_score(85) == 85
    assert nodes_module._coerce_score("72") == 72
    assert nodes_module._coerce_score(150) == 100
    assert nodes_module._coerce_score(-10) == 0
    assert nodes_module._coerce_score(None) == 50
    assert nodes_module._coerce_score("not a number") == 50


def test_coerce_feedback_list_normalizes_shape():
    assert nodes_module._coerce_feedback_list(["gap one", "gap two"]) == ["gap one", "gap two"]
    assert nodes_module._coerce_feedback_list("a single gap as a bare string") == [
        "a single gap as a bare string"
    ]
    assert nodes_module._coerce_feedback_list(None) == []
    assert nodes_module._coerce_feedback_list("") == []
    assert nodes_module._coerce_feedback_list([1, 2]) == ["1", "2"]


def test_coerce_round_count_clamps_and_defaults():
    assert nodes_module._coerce_round_count(2) == 2
    assert nodes_module._coerce_round_count("3") == 3
    assert nodes_module._coerce_round_count(99) == 5
    assert nodes_module._coerce_round_count(-1) == 0
    assert nodes_module._coerce_round_count(None) == 1
    assert nodes_module._coerce_round_count(None, default=3) == 3
    assert nodes_module._coerce_round_count("garbage", default=2) == 2


def test_coerce_steps_handles_proper_list():
    raw = [
        {"step_number": 1, "action": "do a", "expected_result": "get a"},
        {"step_number": 2, "action": "do b", "expected_result": "get b"},
    ]
    steps = nodes_module._coerce_steps(raw)
    assert steps == raw


def test_coerce_steps_falls_back_when_missing_or_flat():
    assert nodes_module._coerce_steps(None, fallback_text="Do the thing") == [
        {"step_number": 1, "action": "Do the thing", "expected_result": ""}
    ]
    assert nodes_module._coerce_steps("Just do this one thing") == [
        {"step_number": 1, "action": "Just do this one thing", "expected_result": ""}
    ]
    assert nodes_module._coerce_steps([], fallback_text="") == []
    assert nodes_module._coerce_steps(None, fallback_text="") == []


def test_coerce_test_matrix_accepts_new_steps_shape():
    raw = [
        {
            "id": "TC-1",
            "category": "sunny_day",
            "title": "Happy path",
            "steps": [{"step_number": 1, "action": "a", "expected_result": "r"}],
            "status": "new",
            "included": True,
        }
    ]
    matrix = nodes_module._coerce_test_matrix(raw)
    assert len(matrix) == 1
    assert matrix[0]["steps"] == [{"step_number": 1, "action": "a", "expected_result": "r"}]


def test_coerce_test_matrix_backward_compatible_with_flat_description():
    """A model that ignores the new prompt and still returns the old flat
    'description' field must be repaired into a single-step list, not
    dropped or crashed on."""
    raw = [
        {
            "id": "TC-1",
            "category": "sunny_day",
            "title": "Happy path",
            "description": "Submit the form with valid data.",
            "status": "new",
            "included": True,
        }
    ]
    matrix = nodes_module._coerce_test_matrix(raw)
    assert len(matrix) == 1
    assert matrix[0]["steps"] == [
        {"step_number": 1, "action": "Submit the form with valid data.", "expected_result": ""}
    ]


async def test_validate_and_repair_json_passes_through_valid_json():
    valid = '{"issues": [{"fields": {"summary": "x"}}]}'
    result = await nodes_module._validate_and_repair_json(valid, "model", "prompt")
    assert '"summary": "x"' in result


async def test_validate_and_repair_json_retries_once_on_malformed_output(monkeypatch):
    calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        calls["count"] += 1
        return '{"issues": [{"fields": {"summary": "fixed"}}]}'

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    broken = "not json at all, no braces"
    result = await nodes_module._validate_and_repair_json(broken, "model", "prompt")

    assert calls["count"] == 1
    assert "fixed" in result


async def test_validate_and_repair_csv_passes_through_consistent_columns():
    valid = "a,b,c\n1,2,3\n4,5,6\n"
    result = await nodes_module._validate_and_repair_csv(valid, "model", "prompt")
    assert result == valid


async def test_validate_and_repair_csv_retries_once_on_inconsistent_columns(monkeypatch):
    calls = {"count": 0}

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        calls["count"] += 1
        return "a,b,c\n1,2,3\n4,5,6\n"

    monkeypatch.setattr(nodes_module, "ollama_chat", fake_ollama_chat)

    broken = "a,b,c\n1,2\n4,5,6,7\n"  # inconsistent column counts
    result = await nodes_module._validate_and_repair_csv(broken, "model", "prompt")

    assert calls["count"] == 1
    assert result == "a,b,c\n1,2,3\n4,5,6\n"
