import csv
import io
import json

from app.graph import nodes as nodes_module
from app.services import export_serializers


def test_coerce_score_clamps_and_defaults():
    assert nodes_module._coerce_score(85) == 85
    assert nodes_module._coerce_score("72") == 72
    assert nodes_module._coerce_score(150) == 100
    assert nodes_module._coerce_score(-10) == 0
    assert nodes_module._coerce_score(None) == 50
    assert nodes_module._coerce_score("not a number") == 50


def test_coerce_string_list_normalizes_shape():
    assert nodes_module._coerce_string_list(["gap one", "gap two"]) == ["gap one", "gap two"]
    assert nodes_module._coerce_string_list("a single gap as a bare string") == [
        "a single gap as a bare string"
    ]
    assert nodes_module._coerce_string_list(None) == []
    assert nodes_module._coerce_string_list("") == []
    assert nodes_module._coerce_string_list([1, 2]) == ["1", "2"]


def test_coerce_quality_gate_feedback_accepts_partial_categorized_dict():
    result = nodes_module._coerce_quality_gate_feedback(
        {
            "data_and_boundaries": ["OTP must be 6 digits"],
            "network_and_resiliency": "Ledger timeout undefined",
        }
    )
    assert result == {
        "data_and_boundaries": ["OTP must be 6 digits"],
        "integration_and_async_behavior": [],
        "network_and_resiliency": ["Ledger timeout undefined"],
        "state_and_lifecycle": [],
    }


def test_coerce_quality_gate_feedback_falls_back_to_all_empty_on_wrong_shape():
    """A model that ignores the categorized shape and returns the old flat
    list (or a bare string) must not crash - and since there's no reliable
    way to guess which category a flat list's items belong to, the safe
    default is all-empty-per-category, matching _coerce_test_matrix's
    existing non-list fallback precedent."""
    empty = {
        "data_and_boundaries": [],
        "integration_and_async_behavior": [],
        "network_and_resiliency": [],
        "state_and_lifecycle": [],
    }
    assert nodes_module._coerce_quality_gate_feedback(["flat", "list", "of", "gaps"]) == empty
    assert nodes_module._coerce_quality_gate_feedback("a bare string") == empty
    assert nodes_module._coerce_quality_gate_feedback(None) == empty


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
        {"step_number": 1, "action": "do a", "data": "x", "result": "get a"},
        {"step_number": 2, "action": "do b", "data": "", "result": "get b"},
    ]
    steps = nodes_module._coerce_steps(raw)
    assert steps == raw


def test_coerce_steps_backward_compatible_with_old_expected_result_key():
    """A model that still returns the pre-rewrite "expected_result" key
    instead of "result" must be repaired, not dropped."""
    raw = [{"step_number": 1, "action": "do a", "expected_result": "get a"}]
    steps = nodes_module._coerce_steps(raw)
    assert steps == [{"step_number": 1, "action": "do a", "data": "", "result": "get a"}]


def test_coerce_steps_falls_back_when_missing_or_flat():
    assert nodes_module._coerce_steps(None, fallback_text="Do the thing") == [
        {"step_number": 1, "action": "Do the thing", "data": "", "result": ""}
    ]
    assert nodes_module._coerce_steps("Just do this one thing") == [
        {"step_number": 1, "action": "Just do this one thing", "data": "", "result": ""}
    ]
    assert nodes_module._coerce_steps([], fallback_text="") == []
    assert nodes_module._coerce_steps(None, fallback_text="") == []


def test_coerce_test_matrix_accepts_new_steps_shape():
    raw = [
        {
            "id": "TC-1",
            "category": "sunny_day",
            "title": "Happy path",
            "preconditions": "User is logged in",
            "priority": "High",
            "test_type": "Functional",
            "module_or_area_path": "Login",
            "steps": [{"step_number": 1, "action": "a", "data": "d", "result": "r"}],
            "status": "new",
            "included": True,
        }
    ]
    matrix = nodes_module._coerce_test_matrix(raw)
    assert len(matrix) == 1
    assert matrix[0]["steps"] == [{"step_number": 1, "action": "a", "data": "d", "result": "r"}]
    assert matrix[0]["preconditions"] == "User is logged in"
    assert matrix[0]["priority"] == "High"
    assert matrix[0]["test_type"] == "Functional"
    assert matrix[0]["module_or_area_path"] == "Login"


def test_coerce_test_matrix_defaults_new_fields_when_absent():
    """A model that ignores the enriched fields entirely must not crash -
    they default to empty strings rather than being omitted."""
    raw = [
        {
            "id": "TC-1",
            "category": "sunny_day",
            "title": "Happy path",
            "steps": [{"step_number": 1, "action": "a", "result": "r"}],
            "status": "new",
            "included": True,
        }
    ]
    matrix = nodes_module._coerce_test_matrix(raw)
    assert matrix[0]["preconditions"] == ""
    assert matrix[0]["priority"] == ""
    assert matrix[0]["test_type"] == ""
    assert matrix[0]["module_or_area_path"] == ""


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
        {"step_number": 1, "action": "Submit the form with valid data.", "data": "", "result": ""}
    ]


def test_merge_multiple_bdd_features_passes_through_single_feature():
    text = (
        "Feature: Account Transfer\n\n"
        "  Scenario: Happy path\n"
        "    Given a valid account\n"
    )
    assert nodes_module._merge_multiple_bdd_features(text) == text


def test_merge_multiple_bdd_features_collapses_duplicates_and_keeps_all_scenarios():
    text = (
        "Feature: Account Transfer\n\n"
        "  Scenario: Happy path\n"
        "    Given a valid account\n\n"
        "Feature: Account Transfer - Edge Cases\n\n"
        "  Scenario: Timeout while transferring\n"
        "    Given the ledger service is unreachable\n\n"
        "Feature: Account Transfer - Rejections\n\n"
        "  Scenario Outline: Over daily limit\n"
        "    Given a transfer over the daily limit\n"
    )
    result = nodes_module._merge_multiple_bdd_features(text)

    assert result.count("Feature:") == 1
    assert result.startswith("Feature: Account Transfer\n")
    assert "Scenario: Happy path" in result
    assert "Scenario: Timeout while transferring" in result
    assert "Scenario Outline: Over daily limit" in result
    assert "Given the ledger service is unreachable" in result
    assert "Given a transfer over the daily limit" in result


def test_merge_multiple_bdd_features_leaves_zero_feature_output_unchanged():
    """Out of scope for this fix - a missing Feature: entirely is a
    different failure mode than the one this guards against."""
    text = "  Scenario: Happy path\n    Given a valid account\n"
    assert nodes_module._merge_multiple_bdd_features(text) == text


_SAMPLE_MATRIX = [
    {
        "id": "TC-1",
        "category": "sunny_day",
        "title": 'Transfer, with "quotes" and\nnewline',
        "preconditions": "User has $10,000 balance",
        "priority": "High",
        "test_type": "Functional",
        "module_or_area_path": "Account Balance Transfer",
        "steps": [
            {"step_number": 1, "action": "Enter amount", "data": "$500", "result": 'Status "pending"'},
            {"step_number": 2, "action": "Wait for ledger", "data": "", "result": "completed"},
        ],
        "status": "new",
        "included": True,
    },
    {
        "id": "TC-2",
        "category": "boundary",
        "title": "Over daily limit is rejected",
        "preconditions": "",
        "priority": "Medium",
        "test_type": "Functional",
        "module_or_area_path": "Account Balance Transfer",
        "steps": [
            {"step_number": 1, "action": "Submit over-limit transfer", "data": "$9999", "result": "Rejected"},
        ],
        "status": "new",
        "included": True,
    },
]


def test_serialize_azure_devops_csv_matches_real_required_headers():
    """Headers verified against Microsoft's own Azure DevOps Test Plans
    bulk-import docs: ID, Work Item Type, Title, Test Step, Step Action,
    Step Expected, Area Path, Assigned To, State."""
    output = export_serializers.serialize_azure_devops_csv(_SAMPLE_MATRIX)
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == [
        "ID", "Work Item Type", "Title", "Test Step", "Step Action",
        "Step Expected", "Area Path", "Assigned To", "State",
    ]
    # 2 steps for TC-1 + 1 step for TC-2 = 3 data rows, no blanking (ADO
    # repeats case-level columns on every step row, unlike qTest/TestRail).
    assert len(rows) == 1 + 3
    assert rows[1][1] == "Test Case"
    assert rows[1][2] == 'Transfer, with "quotes" and\nnewline'
    assert rows[2][6] == "Account Balance Transfer"


def test_serialize_qtest_csv_groups_metadata_on_first_step_row_only():
    output = export_serializers.serialize_qtest_csv(_SAMPLE_MATRIX)
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == [
        "Module", "Precondition", "Type", "Priority", "Step",
        "Step Description", "Expected Result",
    ]
    assert rows[1][:4] == ["Account Balance Transfer", "User has $10,000 balance", "Functional", "High"]
    assert rows[2][:4] == ["", "", "", ""]  # blanked continuation row for TC-1's 2nd step
    assert rows[2][4] == "2"
    assert rows[3][:4] == ["Account Balance Transfer", "", "Functional", "Medium"]  # TC-2's first row


def test_serialize_testrail_csv_groups_metadata_on_first_step_row_only():
    output = export_serializers.serialize_testrail_csv(_SAMPLE_MATRIX)
    rows = list(csv.reader(io.StringIO(output)))
    assert rows[0] == ["Title", "Type", "Priority", "Preconditions", "Step Action", "Step Expected"]
    assert rows[1][0] == 'Transfer, with "quotes" and\nnewline'
    assert rows[2][:4] == ["", "", "", ""]


def test_serialize_jira_xray_json_matches_real_xray_native_fields_shape():
    """Verified against Xray's Import Execution Results docs: steps are FLAT
    objects (action/data/result), no inner "fields" wrapper."""
    output = export_serializers.serialize_jira_xray_json(_SAMPLE_MATRIX, project_key="SPEC")
    parsed = json.loads(output)
    test = parsed["tests"][0]
    assert test["testInfo"]["projectKey"] == "SPEC"
    assert test["testInfo"]["testType"] == "Manual"
    assert test["testInfo"]["priority"] == "High"
    assert test["preconditions"] == "User has $10,000 balance"
    assert test["steps"][0] == {"action": "Enter amount", "data": "$500", "result": 'Status "pending"'}
    assert "fields" not in test["steps"][0]


def test_serializers_never_crash_on_missing_enriched_fields():
    """A test case missing the new fields entirely (e.g. hand-authored via
    the API before the fields existed) must not crash serialization."""
    minimal = [{"id": "TC-1", "category": "sunny_day", "title": "Bare case", "steps": [], "status": "new", "included": True}]

    # qTest's column set has no title/name column at all (matches its real
    # required headers) - only assert it doesn't crash and produces a row.
    qtest_output = export_serializers.serialize_qtest_csv(minimal)
    assert len(list(csv.reader(io.StringIO(qtest_output)))) == 2  # header + 1 data row

    for serializer in (
        export_serializers.serialize_azure_devops_csv,
        export_serializers.serialize_testrail_csv,
    ):
        output = serializer(minimal)
        assert "Bare case" in output

    output = export_serializers.serialize_jira_xray_json(minimal)
    assert json.loads(output)["tests"][0]["testInfo"]["summary"] == "Bare case"
