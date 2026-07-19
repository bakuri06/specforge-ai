"""Deterministic per-platform export serializers.

These are pure functions over an already-validated test matrix (the shape
produced by app.graph.nodes._coerce_test_matrix) - no LLM involved. This is
the fix for the export-compatibility audit's core finding: asking a local
LLM to free-text-generate CSV/JSON directly produced files that didn't match
any of the target platforms' real import contracts (wrong headers, wrong
envelope shape, fabricated field values). Every format-specific layout
concern lives here now; app/graph/prompts.py's FORMATTER_FORMAT_RULES only
still describes bdd, since Gherkin generation is a genuine text-generation
task, not a structured-data mapping one.

Column sets below were verified against each platform's official current
import documentation (see the export-compatibility audit): Azure DevOps
Test Plans' bulk CSV import, Jira/Xray's Import Execution Results JSON
"native fields" step shape, and reasonable approximations of qTest's/
TestRail's real templates for the two platforms whose real bulk importer
doesn't accept plain CSV/Markdown at all (qTest requires .xlsx; TestRail
also accepts XML/API) - CSV here is the deliberately-chosen best-effort
intermediate format for those two, not a claim of exact native compatibility.

The `data` per-step field (Xray's real "data" native field) is intentionally
not emitted by the three CSV serializers below - none of Azure DevOps',
qTest's, or TestRail's real column sets researched in the audit include a
distinct test-data column, so inventing one here would just be the same
"the model added values the platform doesn't expect" mistake this rewrite
exists to fix. The field stays captured in the internal model and editable
in ChecklistEditor.jsx for the one export (Jira/Xray) that actually uses it.
"""

import csv
import io
import json
from typing import Callable


def _grouped_csv(
    headers: list[str],
    matrix: list[dict],
    case_columns: Callable[[dict], list[str]],
    step_columns: Callable[[dict], list[str]],
) -> str:
    """Shared row-grouping helper for the three CSV formats: a case's own
    columns are populated on its first step row and left blank on every
    subsequent step row of that same case, so rows visually group under one
    test case on import - qTest's and TestRail's real templates use this
    convention; Azure DevOps' does not (see serialize_azure_devops_csv)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(headers)
    for item in matrix:
        case_values = case_columns(item)
        steps = item.get("steps") or [{}]
        for index, step in enumerate(steps):
            row_case_values = case_values if index == 0 else [""] * len(case_values)
            writer.writerow(row_case_values + step_columns(step))
    return buffer.getvalue()


def serialize_azure_devops_csv(matrix: list[dict]) -> str:
    """Azure DevOps Test Plans' native bulk-import CSV. Verified required
    headers (Microsoft's own docs): ID, Work Item Type, Title, Test Step,
    Step Action, Step Expected, Area Path, Assigned To, State - Area Path is
    mandatory for Azure DevOps Services import. Unlike qTest/TestRail, ADO's
    format repeats ID/Title (and here, every case-level column) on every
    step row rather than blanking continuation rows."""
    headers = [
        "ID",
        "Work Item Type",
        "Title",
        "Test Step",
        "Step Action",
        "Step Expected",
        "Area Path",
        "Assigned To",
        "State",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(headers)
    for item in matrix:
        steps = item.get("steps") or [{}]
        for step in steps:
            writer.writerow(
                [
                    item.get("id", ""),
                    "Test Case",
                    item.get("title", ""),
                    str(step.get("step_number", "")),
                    step.get("action", ""),
                    step.get("result", ""),
                    item.get("module_or_area_path", ""),
                    "",  # Assigned To - no owner field in the internal model
                    "Design",  # State - reasonable default for a newly authored case
                ]
            )
    return buffer.getvalue()


def serialize_qtest_csv(matrix: list[dict]) -> str:
    """qTest's test-case column set (Module/Precondition/Type/Priority/Step/
    Step Description/Expected Result), row-grouped: metadata on each case's
    first step row only. qTest's real bulk importer requires .xlsx, not
    plain CSV (see module docstring) - this remains a best-effort CSV export."""
    return _grouped_csv(
        headers=["Module", "Precondition", "Type", "Priority", "Step", "Step Description", "Expected Result"],
        matrix=matrix,
        case_columns=lambda item: [
            item.get("module_or_area_path", ""),
            item.get("preconditions", ""),
            item.get("test_type", ""),
            item.get("priority", ""),
        ],
        step_columns=lambda step: [
            str(step.get("step_number", "")),
            step.get("action", ""),
            step.get("result", ""),
        ],
    )


def serialize_testrail_csv(matrix: list[dict]) -> str:
    """TestRail's "Test Case (Steps)" CSV template column set, row-grouped
    the same way as qTest. TestRail's real importer also accepts XML/API
    (see module docstring); this covers the CSV path."""
    return _grouped_csv(
        headers=["Title", "Type", "Priority", "Preconditions", "Step Action", "Step Expected"],
        matrix=matrix,
        case_columns=lambda item: [
            item.get("title", ""),
            item.get("test_type", ""),
            item.get("priority", ""),
            item.get("preconditions", ""),
        ],
        step_columns=lambda step: [
            step.get("action", ""),
            step.get("result", ""),
        ],
    )


def serialize_jira_xray_json(matrix: list[dict], project_key: str = "SPEC") -> str:
    """Xray's Import Execution Results envelope shape, confirmed against
    Xray's own docs: testInfo carries case metadata (testType is a real
    required-in-practice field - Manual/Cucumber/Generic), and each step is
    a FLAT object with native fields action/data/result - no inner "fields"
    wrapper, contrary to an earlier, unverified claim about this schema."""
    tests = []
    for item in matrix:
        steps = [
            {
                "action": step.get("action", ""),
                "data": step.get("data", ""),
                "result": step.get("result", ""),
            }
            for step in (item.get("steps") or [])
        ]
        tests.append(
            {
                "testInfo": {
                    "projectKey": project_key,
                    "summary": item.get("title", ""),
                    "testType": "Manual",
                    "priority": item.get("priority", ""),
                    "labels": [item.get("category", "")] if item.get("category") else [],
                },
                "preconditions": item.get("preconditions", ""),
                "steps": steps,
            }
        )
    return json.dumps({"tests": tests}, indent=2)
