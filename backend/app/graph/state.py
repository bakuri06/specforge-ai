from typing import Literal, TypedDict


class QAHistoryEntry(TypedDict):
    questions: list[str]
    answers: list[str]


class TestStepDict(TypedDict):
    step_number: int
    action: str
    expected_result: str


class TestMatrixItemDict(TypedDict):
    id: str
    category: Literal["sunny_day", "rainy_day", "boundary", "edge_case"]
    title: str
    steps: list[TestStepDict]
    status: Literal["new", "modified", "broken", "unchanged"]
    included: bool


class SpecForgeState(TypedDict, total=False):
    session_id: str

    # Multi-entry routing: which of the three user journeys this session is
    # running (see build.py's route_entry/route_after_ingest). Defaults to
    # "full" if absent - both routers treat any missing/unrecognized value
    # as "full" rather than raising, matching the rest of the codebase's
    # "never trust raw input at a decision point" convention.
    workflow_mode: Literal["full", "qa_direct", "format_only"]

    # Per-session model selection (falls back to app.config.settings defaults
    # when not provided at session start)
    vision_model: str
    reasoning_model: str
    formatter_model: str

    # Phase 1: ingestion
    requirements_draft: str
    image_paths: list[str]
    visual_context: str

    # Requirement evaluation gate (Flow A / "full" only, between ingestion
    # and the BA clarification loop)
    out_of_scope_details: str
    readiness_score: int
    evaluation_feedback: list[str]
    recommended_clarification_rounds: int
    max_clarification_rounds: int
    current_clarification_round: int
    workflow_aborted: bool

    # Phase 2: Agent 1 - BA Requirements Refiner
    qa_history: list[QAHistoryEntry]
    ambiguity_questions: list[str]
    ambiguity_resolved: bool
    polished_spec: str

    # Phase 3: Agent 2 - QA Test Matrix Builder
    legacy_test_cases: str
    gap_qa_history: list[QAHistoryEntry]
    gap_questions: list[str]
    gaps_resolved: bool
    test_matrix: list[TestMatrixItemDict]

    # Phase 4: Agent 3 - Formatter Router
    output_format: Literal["bdd", "testrail", "qtest", "jira_xray", "azure_devops"]
    formatted_output: str

    stage: str
