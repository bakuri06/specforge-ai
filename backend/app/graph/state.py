from typing import Literal, TypedDict


class QAHistoryEntry(TypedDict):
    questions: list[str]
    answers: list[str]


class TestMatrixItemDict(TypedDict):
    id: str
    category: Literal["sunny_day", "rainy_day", "boundary", "edge_case"]
    title: str
    description: str
    status: Literal["new", "modified", "broken", "unchanged"]
    included: bool


class SpecForgeState(TypedDict, total=False):
    session_id: str

    # Phase 1: ingestion
    requirements_draft: str
    image_paths: list[str]
    visual_context: str

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
    output_format: Literal["testrail", "qtest", "playwright"]
    formatted_output: str

    stage: str
