from typing import Literal, Optional

from pydantic import BaseModel

OutputFormat = Literal["bdd", "testrail", "qtest", "jira_xray", "azure_devops"]


class TestStep(BaseModel):
    step_number: int
    action: str
    expected_result: str


class TestMatrixItem(BaseModel):
    id: str
    category: Literal["sunny_day", "rainy_day", "boundary", "edge_case"]
    title: str
    steps: list[TestStep]
    status: Literal["new", "modified", "broken", "unchanged"] = "new"
    included: bool = True


class SessionStateResponse(BaseModel):
    session_id: str
    stage: str
    awaiting_input: Optional[
        Literal[
            "requirement_evaluation",
            "ba_clarification",
            "gap_clarification",
            "checklist_signoff",
        ]
    ] = None
    workflow_mode: Optional[Literal["full", "qa_direct", "format_only"]] = None
    workflow_aborted: bool = False
    out_of_scope_details: Optional[str] = None
    readiness_score: Optional[int] = None
    evaluation_feedback: list[str] = []
    recommended_clarification_rounds: Optional[int] = None
    ambiguity_questions: list[str] = []
    ambiguity_round: int = 1
    gap_questions: list[str] = []
    gap_round: int = 1
    polished_spec: Optional[str] = None
    test_matrix: list[TestMatrixItem] = []
    output_format: Optional[OutputFormat] = None
    formatted_output: Optional[str] = None
    vision_model: Optional[str] = None
    reasoning_model: Optional[str] = None
    formatter_model: Optional[str] = None


class ClarificationAnswers(BaseModel):
    answers: list[str]


class ChecklistSignoff(BaseModel):
    test_matrix: list[TestMatrixItem]
    output_format: OutputFormat


class EvaluationDecision(BaseModel):
    action: Literal["proceed", "abort"] = "proceed"
    max_clarification_rounds: Optional[int] = None
