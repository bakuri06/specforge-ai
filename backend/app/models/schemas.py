from typing import Literal, Optional

from pydantic import BaseModel


class TestMatrixItem(BaseModel):
    id: str
    category: Literal["sunny_day", "rainy_day", "boundary", "edge_case"]
    title: str
    description: str
    status: Literal["new", "modified", "broken", "unchanged"] = "new"
    included: bool = True


class SessionStateResponse(BaseModel):
    session_id: str
    stage: str
    awaiting_input: Optional[
        Literal["ba_clarification", "gap_clarification", "checklist_signoff"]
    ] = None
    ambiguity_questions: list[str] = []
    ambiguity_round: int = 1
    gap_questions: list[str] = []
    gap_round: int = 1
    polished_spec: Optional[str] = None
    test_matrix: list[TestMatrixItem] = []
    output_format: Optional[Literal["testrail", "qtest", "playwright"]] = None
    formatted_output: Optional[str] = None
    vision_model: Optional[str] = None
    reasoning_model: Optional[str] = None
    formatter_model: Optional[str] = None


class ClarificationAnswers(BaseModel):
    answers: list[str]


class ChecklistSignoff(BaseModel):
    test_matrix: list[TestMatrixItem]
    output_format: Literal["testrail", "qtest", "playwright"]
