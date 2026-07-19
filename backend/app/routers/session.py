import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from langgraph.types import Command

from app.config import settings
from app.graph.build import graph
from app.models.schemas import (
    ChecklistSignoff,
    ClarificationAnswers,
    EvaluationDecision,
    RewindRequest,
    SessionStateResponse,
)
from app.services.file_parser import extract_text_from_csv, extract_text_from_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_AWAITING_BY_NEXT_NODE = {
    "evaluation_review": "requirement_evaluation",
    "ba_clarification": "ba_clarification",
    "gap_clarification": "gap_clarification",
    "checklist_signoff": "checklist_signoff",
}

VALID_WORKFLOW_MODES = {"full", "qa_direct", "format_only", "refine_only"}
VALID_OUTPUT_FORMATS = {"bdd", "testrail", "qtest", "jira_xray", "azure_devops"}


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def _to_response(session_id: str) -> SessionStateResponse:
    snapshot = await graph.aget_state(_config(session_id))
    values = snapshot.values
    awaiting = None
    for node_name in snapshot.next:
        if node_name in _AWAITING_BY_NEXT_NODE:
            awaiting = _AWAITING_BY_NEXT_NODE[node_name]
            break

    return SessionStateResponse(
        session_id=session_id,
        stage=values.get("stage", "start"),
        awaiting_input=awaiting,
        workflow_mode=values.get("workflow_mode"),
        workflow_aborted=bool(values.get("workflow_aborted", False)),
        out_of_scope_details=values.get("out_of_scope_details"),
        readiness_score=values.get("readiness_score"),
        evaluation_feedback=values.get("evaluation_feedback", {}),
        recommended_clarification_rounds=values.get("recommended_clarification_rounds"),
        ambiguity_questions=values.get("ambiguity_questions", [])
        if awaiting == "ba_clarification"
        else [],
        ambiguity_round=len(values.get("qa_history", [])) + 1,
        gap_questions=values.get("gap_questions", [])
        if awaiting == "gap_clarification"
        else [],
        gap_round=len(values.get("gap_qa_history", [])) + 1,
        polished_spec=values.get("polished_spec"),
        test_matrix=values.get("test_matrix", []),
        output_format=values.get("output_format"),
        formatted_output=values.get("formatted_output"),
        vision_model=values.get("vision_model") or settings.vision_model,
        reasoning_model=values.get("reasoning_model") or settings.reasoning_model,
        formatter_model=values.get("formatter_model") or settings.formatter_model,
    )


async def _save_upload(upload: UploadFile, upload_dir: str) -> str:
    # Prefix with a per-upload UUID so two files sharing a name (e.g.
    # clipboard-pasted screenshots both called "image.png") don't overwrite
    # each other on disk. This matters more for images than text/CSV: those
    # are read back immediately via _extract_text in the same request, but
    # image_paths are only read later at graph-execution time (ingest_visual_node),
    # by which point every upload in the batch has already been saved - a
    # collision would silently make two image_paths entries point at
    # whichever image was written last.
    dest = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{upload.filename}")
    content = await upload.read()
    with open(dest, "wb") as out:
        out.write(content)
    return dest


def _extract_text(dest: str, filename: str, content_type: str) -> str:
    lower_name = filename.lower()
    if content_type == "application/pdf" or lower_name.endswith(".pdf"):
        return extract_text_from_pdf(dest)
    if content_type == "text/csv" or lower_name.endswith(".csv"):
        return extract_text_from_csv(dest)
    with open(dest, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


@router.post("/", response_model=SessionStateResponse)
async def start_session(
    text: str = Form(""),
    legacy_test_cases: str = Form(""),
    files: list[UploadFile] = File(default=[]),
    legacy_files: list[UploadFile] = File(default=[]),
    vision_model: str = Form(""),
    reasoning_model: str = Form(""),
    formatter_model: str = Form(""),
    workflow_mode: str = Form("full"),
    out_of_scope_details: str = Form(""),
    output_format: str = Form(""),
):
    if workflow_mode not in VALID_WORKFLOW_MODES:
        raise HTTPException(
            400, f"workflow_mode must be one of {sorted(VALID_WORKFLOW_MODES)}."
        )
    if output_format and output_format not in VALID_OUTPUT_FORMATS:
        raise HTTPException(
            400, f"output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}."
        )

    session_id = str(uuid.uuid4())
    logger.info(
        "[%s] POST /api/sessions/: workflow_mode=%s text_chars=%d files=%d legacy_files=%d",
        session_id,
        workflow_mode,
        len(text),
        len(files),
        len(legacy_files),
    )
    text_parts = [text] if text else []
    legacy_parts = [legacy_test_cases] if legacy_test_cases else []
    image_paths: list[str] = []

    upload_dir = os.path.join(settings.storage_dir, session_id)
    os.makedirs(upload_dir, exist_ok=True)

    for upload in files:
        content_type = upload.content_type or ""
        dest = await _save_upload(upload, upload_dir)
        if content_type.startswith("image/"):
            image_paths.append(dest)
        else:
            text_parts.append(_extract_text(dest, upload.filename, content_type))

    for upload in legacy_files:
        content_type = upload.content_type or ""
        dest = await _save_upload(upload, upload_dir)
        legacy_parts.append(_extract_text(dest, upload.filename, content_type))

    if workflow_mode == "qa_direct":
        if not text_parts:
            raise HTTPException(
                400,
                "Provide the pre-refined requirements text or file for QA-direct mode.",
            )
    elif workflow_mode == "format_only":
        if not legacy_parts:
            raise HTTPException(
                400, "Provide existing test cases to translate for format-only mode."
            )
    elif not text_parts and not image_paths:
        raise HTTPException(400, "Provide at least some text or a file to analyze.")

    initial_state = {
        "session_id": session_id,
        "workflow_mode": workflow_mode,
        "out_of_scope_details": out_of_scope_details,
        "requirements_draft": "\n\n".join(text_parts),
        "image_paths": image_paths,
        "legacy_test_cases": "\n\n".join(legacy_parts),
        "qa_history": [],
        "gap_qa_history": [],
        "vision_model": vision_model or settings.vision_model,
        "reasoning_model": reasoning_model or settings.reasoning_model,
        "formatter_model": formatter_model or settings.formatter_model,
    }
    if workflow_mode == "qa_direct":
        # Flow B: the uploaded/pasted text IS the already-refined spec, so
        # feed it straight to qa_matrix_builder_node's polished_spec read
        # instead of running it through the BA refiner.
        initial_state["polished_spec"] = "\n\n".join(text_parts)
    if workflow_mode == "format_only":
        initial_state["output_format"] = output_format or "testrail"

    logger.info("[%s] running graph (this can take a while)...", session_id)
    await graph.ainvoke(initial_state, config=_config(session_id))
    response = await _to_response(session_id)
    logger.info("[%s] start_session: awaiting_input=%s", session_id, response.awaiting_input)
    return response


@router.get("/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    return await _to_response(session_id)


@router.post("/{session_id}/evaluation-decision", response_model=SessionStateResponse)
async def evaluation_decision(session_id: str, payload: EvaluationDecision):
    logger.info(
        "[%s] POST evaluation-decision: action=%s, max_clarification_rounds=%s, "
        "resuming graph...",
        session_id,
        payload.action,
        payload.max_clarification_rounds,
    )
    await graph.ainvoke(Command(resume=payload.model_dump()), config=_config(session_id))
    response = await _to_response(session_id)
    logger.info(
        "[%s] evaluation_decision: awaiting_input=%s, workflow_aborted=%s",
        session_id,
        response.awaiting_input,
        response.workflow_aborted,
    )
    return response


@router.post("/{session_id}/clarify-requirements", response_model=SessionStateResponse)
async def clarify_requirements(session_id: str, payload: ClarificationAnswers):
    logger.info(
        "[%s] POST clarify-requirements: %d answer(s), resuming graph...",
        session_id,
        len(payload.answers),
    )
    await graph.ainvoke(Command(resume=payload.answers), config=_config(session_id))
    response = await _to_response(session_id)
    logger.info(
        "[%s] clarify_requirements: awaiting_input=%s", session_id, response.awaiting_input
    )
    return response


@router.post("/{session_id}/clarify-gaps", response_model=SessionStateResponse)
async def clarify_gaps(session_id: str, payload: ClarificationAnswers):
    logger.info(
        "[%s] POST clarify-gaps: %d answer(s), resuming graph...",
        session_id,
        len(payload.answers),
    )
    await graph.ainvoke(Command(resume=payload.answers), config=_config(session_id))
    response = await _to_response(session_id)
    logger.info("[%s] clarify_gaps: awaiting_input=%s", session_id, response.awaiting_input)
    return response


@router.post("/{session_id}/checklist-signoff", response_model=SessionStateResponse)
async def checklist_signoff(session_id: str, payload: ChecklistSignoff):
    logger.info(
        "[%s] POST checklist-signoff: %d scenario(s), format=%s, resuming graph...",
        session_id,
        len(payload.test_matrix),
        payload.output_format,
    )
    resume_value = {
        "test_matrix": [item.model_dump() for item in payload.test_matrix],
        "output_format": payload.output_format,
    }
    await graph.ainvoke(Command(resume=resume_value), config=_config(session_id))
    response = await _to_response(session_id)
    logger.info("[%s] checklist_signoff: formatting complete", session_id)
    return response


@router.post("/{session_id}/rewind", response_model=SessionStateResponse)
async def rewind_session(session_id: str, payload: RewindRequest):
    """Go back to an earlier pause point and discard everything computed
    after it, so the user can resubmit with different input. Scans state
    history newest-first and forks from the FIRST (i.e. most recent) time
    the graph paused at `target` - for a multi-round clarification loop this
    means "redo my last answer", not an arbitrary earlier round, so no
    `round` parameter is needed. The fork itself
    (`aupdate_state(..., None, as_node="__copy__")`) doesn't touch any
    values, it just clones that checkpoint as the thread's new tip; the
    existing resume endpoints (already resuming against the bare thread
    config, not a pinned checkpoint_id) then pick it up and continue
    normally with no changes needed there. See
    backend/scripts/repro_time_travel.py for how this was validated against
    the real compiled graph before relying on it here."""
    config = _config(session_id)
    target_snapshot = None
    async for snapshot in graph.aget_state_history(config):
        if snapshot.next == (payload.target,):
            target_snapshot = snapshot
            break
    if target_snapshot is None:
        raise HTTPException(
            404, f"No '{payload.target}' step found in this session's history."
        )
    await graph.aupdate_state(target_snapshot.config, None, as_node="__copy__")
    response = await _to_response(session_id)
    logger.info(
        "[%s] rewind_session: target=%s, awaiting_input=%s",
        session_id,
        payload.target,
        response.awaiting_input,
    )
    return response


@router.get("/{session_id}/download")
async def download(session_id: str):
    logger.info("[%s] GET download", session_id)
    snapshot = await graph.aget_state(_config(session_id))
    output = snapshot.values.get("formatted_output")
    if not output:
        raise HTTPException(404, "No formatted output available yet for this session.")

    fmt = snapshot.values.get("output_format", "testrail")
    extension = {
        "bdd": "feature",
        "testrail": "csv",
        "qtest": "csv",
        "jira_xray": "json",
        "azure_devops": "csv",
    }.get(fmt, "txt")
    media_type = {
        "bdd": "text/plain",
        "testrail": "text/csv",
        "qtest": "text/csv",
        "jira_xray": "application/json",
        "azure_devops": "text/csv",
    }.get(fmt, "text/plain")

    content = output.encode("utf-8")
    if media_type == "text/csv":
        # Excel-based consumers (qTest's real .xlsx template, ADO's Excel add-in
        # path) are well known to misinterpret BOM-less UTF-8 CSVs as the
        # system codepage, corrupting non-ASCII characters/emoji/curly quotes.
        content = b"\xef\xbb\xbf" + content

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=specforge_export.{extension}"
        },
    )
