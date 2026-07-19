"""Diagnostic script (not shipped/used by the app) validating LangGraph's
checkpoint time-travel mechanics before building the "go back and edit a
previous step" feature on top of it.

Findings (see the "Rewinding to a previous step" section this feature adds
to CLAUDE.md for the full writeup):
- `graph.aget_state_history(config)` yields StateSnapshots newest-first; each
  snapshot's `.config` carries an explicit `checkpoint_id` you can target.
- `graph.aget_state(config)` with an EXPLICIT checkpoint_id always returns
  that historical snapshot, frozen - it never reflects work done after it.
  Only a config with just `thread_id` (no checkpoint_id) resolves to the
  thread's actual current/latest state.
- To rewind an interrupt-based pause point (evaluation_review,
  ba_clarification, gap_clarification, checklist_signoff) without touching
  any of its already-computed values, fork with
  `graph.aupdate_state(target.config, None, as_node="__copy__")` - this
  clones that checkpoint as the thread's new tip, discarding whatever
  happened after it, with zero risk of accidentally mutating state.
- After that fork, all continuation calls (`Command(resume=...)`) MUST use
  the bare thread-level config (just `{"configurable": {"thread_id": ...}}`,
  no checkpoint_id) - reusing the explicit historical checkpoint_id for the
  next call just re-runs from that same frozen point again instead of
  advancing, since an explicit checkpoint_id always pins to that exact spot.

Run with: python scripts/repro_time_travel.py
"""
import asyncio

from app.graph import nodes as nodes_module
from app.graph.build import graph
from langgraph.types import Command

calls = []


async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
    if "recommended_clarification_rounds" in prompt:
        calls.append("EVALUATOR")
        return {
            "readiness_score": 90,
            "evaluation_feedback": [],
            "recommended_clarification_rounds": 1,
        }
    if "Core Calculation Framework" in prompt:
        calls.append("BA_REFINER")
        return {"ambiguous": False, "questions": [], "polished_spec": "Resolved spec"}
    calls.append("OTHER")
    return "FORMATTED"


async def main():
    nodes_module.ollama_chat = fake_ollama_chat
    config = {"configurable": {"thread_id": "repro-rewind"}}
    initial_state = {
        "session_id": "repro-rewind",
        "workflow_mode": "refine_only",
        "requirements_draft": "reqs",
        "image_paths": [],
        "legacy_test_cases": "",
        "qa_history": [],
        "gap_qa_history": [],
    }
    await graph.ainvoke(initial_state, config=config)
    # First pass: proceed with 0 clarification rounds -> resolves immediately.
    await graph.ainvoke(
        Command(resume={"action": "proceed", "max_clarification_rounds": 0}), config=config
    )
    snap = await graph.aget_state(config)
    print(
        "ORIGINAL outcome: next=",
        snap.next,
        "max_rounds=",
        snap.values.get("max_clarification_rounds"),
    )
    calls.clear()

    # Rewind to the evaluation_review pause point and choose 1 round instead.
    history = [s async for s in graph.aget_state_history(config)]
    target = next(s for s in history if s.next == ("evaluation_review",))

    await graph.aupdate_state(target.config, None, as_node="__copy__")
    bare_config = {"configurable": {"thread_id": "repro-rewind"}}
    await graph.ainvoke(
        Command(resume={"action": "proceed", "max_clarification_rounds": 1}), config=bare_config
    )
    snap = await graph.aget_state(bare_config)
    print(
        "AFTER REWIND: next=",
        snap.next,
        "max_rounds=",
        snap.values.get("max_clarification_rounds"),
        "calls=",
        calls,
    )


asyncio.run(main())
