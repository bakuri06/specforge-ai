from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import SpecForgeState


def build_graph():
    workflow = StateGraph(SpecForgeState)

    workflow.add_node("ingest_visual", nodes.ingest_visual_node)
    workflow.add_node("requirement_evaluator", nodes.requirement_evaluator_node)
    workflow.add_node("evaluation_review", nodes.evaluation_review_node)
    workflow.add_node("ba_refiner", nodes.ba_refiner_node)
    workflow.add_node("ba_clarification", nodes.ba_clarification_node)
    workflow.add_node("qa_matrix_builder", nodes.qa_matrix_builder_node)
    workflow.add_node("gap_clarification", nodes.gap_clarification_node)
    workflow.add_node("checklist_signoff", nodes.checklist_signoff_node)
    workflow.add_node("formatter", nodes.formatter_node)

    # Multi-entry routing: "full", "refine_only", and "qa_direct" all ingest
    # first (so uploaded screenshots are never silently dropped), then
    # diverge; "format_only" skips straight to the formatter with no BA/QA
    # agents involved at all. "refine_only" shares "full"'s exact path
    # through the BA refiner and only diverges afterward (see
    # route_ambiguity's "stop" branch below) instead of needing its own
    # entry branch here.
    workflow.add_conditional_edges(
        START,
        nodes.route_entry,
        {"ingest": "ingest_visual", "translate": "formatter"},
    )
    workflow.add_conditional_edges(
        "ingest_visual",
        nodes.route_after_ingest,
        {"full": "requirement_evaluator", "qa_direct": "qa_matrix_builder"},
    )

    workflow.add_edge("requirement_evaluator", "evaluation_review")
    workflow.add_conditional_edges(
        "evaluation_review",
        nodes.route_after_evaluation,
        {"continue": "ba_refiner", "aborted": END},
    )

    workflow.add_conditional_edges(
        "ba_refiner",
        nodes.route_ambiguity,
        {"resolved": "qa_matrix_builder", "clarify": "ba_clarification", "stop": END},
    )
    workflow.add_edge("ba_clarification", "ba_refiner")

    workflow.add_conditional_edges(
        "qa_matrix_builder",
        nodes.route_gaps,
        {"resolved": "checklist_signoff", "clarify": "gap_clarification"},
    )
    workflow.add_edge("gap_clarification", "qa_matrix_builder")

    workflow.add_edge("checklist_signoff", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile(checkpointer=MemorySaver())


graph = build_graph()
