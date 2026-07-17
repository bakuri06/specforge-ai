from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import SpecForgeState


def build_graph():
    workflow = StateGraph(SpecForgeState)

    workflow.add_node("ingest_visual", nodes.ingest_visual_node)
    workflow.add_node("ba_refiner", nodes.ba_refiner_node)
    workflow.add_node("ba_clarification", nodes.ba_clarification_node)
    workflow.add_node("qa_matrix_builder", nodes.qa_matrix_builder_node)
    workflow.add_node("gap_clarification", nodes.gap_clarification_node)
    workflow.add_node("checklist_signoff", nodes.checklist_signoff_node)
    workflow.add_node("formatter", nodes.formatter_node)

    workflow.add_edge(START, "ingest_visual")
    workflow.add_edge("ingest_visual", "ba_refiner")

    workflow.add_conditional_edges(
        "ba_refiner",
        nodes.route_ambiguity,
        {"resolved": "qa_matrix_builder", "clarify": "ba_clarification"},
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
