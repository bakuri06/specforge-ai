"""Minimal, standalone repro of langgraph.types.interrupt() with no app code.

Run with: python3 scripts/repro_interrupt.py  (from backend/, venv activated)

If this crashes with "RuntimeError: Called get_config outside of a runnable
context", the bug is in the installed langgraph/langchain-core combination
itself (or how this Python interacts with it), not in anything specific to
SpecForge AI's graph/nodes.
"""

import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class DemoState(TypedDict, total=False):
    answer: str


async def pause_node(state: DemoState) -> dict:
    answer = interrupt({"question": "continue?"})
    return {"answer": answer}


def build():
    g = StateGraph(DemoState)
    g.add_node("pause", pause_node)
    g.add_edge(START, "pause")
    g.add_edge("pause", END)
    return g.compile(checkpointer=MemorySaver())


async def main():
    graph = build()
    config = {"configurable": {"thread_id": "repro-1"}}

    print("Invoking graph, expecting it to pause at interrupt()...")
    await graph.ainvoke({"answer": ""}, config=config)

    snapshot = await graph.aget_state(config)
    print("snapshot.next:", snapshot.next)

    print("Resuming with Command(resume='yes')...")
    result = await graph.ainvoke(Command(resume="yes"), config=config)
    print("RESULT:", result)
    print("SUCCESS: interrupt()/resume works in this environment.")


if __name__ == "__main__":
    asyncio.run(main())
