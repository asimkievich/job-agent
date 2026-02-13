# job_graph.py
from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any, List, Callable

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class JobState(TypedDict, total=False):
    href: str
    normalized_href: str
    title: str
    published_at: Optional[str]
    run_id: str
    run_tags: List[str]
    mode: str

    body_text: str
    profile: Dict[str, Any]
    project_id: Optional[str]
    stable_job_id: str

    work_item_path: str
    profile_path: str

    scoring: Dict[str, Any]
    decision: str
    final_lifecycle: str

    error: Optional[str]


def build_job_graph(nodes: Dict[str, Callable]):
    g = StateGraph(JobState)

    g.add_node("capture", nodes["capture"])                  # async
    g.add_node("extract_profile", nodes["extract_profile"])  # sync
    g.add_node("resolve_ids", nodes["resolve_ids"])          # sync
    g.add_node("persist", nodes["persist"])                  # sync
    g.add_node("score", nodes["score"])                      # sync
    g.add_node("decide", nodes["decide"])                    # sync
    g.add_node("write", nodes["write"])                      # sync

    g.add_edge(START, "capture")
    g.add_edge("capture", "extract_profile")
    g.add_edge("extract_profile", "resolve_ids")
    g.add_edge("resolve_ids", "persist")
    g.add_edge("persist", "score")
    g.add_edge("score", "decide")
    g.add_edge("decide", "write")
    g.add_edge("write", END)

    return g.compile(checkpointer=InMemorySaver())
