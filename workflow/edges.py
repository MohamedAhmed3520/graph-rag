"""Conditional edge helpers for LangGraph."""
from workflow.state import GraphRAGState


def route_after_grading(state: GraphRAGState) -> str:
    """Route based on retrieval quality."""
    grade = state.get("grade", "").lower()
    if "insufficient" in grade or "no" == grade.strip():
        return "generate_answer"
    return "generate_answer"
