"""LangGraph workflow assembly."""
from __future__ import annotations

from langgraph.graph import END, StateGraph
from workflow.state import GraphRAGState
from workflow import nodes
from workflow.edges import route_after_grading


def build_ingestion_workflow():
    graph = StateGraph(GraphRAGState)
    for name, fn in [("load_documents", nodes.load_documents), ("chunk", nodes.chunk_documents), ("embed", nodes.embed_documents), ("extract_graph", nodes.extract_graph), ("store_graph", nodes.store_graph), ("store_vectors", nodes.store_vectors)]:
        graph.add_node(name, fn)
    graph.set_entry_point("load_documents")
    graph.add_edge("load_documents", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "extract_graph")
    graph.add_edge("extract_graph", "store_graph")
    graph.add_edge("store_graph", "store_vectors")
    graph.add_edge("store_vectors", END)
    return graph.compile()


def build_qa_workflow(*_, **__):
    """Build the QA workflow.

    Accept and ignore optional keyword arguments for compatibility with the
    Streamlit UI, which passes runtime options such as settings/model/temperature.
    The current node implementations resolve their own configuration.
    """
    graph = StateGraph(GraphRAGState)
    for name, fn in [("receive_question", nodes.receive_question), ("rewrite_query", nodes.rewrite_query), ("retrieve_vector_context", nodes.retrieve_vector_context), ("retrieve_graph_context", nodes.retrieve_graph_context), ("merge_context", nodes.merge_context), ("grade_retrieved_context", nodes.grade_retrieved_context), ("generate_answer", nodes.generate_answer), ("evaluate_answer", nodes.evaluate_answer), ("return_answer", nodes.return_answer)]:
        graph.add_node(name, fn)
    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_vector_context")
    graph.add_edge("retrieve_vector_context", "retrieve_graph_context")
    graph.add_edge("retrieve_graph_context", "merge_context")
    graph.add_edge("merge_context", "grade_retrieved_context")
    graph.add_conditional_edges("grade_retrieved_context", route_after_grading)
    graph.add_edge("generate_answer", "evaluate_answer")
    graph.add_edge("evaluate_answer", "return_answer")
    graph.add_edge("return_answer", END)
    return graph.compile()
