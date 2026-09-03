"""LangGraph workflow assembly."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from workflow.state import GraphRAGState
from workflow import nodes


def build_ingestion_workflow():
    graph = StateGraph(
        GraphRAGState
    )

    graph.add_node(
        "load_documents",
        nodes.load_documents,
    )

    graph.add_node(
        "chunk",
        nodes.chunk_documents,
    )

    graph.add_node(
        "store_vectors",
        nodes.store_vectors,
    )

    graph.add_node(
        "extract_graph",
        nodes.extract_graph,
    )

    graph.add_node(
        "store_graph",
        nodes.store_graph,
    )

    graph.set_entry_point(
        "load_documents"
    )

    graph.add_edge(
        "load_documents",
        "chunk",
    )

    graph.add_edge(
        "chunk",
        "store_vectors",
    )

    graph.add_edge(
        "store_vectors",
        "extract_graph",
    )

    graph.add_edge(
        "extract_graph",
        "store_graph",
    )

    graph.add_edge(
        "store_graph",
        END,
    )

    return graph.compile()


def build_qa_workflow(
    settings=None,
    model=None,
    temperature=None,
):
    graph = StateGraph(
        GraphRAGState
    )

    graph.add_node(
        "receive_question",
        nodes.receive_question,
    )

    graph.add_node(
        "rewrite_query",
        nodes.rewrite_query,
    )

    graph.add_node(
        "retrieve_vector_context",
        nodes.retrieve_vector_context,
    )

    graph.add_node(
        "retrieve_graph_context",
        nodes.retrieve_graph_context,
    )

    graph.add_node(
        "merge_context",
        nodes.merge_context,
    )

    graph.add_node(
        "grade_retrieved_context",
        nodes.grade_retrieved_context,
    )

    graph.add_node(
        "generate_answer",
        nodes.generate_answer,
    )

    graph.add_node(
        "evaluate_answer",
        nodes.evaluate_answer,
    )

    graph.add_node(
        "return_answer",
        nodes.return_answer,
    )

    graph.set_entry_point(
        "receive_question"
    )

    graph.add_edge(
        "receive_question",
        "rewrite_query",
    )

    graph.add_edge(
        "rewrite_query",
        "retrieve_vector_context",
    )

    graph.add_edge(
        "retrieve_vector_context",
        "retrieve_graph_context",
    )

    graph.add_edge(
        "retrieve_graph_context",
        "merge_context",
    )

    graph.add_edge(
        "merge_context",
        "grade_retrieved_context",
    )

    graph.add_edge(
        "grade_retrieved_context",
        "generate_answer",
    )

    graph.add_edge(
        "generate_answer",
        "evaluate_answer",
    )

    graph.add_edge(
        "evaluate_answer",
        "return_answer",
    )

    graph.add_edge(
        "return_answer",
        END,
    )

    return graph.compile()
