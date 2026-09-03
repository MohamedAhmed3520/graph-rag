"""LangGraph nodes for ingestion and question answering."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config.models import create_llm
from config.settings import get_settings

from embeddings.vector_store import VectorStore
from ingestion.ingest import IngestionPipeline

from graph.extractor import GraphExtractor
from graph.builder import GraphBuilder
from graph.neo4j_store import Neo4jStore

from retrievers.vector_retriever import VectorRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.hybrid import HybridRetriever

from prompts.answer import ANSWER_PROMPT
from prompts.grading import GRADING_PROMPT
from prompts.rewrite import REWRITE_PROMPT

from workflow.state import GraphRAGState


def load_documents(
    state: GraphRAGState,
) -> GraphRAGState:

    pipeline = IngestionPipeline(
        get_settings()
    )

    chunks = pipeline.load(
        state.get("file_paths", [])
    )

    errors = list(
        state.get("errors", [])
    )

    if not chunks:
        errors.append(
            "No text chunks were produced. "
            "Check the uploaded document."
        )

    return {
        **state,
        "documents": chunks,
        "status": (
            f"Parsed into {len(chunks)} chunks"
        ),
        "errors": errors,
    }


def chunk_documents(
    state: GraphRAGState,
) -> GraphRAGState:

    chunks = IngestionPipeline(
        get_settings()
    ).split(
        state.get("documents", [])
    )

    return {
        **state,
        "chunks": chunks,
        "status": (
            f"Created {len(chunks)} chunks"
        ),
    }


def store_vectors(
    state: GraphRAGState,
) -> GraphRAGState:

    chunks = state.get("chunks", [])

    if not chunks:
        return {
            **state,
            "status": "No vectors stored",
        }

    try:
        store = VectorStore(
            get_settings()
        )

        store.add_chunks(chunks)

        stats = store.stats()

        return {
            **state,
            "status": (
                f"FAISS indexed "
                f"{stats['vectors']} vectors"
            ),
        }

    except Exception as exc:
        return {
            **state,
            "errors": [
                *state.get("errors", []),
                f"FAISS indexing failed: {exc}",
            ],
            "status": "FAISS indexing failed",
        }


def extract_graph(
    state: GraphRAGState,
) -> GraphRAGState:

    neo = Neo4jStore(
        get_settings()
    )

    if not neo.is_available():
        neo.close()

        return {
            **state,
            "graph_documents": [],
            "status": (
                "Neo4j unavailable; "
                "continuing with vector RAG"
            ),
        }

    try:
        extractor = GraphExtractor()

        docs = extractor.extract_documents(
            state.get("chunks", [])
        )

        return {
            **state,
            "graph_documents": docs,
            "status": (
                f"Extracted graph facts "
                f"from {len(docs)} chunks"
            ),
        }

    except Exception as exc:
        return {
            **state,
            "graph_documents": [],
            "errors": [
                *state.get("errors", []),
                f"Graph extraction failed: {exc}",
            ],
            "status": (
                "Graph extraction failed; "
                "vector RAG remains available"
            ),
        }

    finally:
        neo.close()


def store_graph(
    state: GraphRAGState,
) -> GraphRAGState:

    docs = state.get(
        "graph_documents",
        [],
    )

    if not docs:
        return state

    store = Neo4jStore(
        get_settings()
    )

    if not store.is_available():
        store.close()

        return {
            **state,
            "status": (
                "Neo4j unavailable; "
                "graph not stored"
            ),
        }

    try:
        counts = GraphBuilder(
            store=store
        ).build(docs)

        return {
            **state,
            "status": (
                f"Neo4j stored "
                f"{counts['entities']} entities / "
                f"{counts['relationships']} relationships"
            ),
        }

    finally:
        store.close()


def receive_question(
    state: GraphRAGState,
) -> GraphRAGState:

    return {
        **state,
        "status": "Question received",
    }


def rewrite_query(
    state: GraphRAGState,
) -> GraphRAGState:

    question = (
        state.get("question", "")
        .strip()
    )

    if not question:
        return {
            **state,
            "rewritten_question": "",
            "status": "Empty question",
        }

    # Rewrite is optional. Retrieval can always use
    # the original question if the LLM fails.
    try:
        llm = create_llm()

        response = llm.invoke(
            [
                SystemMessage(
                    content=REWRITE_PROMPT
                ),
                HumanMessage(
                    content=question
                ),
            ]
        )

        rewritten = str(
            response.content
        ).strip()

        if not rewritten:
            rewritten = question

        return {
            **state,
            "rewritten_question": rewritten,
            "status": "Query rewritten",
        }

    except Exception:
        return {
            **state,
            "rewritten_question": question,
            "status": (
                "Query rewrite failed; "
                "using original question"
            ),
        }


def retrieve_vector_context(
    state: GraphRAGState,
) -> GraphRAGState:

    question = (
        state.get("question", "")
        .strip()
    )

    rewritten = (
        state.get("rewritten_question", "")
        .strip()
    )

    queries = []

    if rewritten:
        queries.append(rewritten)

    if question and question not in queries:
        queries.append(question)

    retriever = VectorRetriever(
        settings=get_settings()
    )

    errors = list(
        state.get("errors", [])
    )

    for query in queries:

        try:
            hits = retriever.retrieve(
                query
            )

            if hits:
                return {
                    **state,
                    "vector_context": hits,
                    "status": (
                        f"Vector retrieval: "
                        f"{len(hits)} hits"
                    ),
                }

        except Exception as exc:
            errors.append(
                f"Vector retrieval failed: {exc}"
            )

    return {
        **state,
        "vector_context": [],
        "errors": errors,
        "status": "Vector retrieval: 0 hits",
    }


def retrieve_graph_context(
    state: GraphRAGState,
) -> GraphRAGState:

    question = (
        state.get("question", "")
        .strip()
    )

    rewritten = (
        state.get("rewritten_question", "")
        .strip()
    )

    query = (
        rewritten
        or question
    )

    try:
        hits = GraphRetriever(
            settings=get_settings()
        ).retrieve(
            query,
            top_k=get_settings().retrieval_top_k,
        )

        return {
            **state,
            "graph_context": hits,
            "status": (
                f"Graph retrieval: "
                f"{len(hits)} hits"
            ),
        }

    except Exception as exc:
        return {
            **state,
            "graph_context": [],
            "errors": [
                *state.get("errors", []),
                f"Graph retrieval failed: {exc}",
            ],
            "status": (
                "Graph retrieval unavailable"
            ),
        }


def merge_context(
    state: GraphRAGState,
) -> GraphRAGState:

    merged = HybridRetriever.format_context(
        state.get("vector_context", []),
        state.get("graph_context", []),
    )

    return {
        **state,
        "merged_context": merged,
        "status": "Context merged",
    }


def grade_retrieved_context(
    state: GraphRAGState,
) -> GraphRAGState:

    vector_hits = state.get(
        "vector_context",
        [],
    )

    graph_hits = state.get(
        "graph_context",
        [],
    )

    if not vector_hits and not graph_hits:
        return {
            **state,
            "grade": "insufficient",
            "status": "No evidence retrieved",
        }

    try:
        content = GRADING_PROMPT.format(
            question=state.get(
                "question",
                "",
            ),
            context=state.get(
                "merged_context",
                "",
            ),
        )

        raw = str(
            create_llm()
            .invoke(
                [
                    HumanMessage(
                        content=content
                    )
                ]
            )
            .content
        ).strip()

        cleaned = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            cleaned = cleaned[
                start : end + 1
            ]

        data = json.loads(cleaned)

        sufficient = False

        if isinstance(data, dict):
            value = data.get(
                "sufficient",
                False,
            )

            if isinstance(value, bool):
                sufficient = value

            elif isinstance(value, str):
                sufficient = (
                    value.lower()
                    in {
                        "true",
                        "yes",
                        "1",
                    }
                )

        grade = (
            "sufficient"
            if sufficient
            else "insufficient"
        )

    except Exception:
        # Retrieval evidence exists, so do not throw it
        # away just because the grading model failed.
        grade = (
            "sufficient"
            if vector_hits or graph_hits
            else "insufficient"
        )

    return {
        **state,
        "grade": grade,
        "status": (
            f"Context graded: {grade}"
        ),
    }


def generate_answer(
    state: GraphRAGState,
) -> GraphRAGState:

    vector_hits = state.get(
        "vector_context",
        [],
    )

    graph_hits = state.get(
        "graph_context",
        [],
    )

    if not vector_hits and not graph_hits:
        return {
            **state,
            "answer": (
                "I don't know based on the "
                "provided sources. "
                "No relevant evidence was retrieved."
            ),
            "citations": [],
            "status": (
                "No evidence; answer withheld"
            ),
        }

    content = ANSWER_PROMPT.format(
        question=state.get(
            "question",
            "",
        ),
        context=state.get(
            "merged_context",
            "",
        ),
    )

    answer = str(
        create_llm()
        .invoke(
            [
                HumanMessage(
                    content=content
                )
            ]
        )
        .content
    ).strip()

    citations: list[dict[str, Any]] = []

    for hit in vector_hits:
        metadata = hit.get(
            "metadata",
            {},
        )

        citations.append(
            {
                "source": metadata.get(
                    "source",
                    metadata.get(
                        "source_path",
                        "unknown",
                    ),
                ),
                "chunk_id": metadata.get(
                    "chunk_id",
                    hit.get("id", ""),
                ),
            }
        )

    for hit in graph_hits:
        chunk_id = hit.get(
            "chunk_id"
        )

        if chunk_id:
            citations.append(
                {
                    "source": "graph",
                    "chunk_id": chunk_id,
                }
            )

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "status": "Answer generated",
    }


def evaluate_answer(
    state: GraphRAGState,
) -> GraphRAGState:

    return {
        **state,
        "status": "Answer evaluated",
    }


def return_answer(
    state: GraphRAGState,
) -> GraphRAGState:

    return {
        **state,
        "status": "Complete",
    }
