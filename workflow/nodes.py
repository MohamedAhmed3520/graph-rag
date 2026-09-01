"""LangGraph node implementations."""
from __future__ import annotations

import json
from langchain_core.messages import HumanMessage, SystemMessage
from config.models import create_llm
from config.settings import get_settings
from prompts.answer import ANSWER_PROMPT
from prompts.grading import GRADING_PROMPT
from prompts.rewrite import REWRITE_PROMPT
from workflow.state import GraphRAGState
from ingestion.ingest import IngestionPipeline
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_store import VectorStore
from graph.extractor import GraphExtractor
from graph.builder import GraphBuilder
from retrievers.vector_retriever import VectorRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.hybrid import HybridRetriever


def load_documents(state: GraphRAGState) -> GraphRAGState:
    docs = IngestionPipeline().load(state.get("file_paths", []))
    return {**state, "documents": docs, "status": f"Loaded {len(docs)} documents"}


def chunk_documents(state: GraphRAGState) -> GraphRAGState:
    chunks = IngestionPipeline().split(state.get("documents", []))
    return {**state, "chunks": chunks, "status": f"Created {len(chunks)} chunks"}


def embed_documents(state: GraphRAGState) -> GraphRAGState:
    embedder = EmbeddingModel()
    vectors = embedder.embed_documents([c.text for c in state.get("chunks", [])])
    return {**state, "embeddings": vectors, "status": "Generated embeddings"}


def extract_graph(state: GraphRAGState) -> GraphRAGState:
    graph_docs = GraphExtractor().extract_documents(state.get("chunks", []))
    return {**state, "graph_documents": graph_docs, "status": "Extracted graph facts"}


def store_graph(state: GraphRAGState) -> GraphRAGState:
    GraphBuilder().build(state.get("graph_documents", []))
    return {**state, "status": "Stored graph in Neo4j"}


def store_vectors(state: GraphRAGState) -> GraphRAGState:
    chunks = state.get("chunks", [])
    if not chunks:
        return {**state, "status": "No chunks to store"}
    embedder = EmbeddingModel()
    store = VectorStore(settings=get_settings())
    store.add_chunks(chunks)
    store.persist()
    return {**state, "status": "Stored vectors in FAISS"}


def receive_question(state: GraphRAGState) -> GraphRAGState:
    return {**state, "status": "Question received"}


def rewrite_query(state: GraphRAGState) -> GraphRAGState:
    llm = create_llm()
    msg = llm.invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=state.get("question", ""))])
    return {**state, "rewritten_question": msg.content.strip(), "status": "Query rewritten"}


def retrieve_vector_context(state: GraphRAGState) -> GraphRAGState:
    hits = VectorRetriever().retrieve(state.get("rewritten_question") or state.get("question", ""))
    return {**state, "vector_context": hits, "status": "Vector context retrieved"}


def retrieve_graph_context(state: GraphRAGState) -> GraphRAGState:
    hits = GraphRetriever().retrieve(state.get("rewritten_question") or state.get("question", ""))
    return {**state, "graph_context": hits, "status": "Graph context retrieved"}


def merge_context(state: GraphRAGState) -> GraphRAGState:
    merged = HybridRetriever.format_context(state.get("vector_context", []), state.get("graph_context", []))
    return {**state, "merged_context": merged, "status": "Context merged"}


def grade_retrieved_context(state: GraphRAGState) -> GraphRAGState:
    """Grade whether retrieved context sufficiently answers the question.
    
    Parses JSON response from LLM and normalizes to "sufficient" or "insufficient".
    Handles malformed JSON gracefully.
    """
    llm = create_llm()
    content = GRADING_PROMPT.format(question=state.get("question", ""), context=state.get("merged_context", ""))
    response = llm.invoke([HumanMessage(content=content)]).content.strip()
    
    # Try to parse JSON response
    grade = "insufficient"  # default to insufficient
    try:
        # Try to extract JSON from code fences if present
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response
        
        data = json.loads(json_str)
        
        # Extract the "sufficient" field
        if isinstance(data, dict):
            sufficient = data.get("sufficient")
            # Convert to boolean if it's a string
            if isinstance(sufficient, str):
                sufficient = sufficient.lower() in ("true", "yes", "1")
            
            if sufficient is True:
                grade = "sufficient"
            else:
                grade = "insufficient"
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        # Fallback: check if response contains "sufficient" or "yes"
        lower_response = response.lower()
        if "sufficient" in lower_response or "yes" in lower_response:
            grade = "sufficient"
        else:
            grade = "insufficient"
    
    return {**state, "grade": grade, "status": "Context graded"}


def generate_answer(state: GraphRAGState) -> GraphRAGState:
    llm = create_llm()
    content = ANSWER_PROMPT.format(question=state.get("question", ""), context=state.get("merged_context", ""))
    answer = llm.invoke([HumanMessage(content=content)]).content
    citations = [{"source": h.get("metadata", {}).get("source", "unknown"), "chunk_id": h.get("metadata", {}).get("chunk_id", h.get("id", ""))} for h in state.get("vector_context", [])]
    return {**state, "answer": answer, "citations": citations, "status": "Answer generated"}


def evaluate_answer(state: GraphRAGState) -> GraphRAGState:
    return {**state, "status": "Answer evaluated", "answer": state.get("answer", "I don't know.")}


def return_answer(state: GraphRAGState) -> GraphRAGState:
    return {**state, "status": "Complete"}
