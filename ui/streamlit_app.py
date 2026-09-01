"""Production Streamlit interface for Graph RAG."""
from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from embeddings.vector_store import FaissVectorStore
from ingestion.ingest import ingest_documents
from ui.components import metric_row, render_citations, render_context, workflow_status
from ui.sidebar import render_sidebar
from ui.utils import graph_stats, initialize_session, render_pyvis_graph, save_uploaded_files, vector_stats
from workflow.workflow import build_qa_workflow


def run() -> None:
    """Run the Streamlit Graph RAG app."""
    st.set_page_config(page_title="Graph RAG", page_icon="🧠", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #151b2d 100%); }
    div[data-testid="stSidebar"] { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)
    initialize_session()
    settings = get_settings()
    options = render_sidebar(settings)
    st.title("🧠 Production Graph RAG")
    st.caption("Hybrid vector search + Neo4j graph traversal + LangGraph orchestration + OpenRouter LLMs")
    metric_row(graph_stats(settings), vector_stats(settings))
    with st.sidebar:
        uploaded = st.file_uploader("Upload PDF, TXT, DOCX, or Markdown", type=["pdf", "txt", "docx", "md", "markdown"], accept_multiple_files=True)
        process = st.button("🚀 Process documents", use_container_width=True)
        rebuild = st.button("♻️ Rebuild graph", use_container_width=True)
        clear_chat = st.button("🧹 Reset conversation", use_container_width=True)
        clear_vectors = st.button("🗑️ Clear vector database", use_container_width=True)
    if clear_chat:
        st.session_state.messages = []
        st.session_state.last_state = {}
        st.rerun()
    if clear_vectors:
        FaissVectorStore(settings).clear()
        st.success("Vector database cleared.")
    if process and uploaded:
        paths = save_uploaded_files(uploaded, settings.raw_data_dir)
        with st.spinner("Running ingestion LangGraph workflow..."):
            state = ingest_documents(paths)
        st.session_state.last_state = state
        if isinstance(state, dict):
            workflow_status(state.get("status", "Ingestion complete"), state.get("errors"))
        else:
            workflow_status(f"Ingestion complete ({len(state)} chunks)")
    if rebuild:
        st.info("Rebuild by uploading documents and pressing Process documents; graph upserts are idempotent.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask a question about your knowledge base...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                workflow = build_qa_workflow(settings=settings, model=options.model, temperature=options.temperature)
                result = workflow.invoke({"question": question})
                answer = result.get("answer", "I don't know based on the available evidence.")
                placeholder.markdown(answer)
                render_context(result.get("vector_context", []), result.get("graph_context", []))
                render_citations(result.get("citations", []))
                workflow_status(result.get("status", "Answer generated"), result.get("errors"))
                st.session_state.last_state = result
            except Exception as exc:
                answer = f"I could not complete the request: {exc}"
                placeholder.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    st.subheader("🕸️ Knowledge Graph Visualization")
    search = st.text_input("Search nodes", placeholder="Type an entity name...")
    try:
        from graph.graph_search import GraphSearch
        from graph.neo4j_store import Neo4jStore

        store = Neo4jStore(settings)
        graph = GraphSearch(store).subgraph(search or "", limit=80)
        store.close()
        render_pyvis_graph(graph.get("nodes", []), graph.get("edges", []))
    except Exception as exc:
        st.warning(f"Graph visualization unavailable: {exc}")


def render_app() -> None:
    """Backward-compatible alias for legacy imports and tests."""
    run()


if __name__ == "__main__":
    run()