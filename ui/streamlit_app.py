
"""Production Streamlit interface for Graph RAG."""
from __future__ import annotations

import streamlit as st

from config.settings import get_settings

from embeddings.vector_store import FaissVectorStore

from graph.graph_search import GraphSearch
from graph.neo4j_store import Neo4jStore

from ui.components import (
    metric_row,
    render_citations,
    render_context,
    workflow_status,
)

from ui.sidebar import render_sidebar

from ui.utils import (
    graph_stats,
    initialize_session,
    render_pyvis_graph,
    save_uploaded_files,
    vector_stats,
)

from workflow.workflow import (
    build_ingestion_workflow,
    build_qa_workflow,
)


def render_health_panel(settings) -> None:
    """Render live OpenRouter, FAISS and Neo4j health information."""
    st.subheader("📊 RAG Health")

    # -------------------------
    # OpenRouter / LLM key
    # -------------------------
    st.markdown("### 🤖 OpenRouter (LLM)")

    try:
        _, source = settings.resolve_openrouter_key()
        source_label = {
            "streamlit": "Streamlit Secrets",
            "env": "environment variable",
            "env_file": ".env file (committed)",
        }.get(source, source)
        st.caption(f"Active key source: **{source_label}**")

        if source == "env_file":
            st.warning(
                "The API key is being read from the committed `.env` file. "
                "Set `OPENROUTER_API_KEY` in Streamlit → App settings → "
                "Secrets; it now takes precedence."
            )
    except Exception as exc:
        st.error(f"No OpenRouter key configured: {exc}")

    if st.button("🔑 Test OpenRouter key", use_container_width=False):
        with st.spinner("Verifying OpenRouter API key..."):
            result = settings.verify_openrouter_key()
        status = result.get("status")
        message = result.get("message", "")
        if status == "ok":
            st.success(message)
            detail = result.get("detail")
            if isinstance(detail, dict):
                st.caption(
                    "Usage / limit: "
                    + ", ".join(
                        f"{k}={v}"
                        for k, v in detail.items()
                        if k in {"limit", "usage", "limit_remaining", "is_free_tier"}
                    )
                )
        elif status == "invalid":
            st.error(message)
        else:
            st.warning(message)

    st.divider()

    col1, col2 = st.columns(2)

    # -------------------------
    # FAISS
    # -------------------------
    with col1:
        st.markdown("### 📚 FAISS")

        try:
            vector_store = FaissVectorStore(settings)
            stats = vector_store.stats()

            st.metric(
                "Vectors",
                stats.get("vectors", 0),
            )

            st.metric(
                "Documents",
                stats.get("documents", 0),
            )

            st.metric(
                "Embedding dimension",
                stats.get("dimension", "—"),
            )

            if stats.get("vectors", 0) == 0:
                st.warning(
                    "FAISS is empty. "
                    "Process at least one document before asking questions."
                )

        except Exception as exc:
            st.error(
                f"FAISS health check failed: {exc}"
            )

    # -------------------------
    # Neo4j
    # -------------------------
    with col2:
        st.markdown("### 🕸️ Neo4j")

        try:
            store = Neo4jStore(settings)

            if store.is_available():
                stats = store.stats()

                st.success(
                    "Neo4j connected"
                )

                st.metric(
                    "Graph nodes",
                    stats.get("nodes", 0),
                )

                st.metric(
                    "Relationships",
                    stats.get("relationships", 0),
                )
            else:
                st.warning(
                    "Neo4j unavailable. "
                    "Vector-only RAG will still work."
                )

            store.close()

        except Exception as exc:
            st.error(
                f"Neo4j health check failed: {exc}"
            )


def render_last_retrieval(state: dict) -> None:
    """Render useful retrieval diagnostics from the last QA run."""

    if not state:
        return

    vector_context = state.get(
        "vector_context",
        [],
    )

    graph_context = state.get(
        "graph_context",
        [],
    )

    rewritten = state.get(
        "rewritten_question",
        "",
    )

    status = state.get(
        "status",
        "",
    )

    errors = state.get(
        "errors",
        [],
    )

    st.subheader(
        "🔎 Last Retrieval"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Vector hits",
        len(vector_context),
    )

    col2.metric(
        "Graph hits",
        len(graph_context),
    )

    col3.metric(
        "Total evidence",
        len(vector_context)
        + len(graph_context),
    )

    if rewritten:
        with st.expander(
            "Rewritten query",
            expanded=False,
        ):
            st.write(rewritten)

    if status:
        st.caption(
            f"Workflow status: {status}"
        )

    if errors:
        with st.expander(
            "⚠️ Workflow errors",
            expanded=True,
        ):
            for error in errors:
                st.warning(str(error))


def run() -> None:
    """Run the Streamlit Graph RAG application."""

    # =========================================================
    # Page configuration
    # =========================================================

    st.set_page_config(
        page_title="Graph RAG",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(
                    135deg,
                    #0e1117 0%,
                    #151b2d 100%
                );
        }

        div[data-testid="stSidebar"] {
            background-color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # Session
    # =========================================================

    initialize_session()

    settings = get_settings()

    # =========================================================
    # Sidebar
    # =========================================================

    options = render_sidebar(settings)

    st.title(
        "🧠 Production Graph RAG"
    )

    st.caption(
        "Hybrid vector search + Neo4j graph traversal "
        "+ LangGraph orchestration + OpenRouter LLMs"
    )

    # =========================================================
    # Top metrics
    # =========================================================

    graph_metrics = graph_stats(
        settings
    )

    vector_metrics = vector_stats(
        settings
    )

    metric_row(
        graph_metrics,
        vector_metrics,
    )

    # =========================================================
    # Sidebar actions
    # =========================================================

    with st.sidebar:

        st.divider()

        uploaded = st.file_uploader(
            "Upload PDF, TXT, DOCX, or Markdown",
            type=[
                "pdf",
                "txt",
                "docx",
                "md",
                "markdown",
            ],
            accept_multiple_files=True,
        )

        process = st.button(
            "🚀 Process documents",
            use_container_width=True,
        )

        rebuild = st.button(
            "♻️ Rebuild indexes",
            use_container_width=True,
        )

        clear_chat = st.button(
            "🧹 Reset conversation",
            use_container_width=True,
        )

        clear_vectors = st.button(
            "🗑️ Clear vector database",
            use_container_width=True,
        )

    # =========================================================
    # Reset chat
    # =========================================================

    if clear_chat:
        st.session_state.messages = []
        st.session_state.last_state = {}
        st.rerun()

    # =========================================================
    # Clear vectors
    # =========================================================

    if clear_vectors:

        try:
            FaissVectorStore(
                settings
            ).clear()

            st.success(
                "FAISS vector database cleared."
            )

            st.info(
                "Upload the document again and press "
                "'Process documents'."
            )

        except Exception as exc:
            st.error(
                f"Could not clear FAISS: {exc}"
            )

    # =========================================================
    # Rebuild indexes
    # =========================================================

    if rebuild:

        try:
            FaissVectorStore(
                settings
            ).clear()

            st.success(
                "FAISS index cleared."
            )

            st.info(
                "Upload your documents again "
                "and press Process documents."
            )

        except Exception as exc:
            st.error(
                f"Could not rebuild index: {exc}"
            )

    # =========================================================
    # Process uploaded documents
    # =========================================================

    if process:

        if not uploaded:
            st.warning(
                "Upload at least one document first."
            )

        else:

            # Preflight: confirm the OpenRouter key is valid before spending
            # time on parsing/extraction, which otherwise fails per-chunk.
            with st.spinner("Checking OpenRouter API key..."):
                key_check = settings.verify_openrouter_key()

            if key_check.get("status") == "missing":
                st.error(
                    "No OpenRouter API key is configured. Add "
                    "`OPENROUTER_API_KEY` in Streamlit → App settings → "
                    f"Secrets. ({key_check.get('message', '')})"
                )
                st.stop()

            if key_check.get("status") == "invalid":
                st.error(key_check.get("message"))
                st.info(
                    "Get a valid key at https://openrouter.ai/keys, then set "
                    "it in Streamlit → App settings → Secrets as "
                    "`OPENROUTER_API_KEY`."
                )
                st.stop()

            if key_check.get("status") == "error":
                st.warning(
                    "Could not verify the OpenRouter key up front "
                    f"({key_check.get('message')}). Continuing anyway."
                )

            paths = save_uploaded_files(
                uploaded,
                settings.raw_data_dir,
            )

            st.info(
                f"Preparing {len(paths)} document(s)..."
            )

            with st.spinner(
                "Running ingestion workflow..."
            ):

                try:
                    ingestion_workflow = (
                        build_ingestion_workflow()
                    )

                    state = ingestion_workflow.invoke(
                        {
                            "file_paths": paths
                        }
                    )

                    st.session_state.last_state = (
                        state
                    )

                except Exception as exc:

                    st.error(
                        "Document ingestion failed."
                    )

                    st.exception(
                        exc
                    )

                    state = None

            # -------------------------------------------------
            # Ingestion result
            # -------------------------------------------------

            if isinstance(state, dict):

                chunks = state.get(
                    "chunks",
                    [],
                )

                errors = state.get(
                    "errors",
                    [],
                )

                status = state.get(
                    "status",
                    "Ingestion complete",
                )

                workflow_status(
                    status,
                    errors,
                )

                # ---------------------------------------------
                # Actual FAISS stats
                # ---------------------------------------------

                try:
                    vector_store = (
                        FaissVectorStore(
                            settings
                        )
                    )

                    vector_stats_now = (
                        vector_store.stats()
                    )

                    st.success(
                        "Ingestion workflow finished."
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    col1.metric(
                        "Chunks",
                        len(chunks),
                    )

                    col2.metric(
                        "FAISS vectors",
                        vector_stats_now.get(
                            "vectors",
                            0,
                        ),
                    )

                    col3.metric(
                        "Stored documents",
                        vector_stats_now.get(
                            "documents",
                            0,
                        ),
                    )

                    if vector_stats_now.get(
                        "vectors",
                        0,
                    ) == 0:

                        st.error(
                            "No vectors were indexed. "
                            "Retrieval will return nothing until "
                            "the indexing problem is fixed."
                        )

                except Exception as exc:

                    st.error(
                        f"Could not inspect FAISS after ingestion: {exc}"
                    )

            else:

                st.error(
                    "Ingestion returned no workflow state."
                )

    # =========================================================
    # Health panel
    # =========================================================

    with st.expander(
        "📊 RAG system health",
        expanded=False,
    ):

        render_health_panel(
            settings
        )

    # =========================================================
    # Conversation history
    # =========================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # =========================================================
    # Question input
    # =========================================================

    question = st.chat_input(
        "Ask a question about your knowledge base..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            placeholder = st.empty()

            try:

                # ---------------------------------------------
                # Before QA: inspect vector database
                # ---------------------------------------------

                try:

                    vector_store = (
                        FaissVectorStore(
                            settings
                        )
                    )

                    stats = vector_store.stats()

                    if stats.get(
                        "vectors",
                        0,
                    ) == 0:

                        answer = (
                            "I can't retrieve anything yet "
                            "because the FAISS vector database "
                            "is empty. Please process your documents "
                            "first."
                        )

                        placeholder.warning(
                            answer
                        )

                        state = {
                            "answer": answer,
                            "vector_context": [],
                            "graph_context": [],
                            "citations": [],
                            "status": (
                                "FAISS index is empty"
                            ),
                            "errors": [],
                        }

                        st.session_state.last_state = (
                            state
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        return

                except Exception as exc:

                    placeholder.error(
                        "Could not inspect the vector database."
                    )

                    st.exception(
                        exc
                    )

                    return

                # ---------------------------------------------
                # QA workflow
                # ---------------------------------------------

                workflow = build_qa_workflow(
                    settings=settings,
                    model=options.model,
                    temperature=options.temperature,
                )

                result = workflow.invoke(
                    {
                        "question": question
                    }
                )

                st.session_state.last_state = (
                    result
                )

                # ---------------------------------------------
                # Answer
                # ---------------------------------------------

                answer = result.get(
                    "answer",
                    "I don't know based on the available evidence.",
                )

                placeholder.markdown(
                    answer
                )

                # ---------------------------------------------
                # Retrieval diagnostic
                # ---------------------------------------------

                vector_context = result.get(
                    "vector_context",
                    [],
                )

                graph_context = result.get(
                    "graph_context",
                    [],
                )

                if not vector_context and not graph_context:

                    st.error(
                        "⚠️ The workflow completed, "
                        "but retrieved zero pieces of evidence."
                    )

                else:

                    st.success(
                        f"Retrieved "
                        f"{len(vector_context)} vector chunks "
                        f"and "
                        f"{len(graph_context)} graph facts."
                    )

                # ---------------------------------------------
                # Context
                # ---------------------------------------------

                render_context(
                    vector_context,
                    graph_context,
                )

                # ---------------------------------------------
                # Citations
                # ---------------------------------------------

                render_citations(
                    result.get(
                        "citations",
                        [],
                    )
                )

                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                workflow_status(
                    result.get(
                        "status",
                        "Answer generated",
                    ),
                    result.get(
                        "errors",
                    ),
                )

                # ---------------------------------------------
                # Diagnostics
                # ---------------------------------------------

                render_last_retrieval(
                    result
                )

            except Exception as exc:

                import traceback

                error_msg = str(
                    exc
                )

                error_trace = traceback.format_exc()

                answer = (
                    f"I could not complete the request: "
                    f"{error_msg}"
                )

                placeholder.error(
                    answer
                )

                st.error(
                    "Full traceback:"
                )

                st.code(
                    error_trace,
                    language="text",
                )

                st.session_state.last_state = {
                    "answer": answer,
                    "vector_context": [],
                    "graph_context": [],
                    "citations": [],
                    "status": "QA failed",
                    "errors": [
                        error_msg
                    ],
                }

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

    # =========================================================
    # Knowledge graph visualization
    # =========================================================

    st.subheader(
        "🕸️ Knowledge Graph Visualization"
    )

    search = st.text_input(
        "Search nodes",
        placeholder="Type an entity name...",
    )

    try:

        store = Neo4jStore(
            settings
        )

        if not store.is_available():

            st.info(
                "Neo4j is not currently available. "
                "Graph visualization will appear after "
                "connecting the configured Neo4j Aura instance."
            )

            store.close()

        else:

            graph = GraphSearch(
                store
            ).subgraph(
                search or "",
                limit=80,
            )

            store.close()

            nodes = graph.get(
                "nodes",
                [],
            )

            edges = graph.get(
                "edges",
                [],
            )

            if nodes or edges:

                render_pyvis_graph(
                    nodes,
                    edges,
                )

            else:

                st.info(
                    "No graph results found for this search."
                )

    except Exception as exc:

        st.warning(
            f"Graph visualization unavailable: {exc}"
        )


def render_app() -> None:
    """Backward-compatible alias."""
    run()


if __name__ == "__main__":
    run()
