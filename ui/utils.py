"""UI utilities, including persistence and graph visualization."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import Settings
from embeddings.vector_store import FaissVectorStore
from graph.neo4j_store import Neo4jStore


def initialize_session() -> None:
    """Initialize Streamlit session state defaults."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_state", {})
    st.session_state.setdefault("workflow_status", "Idle")


def save_uploaded_files(uploaded_files: list[Any], target_dir: Path) -> list[str]:
    """Persist Streamlit uploaded files to disk and return paths."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for uploaded in uploaded_files:
        path = target_dir / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        paths.append(str(path))
    return paths


def graph_stats(settings: Settings) -> dict[str, int]:
    """Safely fetch Neo4j graph statistics."""
    try:
        store = Neo4jStore(settings)
        stats = store.stats()
        store.close()
        return stats
    except Exception:
        return {"nodes": 0, "relationships": 0}


def vector_stats(settings: Settings) -> dict[str, Any]:
    """Safely fetch FAISS vector store statistics."""
    try:
        store = FaissVectorStore(settings)
        return store.stats()
    except Exception:
        return {"documents": 0, "dimension": "—"}


def render_pyvis_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Render an interactive PyVis graph inside Streamlit."""
    try:
        from pyvis.network import Network
    except ImportError:
        st.info("Install `pyvis` to enable interactive graph visualization.")
        if nodes or edges:
            st.json({"nodes": nodes, "edges": edges})
        return

    net = Network(height="520px", width="100%", bgcolor="#0e1117", font_color="#fafafa", directed=True)
    net.barnes_hut()
    for node in nodes:
        node_id = str(node.get("id") or node.get("name"))
        net.add_node(node_id, label=str(node.get("name", node_id)), title=str(node), group=node.get("type", "Entity"))
    for edge in edges:
        source = str(edge.get("source") or edge.get("subject"))
        target = str(edge.get("target") or edge.get("object"))
        label = str(edge.get("relation", edge.get("type", "RELATED_TO")))
        net.add_edge(source, target, label=label, title=str(edge), value=float(edge.get("confidence", 1.0)))
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as handle:
        net.write_html(handle.name, notebook=False)
        html = Path(handle.name).read_text(encoding="utf-8")
    
    # Use st.html() if available (Streamlit >= 1.27.0), fallback to st.components.v1.html()
    try:
        st.html(html)
    except AttributeError:
        st.components.v1.html(html, height=540, scrolling=True)