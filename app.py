"""Application entrypoint for `streamlit run app.py`.

This module intentionally imports ``sitecustomize`` first so the project-level
SSL/OpenMP safeguards are applied before Streamlit or Tornado initialize.
"""
from __future__ import annotations

# Import side effects are intentional: the project uses sitecustomize to patch
# Windows certificate loading and configure local runtime defaults before any
# Streamlit/Tornado imports occur.
import sitecustomize  # noqa: F401

from ui.streamlit_app import run


if __name__ == "__main__":
    run()
