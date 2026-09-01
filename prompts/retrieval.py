"""Retrieval prompts."""

CONTEXT_MERGE_PROMPT = """
<role>You are a retrieval analyst merging vector chunks and graph paths.</role>
<task>Create a compact evidence pack for answering the user question.</task>
<question>{question}</question>
<vector_context>{vector_context}</vector_context>
<graph_context>{graph_context}</graph_context>
<instructions>Remove duplicates, preserve source ids, and keep contradictory evidence visible.</instructions>
""".strip()