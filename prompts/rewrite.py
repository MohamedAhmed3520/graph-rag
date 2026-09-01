"""Query rewriting prompts."""

QUERY_REWRITE_PROMPT = """
<role>You rewrite user questions for hybrid vector and graph retrieval.</role>
<question>{question}</question>
<instructions>Expand acronyms cautiously, include entity aliases, and preserve user intent.</instructions>
Return only the rewritten query.
""".strip()

# Backward-compatible alias used by the workflow layer.
REWRITE_PROMPT = QUERY_REWRITE_PROMPT