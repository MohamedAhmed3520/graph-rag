"""Context and answer grading prompts."""

CONTEXT_GRADING_PROMPT = """
<role>You grade whether retrieved evidence can answer a question.</role>
<question>{question}</question>
<context>{context}</context>
Return JSON: {"sufficient": boolean, "score": float, "reason": str}
""".strip()

# Backward-compatible alias used by the workflow layer.
GRADING_PROMPT = CONTEXT_GRADING_PROMPT

ANSWER_EVALUATION_PROMPT = """
<role>You are a factuality evaluator.</role>
<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>
Return JSON: {"grounded": boolean, "score": float, "issues": [str]}
""".strip()

# Backward-compatible alias for answer grading workflows.
ANSWER_GRADING_PROMPT = ANSWER_EVALUATION_PROMPT