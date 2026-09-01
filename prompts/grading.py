"""Context and answer grading prompts."""

CONTEXT_GRADING_PROMPT = """
<role>You grade whether retrieved evidence can answer a question.</role>
<question>{question}</question>
<context>{context}</context>

Respond with ONLY a JSON object on a single line, no markdown formatting, no code fences:
{{"sufficient": true, "score": 0.9, "reason": "explanation"}}

Where:
- sufficient: boolean (true if context answers the question well, false otherwise)
- score: float between 0.0 and 1.0
- reason: string explanation of your assessment
""".strip()

# Backward-compatible alias used by the workflow layer.
GRADING_PROMPT = CONTEXT_GRADING_PROMPT

ANSWER_EVALUATION_PROMPT = """
<role>You are a factuality evaluator.</role>
<question>{question}</question>
<answer>{answer}</answer>
<context>{context}</context>

Respond with ONLY a JSON object on a single line, no markdown formatting, no code fences:
{{"grounded": true, "score": 0.9, "issues": []}}

Where:
- grounded: boolean (true if answer is factually grounded in context, false otherwise)
- score: float between 0.0 and 1.0
- issues: array of strings describing any factual issues
""".strip()

# Backward-compatible alias for answer grading workflows.
ANSWER_GRADING_PROMPT = ANSWER_EVALUATION_PROMPT
