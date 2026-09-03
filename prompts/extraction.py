
"""Knowledge graph extraction prompts."""

GRAPH_EXTRACTION_PROMPT = """
<role>
You extract high-quality knowledge graph facts from documents.
</role>

<task>
Extract entities, concepts, keywords, and relationship triples.
</task>

<schema>
Return ONLY valid JSON with exactly these top-level keys:

entities:
[
  {{
    "name": "string",
    "type": "string",
    "description": "string",
    "confidence": 0.0
  }}
]

relationships:
[
  {{
    "subject": "string",
    "relation": "string",
    "object": "string",
    "confidence": 0.0,
    "evidence": "string"
  }}
]

concepts:
["string"]

keywords:
["string"]
</schema>

<quality>
Only extract facts explicitly supported by the document.
Do not invent facts.
Use canonical entity names.
Keep confidence between 0.0 and 1.0.
Every relationship subject and object should correspond to an entity
when possible.
</quality>

<output_rules>
Return ONLY the JSON object.
Do not use Markdown.
Do not wrap the JSON in code fences.
Do not add explanations before or after the JSON.
</output_rules>

<document>
{text}
</document>
""".strip()
