"""Knowledge graph extraction prompts."""

GRAPH_EXTRACTION_PROMPT = """
<role>You extract high-quality knowledge graph facts from documents.</role>
<task>Extract entities, concepts, keywords, and relationship triples.</task>
<schema>
Return strict JSON with keys: entities, relationships, concepts, keywords.
entities: [{"name": str, "type": str, "description": str, "confidence": float}]
relationships: [{"subject": str, "relation": str, "object": str, "confidence": float, "evidence": str}]
concepts: [str]
keywords: [str]
</schema>
<quality>
Only extract facts explicitly supported by the text. Use canonical entity names.
</quality>
<document>{text}</document>
""".strip()