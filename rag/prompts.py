GRAPH_EXTRACTION_PROMPT = """
You are an information extraction system.

Extract a compact knowledge graph from the following
agriculture-related text.

Return ONLY valid JSON.

Schema:

{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type"
    }}
  ],
  "relations": [
    {{
      "source": "entity name",
      "relation": "short relation",
      "target": "entity name"
    }}
  ]
}}

Rules:

1. Extract only entities explicitly supported by the text.
2. Prefer meaningful domain entities.
3. Do not create entities for generic words.
4. Relations must be short predicates.
5. Do not infer unsupported facts.
6. Keep the graph compact.

Text:

{text}
"""

ANSWER_PROMPT = """
You are a retrieval-augmented question answering system.

Answer the question using ONLY the supplied context.

The context consists of:
1. Vector-retrieved passages.
2. Knowledge graph information.

If the context does not contain enough information,
say that the answer cannot be determined from the
provided context.

Question:
{question}

Context:
{context}

Answer:
""" 