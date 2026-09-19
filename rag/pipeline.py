from rag.prompts import ANSWER_PROMPT


class GraphRAG:

    def __init__(
        self,
        retriever,
        llm
    ):

        self.retriever = retriever
        self.llm = llm

    def build_context(
        self,
        results
    ):

        parts = []

        parts.append(
            "=== VECTOR RETRIEVED PASSAGES ==="
        )

        for i, item in enumerate(
            results["vector"]
        ):

            parts.append(
                f"""
PASSAGE {i + 1}

Title: {item.get("title", "")}
Authors: {item.get("authors", "")}

{item["text"]}
"""
            )

        parts.append(
            "\n=== KNOWLEDGE GRAPH ==="
        )

        for item in results["graph"]:

            parts.append(
                f"""
Chunk: {item["chunk_id"]}

Entity:
{item["source_entity"]}

Related entity:
{item["neighbor_entity"]}

Relation available in graph.
"""
            )

        return "\n".join(parts)

    def answer(
        self,
        question,
        top_k=8
    ):

        results = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        context = self.build_context(
            results
        )

        prompt = ANSWER_PROMPT.format(
            question=question,
            context=context
        )

        answer = self.llm.generate(
            prompt,
            max_new_tokens=1024,
        )

        return {
            "answer": answer,
            "retrieval": results
        }