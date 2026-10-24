from rag.prompts import ANSWER_PROMPT


class GraphRAG:

    def __init__(
        self,
        retriever,
        llm
    ):

        self.retriever = retriever
        self.llm = llm

    def build_context(self, results):
        """
        Build the context passed to the LLM from vector and graph retrieval.

        Args:
            results (dict): Retrieval results containing:
                - vector: Retrieved vector-search passages.
                - graph: Knowledge graph expansion results.

        Returns:
            str: Formatted context for the LLM prompt.
        """

        parts = []

        parts.append("=== VECTOR RETRIEVED PASSAGES ===")

        vector_results = results.get("vector", [])

        if vector_results:
            for i, item in enumerate(vector_results):
                parts.append(
                    f"""
    PASSAGE {i + 1}

    Chunk ID: {item.get("chunk_id", "")}
    Title: {item.get("title", "")}
    Authors: {item.get("authors", "")}

    {item.get("text", "")}
    """
                )
        else:
            parts.append("No vector passages were retrieved.")

        parts.append("\n=== KNOWLEDGE GRAPH ===")

        graph_results = results.get("graph", [])

        if graph_results:
            for i, item in enumerate(graph_results):

                source_entity = item.get("source_entity", "")
                source_type = item.get("source_type", "")

                neighbor_entity = item.get("neighbor_entity", "")
                neighbor_type = item.get("neighbor_type", "")

                # Extract relationship types from the graph path.
                relation_path = item.get("relation_path", [])

                relation_types = [
                    relation.get("type", "")
                    for relation in relation_path
                    if relation.get("type")
                ]

                relation_text = (
                    " -> ".join(relation_types)
                    if relation_types
                    else "Unknown relation"
                )

                # Format the entity path for multi-hop traversal.
                entity_path = item.get("entity_path", [])

                entity_path_text = " -> ".join(
                    node.get("name", "")
                    for node in entity_path
                    if node.get("name")
                )

                parts.append(
                    f"""
    GRAPH FACT {i + 1}

    Chunk ID:
    {item.get("chunk_id", "")}

    Source entity:
    {source_entity}

    Source entity type:
    {source_type}

    Relationship:
    {relation_text}

    Related entity:
    {neighbor_entity}

    Related entity type:
    {neighbor_type}

    Entity path:
    {entity_path_text}
    """
                )
        else:
            parts.append("No knowledge graph relationships were retrieved.")

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
            prompt
        )

        return {
            "answer": answer,
            "retrieval": results
        }