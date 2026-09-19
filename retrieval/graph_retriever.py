class GraphRetriever:

    def __init__(
        self,
        neo4j_client,
        vector_store,
        embedding_model
    ):

        self.db = neo4j_client
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def vector_search(
        self,
        query,
        k=8
    ):

        embedding = self.embedding_model.encode(
            [query]
        )

        return self.vector_store.search(
            embedding,
            k=k
        )

    def graph_expand(
        self,
        chunk_ids,
        max_hops=2,
        limit=20
    ):

        if not chunk_ids:
            return []

        query = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids

        MATCH path =
            (c)-[:MENTIONS]->(e:Entity)
            -[:RELATES_TO*0..2]-(neighbor:Entity)

        RETURN
            c.id AS chunk_id,
            e.name AS source_entity,
            e.type AS source_type,
            neighbor.name AS neighbor_entity,
            neighbor.type AS neighbor_type

        LIMIT $limit
        """

        return self.db.run(
            query,
            {
                "chunk_ids": chunk_ids,
                "limit": limit
            }
        )

    def retrieve(
        self,
        query,
        top_k=8,
        graph_limit=20
    ):

        vector_results = self.vector_search(
            query,
            k=top_k
        )

        chunk_ids = [
            item["chunk_id"]
            for item in vector_results
        ]

        graph_results = self.graph_expand(
            chunk_ids,
            limit=graph_limit
        )

        return {
            "vector": vector_results,
            "graph": graph_results
        }