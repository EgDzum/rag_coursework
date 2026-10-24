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

    def graph_expand(self, chunk_ids, max_hops=2, limit=20):
        """
        Expand the knowledge graph from entities mentioned in the retrieved chunks.

        Args:
            chunk_ids (list[str]): IDs of vector-retrieved chunks.
            max_hops (int): Maximum number of RELATES_TO hops.
            limit (int): Maximum number of graph results.

        Returns:
            list[dict]: Graph relationships and entity paths.
        """

        if not chunk_ids:
            return []

        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_hops must be an integer >= 1")

        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")

        query = f"""
        MATCH path =
            (c:Chunk)-[:MENTIONS]->(source:Entity)
            -[:RELATES_TO*1..{max_hops}]-(neighbor:Entity)

        WHERE c.id IN $chunk_ids
        AND source.id <> neighbor.id

        RETURN DISTINCT
            c.id AS chunk_id,

            source.id AS source_id,
            source.name AS source_entity,
            source.type AS source_type,

            neighbor.id AS neighbor_id,
            neighbor.name AS neighbor_entity,
            neighbor.type AS neighbor_type,

            [node IN nodes(path) | {{
                id: node.id,
                name: node.name,
                type: node.type
            }}] AS entity_path,

            [relation IN relationships(path) | {{
                type: relation.type
            }}] AS relation_path

        LIMIT $limit
        """

        return self.db.run(
            query,
            {
                "chunk_ids": chunk_ids,
                "limit": limit,
            },
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