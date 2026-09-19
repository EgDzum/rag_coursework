from neo4j import GraphDatabase


class Neo4jClient:

    def __init__(
        self,
        uri,
        username,
        password,
        database="neo4j"
    ):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

        self.database = database

    def close(self):
        self.driver.close()

    def run(
        self,
        query,
        parameters=None
    ):

        with self.driver.session(
            database=self.database
        ) as session:

            return session.run(
                query,
                parameters or {}
            ).data()

    def create_schema(self):

        queries = [

            """
            CREATE CONSTRAINT document_id IF NOT EXISTS
            FOR (d:Document)
            REQUIRE d.id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT chunk_id IF NOT EXISTS
            FOR (c:Chunk)
            REQUIRE c.id IS UNIQUE
            """,

            """
            CREATE CONSTRAINT entity_id IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.id IS UNIQUE
            """
        ]

        for query in queries:
            self.run(query)

    def create_document(
        self,
        document_id,
        title,
        authors
    ):

        self.run(
            """
            MERGE (d:Document {id: $id})
            SET
                d.title = $title,
                d.authors = $authors
            """,
            {
                "id": document_id,
                "title": title,
                "authors": authors
            }
        )

    def create_chunk(
        self,
        chunk_id,
        document_id,
        text,
        index
    ):

        self.run(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET
                c.text = $text,
                c.chunk_index = $index

            WITH c

            MATCH (d:Document {id: $document_id})

            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "index": index
            }
        )

    def create_entity(
        self,
        entity_id,
        name,
        entity_type
    ):

        self.run(
            """
            MERGE (e:Entity {id: $id})
            SET
                e.name = $name,
                e.type = $type
            """,
            {
                "id": entity_id,
                "name": name,
                "type": entity_type
            }
        )

    def link_chunk_entity(
        self,
        chunk_id,
        entity_id
    ):

        self.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {id: $entity_id})

            MERGE (c)-[:MENTIONS]->(e)
            """,
            {
                "chunk_id": chunk_id,
                "entity_id": entity_id
            }
        )

    def create_relation(
        self,
        source_id,
        target_id,
        relation
    ):

        # Relationship type cannot safely be parameterized
        # so we store the semantic relation as a property.

        self.run(
            """
            MATCH (s:Entity {id: $source})
            MATCH (t:Entity {id: $target})

            MERGE (s)-[r:RELATES_TO {type: $relation}]->(t)
            """,
            {
                "source": source_id,
                "target": target_id,
                "relation": relation
            }
        )