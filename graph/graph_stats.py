from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "password"


def main():
    driver = GraphDatabase.driver(
        URI,
        auth=(USER, PASSWORD),
    )

    with driver.session() as session:
        nodes = session.run(
            "MATCH (n) RETURN count(n) AS count"
        ).single()["count"]

        relationships = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        ).single()["count"]

        node_types = session.run(
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS count
            ORDER BY count DESC
            """
        )

        relationship_types = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(*) AS count
            ORDER BY count DESC
            """
        )

        print("=" * 50)
        print("GRAPH STATISTICS")
        print("=" * 50)

        print(f"\nTotal nodes:         {nodes:,}")
        print(f"Total relationships: {relationships:,}")

        print("\nNodes by label:")
        for record in node_types:
            print(f"  {record['label']:<20} {record['count']:,}")

        print("\nRelationships by type:")
        for record in relationship_types:
            print(f"  {record['type']:<20} {record['count']:,}")

        print("=" * 50)

    driver.close()


if __name__ == "__main__":
    main()