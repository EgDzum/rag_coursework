import argparse
import json
from pathlib import Path

from tqdm import tqdm

from retrieval.graph_retriever import GraphRetriever
from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore
from graph.neo4j_client import Neo4jClient


def load_dataset(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("data", data.get("records", []))

    if not isinstance(data, list):
        raise ValueError("Expected a list of dataset records.")

    return data


def graph_fact_to_text(fact):
    """Convert a retrieved graph result into judge-readable text."""
    source = fact.get("source_entity", "")
    neighbor = fact.get("neighbor_entity", "")

    relations = [
        rel.get("type", "")
        for rel in fact.get("relation_path", [])
        if rel.get("type")
    ]

    entity_path = [
        node.get("name", "")
        for node in fact.get("entity_path", [])
        if node.get("name")
    ]

    parts = []

    if source and neighbor:
        parts.append(f"Source entity: {source}")
        parts.append(f"Related entity: {neighbor}")

    if relations:
        parts.append("Relationship types: " + " -> ".join(relations))

    if entity_path:
        parts.append("Entity path: " + " -> ".join(entity_path))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--graph-limit", type=int, default=20)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=1)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_items > 0:
        dataset = dataset[:args.max_items]

    embedding_model = QwenEmbedding(args.embedding_model)

    vector_store = FAISSStore(
        index_path=args.index,
        metadata_path=args.metadata,
    )
    vector_store.load()

    neo4j_client = Neo4jClient(uri="bolt://localhost:7687", username="neo4j", password="password")

    retriever = GraphRetriever(
        neo4j_client=neo4j_client,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(tqdm(dataset, desc="Graph retrieval")):
            question = item.get("input", "")

            if not isinstance(question, str) or not question.strip():
                continue

            vector_results = retriever.vector_search(
                question,
                k=args.top_k,
            )

            seed_chunk_ids = [
                result["chunk_id"]
                for result in vector_results
                if result.get("chunk_id") is not None
            ]

            graph_results = retriever.graph_expand(
                chunk_ids=seed_chunk_ids,
                max_hops=args.max_hops,
                limit=args.graph_limit,
            )

            facts = []

            for rank, fact in enumerate(graph_results, start=1):
                fact = dict(fact)
                fact["rank"] = rank
                fact["fact_text"] = graph_fact_to_text(fact)
                facts.append(fact)

            record = {
                "question_id": item.get("_id", idx),
                "question": question,
                "answers": item.get("answers", []),
                "seed_chunk_ids": seed_chunk_ids,
                "graph_facts": facts,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Graph retrieval results saved to: {output_path}")

    if hasattr(neo4j_client, "close"):
        neo4j_client.close()


if __name__ == "__main__":
    main()