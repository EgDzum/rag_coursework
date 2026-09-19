import argparse
import yaml
from pathlib import Path

from graph.neo4j_client import Neo4jClient

from llm.qwen import QwenLLM

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore
from retrieval.graph_retriever import GraphRetriever

from rag.pipeline import GraphRAG



PROJECT_ROOT = Path(__file__).resolve()
CONFIG_PATH = PROJECT_ROOT.parent / "configs" / "config.yml"


def load_config():

    with open(
        CONFIG_PATH
    ) as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        type=str,
        required=True
    )

    args = parser.parse_args()

    config = load_config()

    # Embeddings
    embedding_config = config["embedding"]

    embeddings = QwenEmbedding(
        embedding_config["model"],
        embedding_config["batch_size"],
        embedding_config["normalize"],
        embedding_config["max_length"]
    )

    # FAISS
    store = FAISSStore(
        config["faiss"]["index_path"],
        config["faiss"]["metadata_path"]
    )

    store.load()

    # Neo4j
    neo4j_config = config["neo4j"]

    db = Neo4jClient(
        neo4j_config["uri"],
        neo4j_config["username"],
        neo4j_config["password"],
        neo4j_config["database"]
    )

    # Retriever
    retriever = GraphRetriever(
        db,
        store,
        embeddings
    )
    
    # LLM

    llm_config = config["llm"]

    llm = QwenLLM(
        llm_config["model"],
	max_input_tokens=llm_config["max_input_tokens"],
        max_new_tokens=llm_config["max_new_tokens"],
	temperature=llm_config["temperature"]
    )

    # GraphRAG
    rag = GraphRAG(
        retriever,
        llm
    )

    result = rag.answer(
        args.question,
        top_k=config["retrieval"]["top_k_vector"]
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("VECTOR SOURCES")
    print("=" * 80)

    for item in result["retrieval"]["vector"]:

        print(
            f"""
[{item['score']:.4f}]
{item['title']}
{item['chunk_id']}
"""
        )

    db.close()


if __name__ == "__main__":
    main()