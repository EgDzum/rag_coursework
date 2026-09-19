import json
import yaml
from pathlib import Path

from tqdm import tqdm

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT.parent / "configs" / "config.yml"
# CONFIG_PATH = "configs/config.yaml"
DATA_PATH = "data/processed/chunks.jsonl"


def load_chunks():

    chunks = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            chunks.append(
                json.loads(line)
            )

    return chunks


def main():

    print("retrieval started")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    chunks = load_chunks()

    texts = [
        item["text"]
        for item in chunks
    ]

    embedding_config = config["embedding"]

    encoder = QwenEmbedding(
        model_name=embedding_config["model"],
        batch_size=embedding_config["batch_size"],
        normalize=embedding_config["normalize"],
        max_length=embedding_config["max_length"]
    )

    embeddings = encoder.encode(texts)

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "text": chunk["text"],
            "title": chunk["title"],
            "authors": chunk["authors"],
            "question": chunk["question"],
            "answers": chunk["answers"],
            "label": chunk["label"]
        })

    store = FAISSStore(
        config["faiss"]["index_path"],
        config["faiss"]["metadata_path"]
    )

    store.build(
        embeddings,
        metadata
    )

    store.save()

    print(
        f"FAISS index created: {len(metadata)} chunks"
    )


if __name__ == "__main__":
    main()