import argparse
import json
import yaml

from pathlib import Path

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore


PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = (
    PROJECT_ROOT.parent
    / "configs"
    / "config.yml"
)

DEFAULT_DATA_PATH = (
    PROJECT_ROOT.parent
    / "data"
    / "processed"
    / "legal_chunks.jsonl"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build or extend a FAISS index"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to chunks JSONL file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["build", "append"],
        default="build",
        help="Build a new index or append vectors"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config YAML"
    )

    return parser.parse_args()


def load_chunks(data_path):

    chunks = []

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            chunks.append(
                json.loads(line)
            )

    return chunks


def prepare_metadata(chunks):

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

    return metadata


def main():

    args = parse_args()

    print(
        f"Retrieval started | mode={args.mode}"
    )

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    data_path = Path(
        args.data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    chunks = load_chunks(
        data_path
    )

    if not chunks:
        raise ValueError(
            "No chunks found in input file"
        )

    print(
        f"Loaded chunks: {len(chunks)}"
    )

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

    print("Creating embeddings...")

    embeddings = encoder.encode(
        texts
    )

    metadata = prepare_metadata(
        chunks
    )

    store = FAISSStore(
        config["faiss"]["index_path"],
        config["faiss"]["metadata_path"]
    )

    if args.mode == "build":

        print("Building a new FAISS index...")

        store.build(
            embeddings,
            metadata
        )

    elif args.mode == "append":

        print("Loading existing FAISS index...")

        store.load()

        print(
            f"Existing vectors: {store.index.ntotal}"
        )

        print(
            f"Existing metadata: {len(store.metadata)}"
        )

        store.append(
            embeddings,
            metadata
        )

    store.save()

    print(
        "FAISS index saved successfully"
    )

    print(
        f"Total vectors: {store.index.ntotal}"
    )

    print(
        f"Total metadata records: {len(store.metadata)}"
    )


if __name__ == "__main__":
    main()