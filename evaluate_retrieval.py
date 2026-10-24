import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore


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
        # Supports datasets wrapped in {"data": [...]}.
        data = data.get("data", data.get("records", []))

    if not isinstance(data, list):
        raise ValueError("Expected a list of dataset records.")

    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline retrieval for later LLM-as-judge evaluation"
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--question-field", default="question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0,
                        help="0 means process the entire dataset")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset)

    if args.max_items > 0:
        dataset = dataset[:args.max_items]

    print(f"Questions to process: {len(dataset)}")
    print("Loading embedding model...")

    # If your QwenEmbedding constructor uses a different argument
    # name, adjust this line to match your implementation.
    embedding_model = QwenEmbedding(args.embedding_model)

    print("Loading FAISS index and metadata...")

    vector_store = FAISSStore(
        index_path=args.index,
        metadata_path=args.metadata,
    )
    vector_store.load()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_questions = 0
    missing_passages = 0
    processed = 0

    with output_path.open("w", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Retrieving"):
            question = item.get(args.question_field)

            if not isinstance(question, str) or not question.strip():
                missing_questions += 1
                continue

            query_embedding = embedding_model.encode([question])
            results = vector_store.search(
                query_embedding,
                k=args.top_k,
            )

            passages = []

            for rank, result in enumerate(results, start=1):
                passage_text = result.get("text", "")

                if not isinstance(passage_text, str) or not passage_text.strip():
                    missing_passages += 1

                passages.append({
                    "rank": rank,
                    "chunk_id": result.get("chunk_id"),
                    "document_id": result.get("document_id"),
                    "title": result.get("title", ""),
                    "text": passage_text,
                    "score": result.get("score"),
                })

            record = {
                "question_id": item.get("id", processed),
                "question": question,
                "answers": item.get("answers", []),
                "retrieved_passages": passages,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    print("\nRetrieval completed.")
    print(f"Questions saved: {processed}")
    print(f"Questions skipped: {missing_questions}")
    print(f"Passages with empty text: {missing_passages}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()