import argparse
import json
import yaml

from pathlib import Path

from graph.neo4j_client import Neo4jClient

from llm.qwen import QwenLLM

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore
from retrieval.graph_retriever import GraphRetriever

from rag.pipeline import GraphRAG


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str):
    """
    Supported formats:

    JSONL:
    {"question": "..."}
    {"question": "..."}

    JSON:
    [
        {"question": "..."},
        {"question": "..."}
    ]
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() == ".jsonl":
        questions = []

        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

                questions.append(item)

        return questions

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "JSON dataset must contain a list of objects."
            )

        return data

    raise ValueError(
        "Unsupported dataset format. Use .json or .jsonl."
    )


def get_question(item: dict, question_field: str = "question", answer_field: str = "answers"):
    """
    Extracts a question from a dataset record.
    """

    question = item.get(question_field)
    answer = item.get(answer_field)[0]

    if question is None:
        raise KeyError(
            f"Question field '{question_field}' "
            f"not found in dataset item: {item}"
        )

    question = str(question).strip()

    if not question:
        raise ValueError(
            f"Empty question in dataset item: {item}"
        )

    return question, str(answer).strip()


def save_result(output_file, result: dict):
    """
    Saves a single record to JSONL.
    """

    output_file.write(
        json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )

    output_file.flush()


def build_output_record(item: dict, question: str, result: dict, index: int, ground_truth_answer: str):
    """
    Builds a result record.

    The following data is preserved:
    - the original dataset record;
    - the question index;
    - the question;
    - the answer;
    - retrieval results.

    Additional fields returned by GraphRAG
    are also preserved.
    """

    output = {
        "id": item.get(
            "id",
            item.get("question_id", index),
        ),
        "question": question,
        "answer": result.get("answer"),
        "retrieval": result.get("retrieval"),
        "ground_truth_answer": ground_truth_answer
    }

    for key, value in result.items():
        if key not in output:
            output[key] = value

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG over a complete dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSON or JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file.",
    )

    parser.add_argument(
        "--question-field",
        type=str,
        default="input",
        help="Field containing the question.",
    )

    args = parser.parse_args()

    config = load_config()

    dataset = load_dataset(args.dataset)

    if not dataset:
        raise ValueError(
            "Dataset is empty."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    embedding_config = config["embedding"]

    embeddings = QwenEmbedding(
        embedding_config["model"],
        embedding_config["batch_size"],
        embedding_config["normalize"],
        embedding_config["max_length"],
    )

    store = FAISSStore(
        config["faiss"]["index_path"],
        config["faiss"]["metadata_path"],
    )

    store.load()

    neo4j_config = config["neo4j"]

    db = Neo4jClient(
        neo4j_config["uri"],
        neo4j_config["username"],
        neo4j_config["password"],
        neo4j_config["database"],
    )

    try:
        retriever = GraphRetriever(
            db,
            store,
            embeddings,
        )

        llm_config = config["llm"]

        llm = QwenLLM(
            model_name=llm_config["model"],
            max_input_tokens=llm_config["max_input_tokens"],
            max_new_tokens=llm_config["max_new_tokens"],
            temperature=llm_config["temperature"],
        )

        rag = GraphRAG(
            retriever,
            llm,
        )

        top_k = config["retrieval"]["top_k_vector"]

        total_questions = len(dataset)

        print("=" * 80)
        print("GRAPH RAG DATASET INFERENCE")
        print("=" * 80)
        print(f"Dataset: {args.dataset}")
        print(f"Questions: {total_questions}")
        print(f"Output: {args.output}")
        print(f"Top-k vector: {top_k}")
        print("=" * 80)

        processed = 0
        failed = 0

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as output_file:

            for index, item in enumerate(dataset, start=1):

                try:
                    question, answer = get_question(
                        item,
                        args.question_field,
                    )

                    print(
                        f"\n[{index}/{total_questions}] "
                        f"Processing question..."
                    )

                    result = rag.answer(
                        question,
                        top_k=top_k,
                    )

                    output_record = build_output_record(
                        item=item,
                        question=question,
                        result=result,
                        index=index,
                        ground_truth_answer=answer
                    )

                    save_result(
                        output_file,
                        output_record,
                    )

                    processed += 1

                    print(
                        f"[{index}/{total_questions}] "
                        f"Saved successfully."
                    )

                except Exception as exc:
                    failed += 1

                    print(
                        f"[{index}/{total_questions}] "
                        f"ERROR: {type(exc).__name__}: {exc}"
                    )
                    error_record = {
                        "id": item.get(
                            "id",
                            item.get("question_id", index),
                        ),
                        "question": item.get(
                            args.question_field
                        ),
                        "answer": None,
                        "retrieval": None,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }

                    save_result(
                        output_file,
                        error_record,
                    )

        print("\n" + "=" * 80)
        print("FINISHED")
        print("=" * 80)
        print(f"Total questions: {total_questions}")
        print(f"Processed successfully: {processed}")
        print(f"Failed: {failed}")
        print(f"Output file: {output_path}")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    main()