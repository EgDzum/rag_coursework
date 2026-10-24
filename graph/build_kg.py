import argparse
import hashlib
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from llm.qwen import QwenLLM
from rag.prompts import GRAPH_EXTRACTION_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build knowledge graph from a chunks JSONL dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input chunks JSONL file."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to configuration YAML file."
    )

    return parser.parse_args()


def entity_id(name):
    return hashlib.sha256(
        name.lower().strip().encode()
    ).hexdigest()[:32]


def count_rows(file_path):
    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        return sum(
            1
            for line in f
            if line.strip()
        )

def get_input_chunk_ids(file_path):
    chunk_ids = set()

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)

            chunk_id = chunk.get("chunk_id")

            if chunk_id is not None:
                chunk_ids.add(chunk_id)

    return chunk_ids


def get_processed_chunk_ids(db):
    query = """
    MATCH (c:Chunk)
    RETURN c.id AS chunk_id
    """

    results = db.run(query)

    return {
        record["chunk_id"]
        for record in results
        if record["chunk_id"] is not None
    }


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Dataset path
    # --------------------------------------------------------

    data_path = args.input.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    print(
        f"Input dataset: {data_path}",
        flush=True
    )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    neo4j_config = config["neo4j"]

    # --------------------------------------------------------
    # Connect to Neo4j
    # --------------------------------------------------------

    db = Neo4jClient(
        neo4j_config["uri"],
        neo4j_config["username"],
        neo4j_config["password"],
        neo4j_config["database"]
    )

    db.create_schema()

    # --------------------------------------------------------
    # Load LLM
    # --------------------------------------------------------

    llm = QwenLLM(
        model_name=config["llm"]["model"],
        max_new_tokens=config["llm"]["max_new_tokens"],
        max_input_tokens=config["llm"]["max_input_tokens"],
        temperature=config["llm"]["temperature"]
    )

    # --------------------------------------------------------
    # Calculate processing statistics
    # --------------------------------------------------------

    total_rows = count_rows(data_path)

    print(
        "Counting chunks in input dataset...",
        flush=True
        )


    # print(
    #     f"Total rows in dataset: {total_rows}",
    #     flush=True
    # )

    # print(
    #     "Retrieving already processed chunks from Neo4j...",
    #     flush=True
    # )
    input_chunk_ids = get_input_chunk_ids(data_path)

    processed_chunk_ids = get_processed_chunk_ids(db)

    print(
        f"Unique chunks in dataset: {total_rows}",
        flush=True
        )

    print(
        "Retrieving already processed chunks from Neo4j...",
        flush=True
        )

    processed_in_input = (input_chunk_ids & processed_chunk_ids)

    remaining_chunk_ids = (input_chunk_ids - processed_chunk_ids)

    remaining_rows = len(remaining_chunk_ids)


    remaining_rows = len(remaining_chunk_ids)

    print(
        f"Chunks already in Neo4j: {len(processed_chunk_ids)}",
        flush=True
    )

    print(
        f"Rows remaining to process: {remaining_rows}",
        flush=True
    )

    print(
        f"Chunks from current dataset already processed: "
        f"{len(processed_in_input)}",
        flush=True
        )

    if remaining_rows <= 0:

        print(
            "All chunks are already present in Neo4j.",
            flush=True
        )

        db.close()

        return

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    processed_this_run = 0
    skipped_this_run = 0

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as f:

        progress_bar = tqdm(
            total=remaining_rows,
            desc="Building knowledge graph",
            unit="row"
        )

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            document_id = chunk["document_id"]
            chunk_id = chunk["chunk_id"]

            # ------------------------------------------------
            # Skip chunks already present in Neo4j
            # ------------------------------------------------

            if chunk_id in processed_chunk_ids:

                skipped_this_run += 1

                continue

            # ------------------------------------------------
            # Create document and chunk
            # ------------------------------------------------

            db.create_document(
                document_id,
                chunk["title"],
                chunk["authors"]
            )

            db.create_chunk(
                chunk_id,
                document_id,
                chunk["text"],
                chunk["chunk_index"]
            )

            # ------------------------------------------------
            # Extract graph using LLM
            # ------------------------------------------------

            prompt = GRAPH_EXTRACTION_PROMPT.format(
                text=chunk["text"]
            )

            raw_output = llm.generate(
                prompt,
                max_new_tokens=768
            )

            graph = llm.extract_json(
                raw_output
            )

            # ------------------------------------------------
            # Process extracted graph
            # ------------------------------------------------

            if graph:

                entities = graph.get(
                    "entities",
                    []
                )

                relations = graph.get(
                    "relations",
                    []
                )

                name_to_id = {}

                # --------------------------------------------
                # Create entities
                # --------------------------------------------

                for entity in entities:

                    name = entity.get(
                        "name",
                        ""
                    ).strip()

                    if not name:
                        continue

                    eid = entity_id(name)

                    name_to_id[name.lower()] = eid

                    db.create_entity(
                        eid,
                        name,
                        entity.get(
                            "type",
                            "Entity"
                        )
                    )

                    db.link_chunk_entity(
                        chunk_id,
                        eid
                    )

                # --------------------------------------------
                # Create relations
                # --------------------------------------------

                for relation in relations:

                    source = relation.get(
                        "source",
                        ""
                    ).strip()

                    target = relation.get(
                        "target",
                        ""
                    ).strip()

                    predicate = relation.get(
                        "relation",
                        "related_to"
                    ).strip()

                    source_id = name_to_id.get(
                        source.lower()
                    )

                    target_id = name_to_id.get(
                        target.lower()
                    )

                    if not source_id or not target_id:
                        continue

                    db.create_relation(
                        source_id,
                        target_id,
                        predicate
                    )

            # ------------------------------------------------
            # Update progress
            # ------------------------------------------------

            processed_chunk_ids.add(chunk_id)

            processed_this_run += 1

            progress_bar.update(1)

            remaining = max(
                progress_bar.total - progress_bar.n,
                0
                )

            progress_bar.set_postfix(
                remaining=remaining
            )

        progress_bar.close()

    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    db.close()

    print(
        f"Processed in this run: {processed_this_run}",
        flush=True
    )

    print(
        f"Skipped existing chunks: {skipped_this_run}",
        flush=True
    )

    print(
        "Knowledge graph construction completed.",
        flush=True
    )


if __name__ == "__main__":
    main()