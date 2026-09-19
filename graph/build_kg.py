import hashlib
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from graph.neo4j_client import Neo4jClient
from llm.qwen import QwenLLM
from rag.prompts import GRAPH_EXTRACTION_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT.parent / "configs" / "config.yml"
DATA_PATH = "data/processed/chunks.jsonl"


def entity_id(name):

    return hashlib.sha256(
        name.lower().strip().encode()
    ).hexdigest()[:32]


def main():

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    neo4j_config = config["neo4j"]

    db = Neo4jClient(
        neo4j_config["uri"],
        neo4j_config["username"],
        neo4j_config["password"],
        neo4j_config["database"]
    )

    db.create_schema()

    llm = QwenLLM(
        config["llm"]["model"],
        config["llm"]["max_input_tokens"]
    )

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in tqdm(f):

            chunk = json.loads(line)

            document_id = chunk["document_id"]
            chunk_id = chunk["chunk_id"]

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

            prompt = GRAPH_EXTRACTION_PROMPT.format(
                text=chunk["text"]
            )

            raw_output = llm.generate(
                prompt,
                max_new_tokens=768,
                temperature=0.0
            )

            graph = llm.extract_json(
                raw_output
            )

            if not graph:
                continue

            entities = graph.get(
                "entities",
                []
            )

            relations = graph.get(
                "relations",
                []
            )

            name_to_id = {}

            for entity in entities:

                name = entity.get("name", "").strip()

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

    db.close()


if __name__ == "__main__":
    main()