#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple
import os

import yaml
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT.parent / "configs" / "config.yml"

RELATION_LABELS = [
    "associated with",
    "related to",
    "combined with",
    "part of",
    "type of",
    "affects",
    "influences",
    "dependson",
    "used for",
    "causes",
    "increases",
    "decreases",
    "produces",
    "requires",
    "supports",
    "prevents",
    "contains",
    "located in",
    "applies to",
    "interacts with",
]
DEFAULT_BATCH_SIZE = 5000
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add random noisy RELATES_TO relationships to Neo4j."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=(
            f"Path to config.yml "
            f"(default: {DEFAULT_CONFIG_PATH})"
        ),
    )

    parser.add_argument(
        "--noise-percent",
        type=float,
        required=True,
        help=(
            "Percentage of the original RELATES_TO relationships "
            "to add as noise. Example: 5 means 5%%."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducibility "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Number of relationships written per transaction "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Calculate the amount of noise and generate candidate "
            "relationships, but do not modify Neo4j."
        ),
    )

    parser.add_argument(
        "--allow-existing-noise",
        action="store_true",
        help=(
            "Allow the script to run when NOISE relationships already "
            "exist. By default the script stops to avoid accidentally "
            "adding noise on top of an already noisy graph."
        ),
    )

    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """
    Load project configuration from config.yml.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise RuntimeError(
            f"Config file is empty: {path}"
        )

    return config


def get_neo4j_config(config: dict) -> Tuple[str, str, str, str]:
    """
    Extract Neo4j connection parameters.

    The password can also be supplied through the NEO4J_PASSWORD
    environment variable.
    """
    neo4j_config = config.get("neo4j", {})

    uri = neo4j_config.get(
        "uri",
        "bolt://localhost:7688",
    )

    user = neo4j_config.get(
        "user",
        "neo4j",
    )

    database = neo4j_config.get(
        "database",
        "neo4j",
    )

    password = os.environ.get("NEO4J_PASSWORD")

    if password is None:
        password = neo4j_config.get("password")

    if password is None:
        raise RuntimeError(
            "Neo4j password was not found. "
            "Set NEO4J_PASSWORD or provide password in config.yml."
        )

    return uri, user, password, database

class Neo4jNoiseInjector:
    """
    Neo4j client used specifically for graph-noise experiments.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
    ):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )

        self.database = database

    def close(self):
        self.driver.close()

    def verify_connection(self):
        """
        Verify that Neo4j is reachable.
        """
        with self.driver.session(
            database=self.database
        ) as session:
            session.run("RETURN 1").consume()

    def get_entity_count(self) -> int:
        """
        Count Entity nodes.
        """
        query = """
        MATCH (e:Entity)
        RETURN count(e) AS count
        """

        with self.driver.session(
            database=self.database
        ) as session:
            record = session.run(query).single()

        return int(record["count"])

    def get_relates_to_count(self) -> int:
        """
        Count original relationships, excluding injected noise.
        Supports both legacy type='NOISE' edges and new is_noise markers.
        """
        query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE coalesce(r.is_noise, false) = false
        AND coalesce(r.type, '') <> 'NOISE'
        RETURN count(r) AS count
        """

        with self.driver.session(database=self.database) as session:
            record = session.run(query).single()

        return int(record["count"])

    def get_noise_count(self) -> int:
        """
        Count both legacy noise relationships and relationships
        injected by the updated script.
        """
        query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE coalesce(r.is_noise, false) = true
        OR r.type = 'NOISE'
        RETURN count(r) AS count
        """

        with self.driver.session(database=self.database) as session:
            record = session.run(query).single()

        return int(record["count"])

    def get_entity_ids(self) -> List[int]:
        """
        Get Neo4j internal IDs for all Entity nodes.
        """
        query = """
        MATCH (e:Entity)
        RETURN id(e) AS entity_id
        ORDER BY id(e)
        """

        with self.driver.session(
            database=self.database
        ) as session:
            result = session.run(query)

            return [
                int(record["entity_id"])
                for record in result
            ]

    def get_existing_entity_pairs(self) -> set:
        """
        Retrieve all directed Entity -> Entity pairs connected by
        RELATES_TO.

        A pair is represented as:

            (source_id, target_id)

        Existing relationships are excluded when generating noise.
        """
        query = """
        MATCH (source:Entity)-[:RELATES_TO]->(target:Entity)
        RETURN id(source) AS source_id,
               id(target) AS target_id
        """

        with self.driver.session(
            database=self.database
        ) as session:
            result = session.run(query)

            return {
                (
                    int(record["source_id"]),
                    int(record["target_id"]),
                )
                for record in result
            }

    def add_noise_batch(self, relationships: List[Tuple[int, int, str]]) -> int:
        """
        Add noisy RELATES_TO relationships with randomly assigned
        semantic-looking predicates.

        Each relationship is marked with is_noise=true so that noise
        can be counted separately from original graph relationships.
        """
        query = """
        UNWIND $relationships AS rel

        MATCH (source:Entity)
        WHERE id(source) = rel.source_id

        MATCH (target:Entity)
        WHERE id(target) = rel.target_id
        AND source <> target
        AND NOT EXISTS {
            MATCH (source)-[:RELATES_TO]->(target)
        }

        CREATE (source)-[:RELATES_TO {
            type: rel.relation_type,
            is_noise: true
        }]->(target)

        RETURN count(*) AS created
        """

        params = {
            "relationships": [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                }
                for source_id, target_id, relation_type in relationships
            ]
        }

        with self.driver.session(database=self.database) as session:
            record = session.run(query, params).single()

        return int(record["created"])


def generate_noise_edges(
    entity_ids: List[int],
    number_of_edges: int,
    existing_pairs: set,
    seed: int,
) -> List[Tuple[int, int, str]]:
    """
    Generate unique random directed Entity -> Entity pairs.

    Each generated edge receives a randomly selected predicate.
    """
    if number_of_edges <= 0:
        return []

    if len(entity_ids) < 2:
        raise RuntimeError(
            "At least two Entity nodes are required."
        )

    rng = random.Random(seed)
    generated = set()

    total_possible_pairs = len(entity_ids) * (len(entity_ids) - 1)
    available_pairs = total_possible_pairs - len(existing_pairs)

    if number_of_edges > available_pairs:
        raise RuntimeError(
            "Cannot generate the requested amount of noise.\n"
            f"Requested:       {number_of_edges:,}\n"
            f"Available pairs: {available_pairs:,}\n"
            f"Entities:        {len(entity_ids):,}\n"
            "The requested noise level is too high for the number "
            "of available Entity pairs."
        )

    while len(generated) < number_of_edges:
        source_id = rng.choice(entity_ids)
        target_id = rng.choice(entity_ids)

        if source_id == target_id:
            continue

        pair = (source_id, target_id)

        if pair in existing_pairs or pair in generated:
            continue

        generated.add(pair)

    # Assign predicates after generating pairs.
    # Sorting ensures stable ordering for a fixed seed.
    return [
        (
            source_id,
            target_id,
            rng.choice(RELATION_LABELS),
        )
        for source_id, target_id in sorted(generated)
    ]

def main():
    args = parse_args()

    if args.noise_percent < 0:
        raise ValueError(
            "--noise-percent must be >= 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    print("=" * 72, flush=True)
    print("Graph Noise Injection", flush=True)
    print("=" * 72, flush=True)

    print(f"Config:          {args.config}", flush=True)
    print(f"Noise level:     {args.noise_percent}%", flush=True)
    print(f"Random seed:     {args.seed}", flush=True)
    print(f"Batch size:      {args.batch_size}", flush=True)
    print(f"Dry run:         {args.dry_run}", flush=True)
    print(flush=True)

    config = load_config(args.config)

    uri, user, password, database = get_neo4j_config(
        config
    )

    print(f"Neo4j URI:       {uri}", flush=True)
    print(f"Neo4j database:  {database}", flush=True)
    print(flush=True)

    injector = Neo4jNoiseInjector(
        uri=uri,
        user=user,
        password=password,
        database=database,
    )

    try:
        print("Checking Neo4j connection...", flush=True)

        injector.verify_connection()

        print("Neo4j connection OK.", flush=True)
        print(flush=True)

        entity_count = injector.get_entity_count()
        relates_to_count = injector.get_relates_to_count()
        noise_count = injector.get_noise_count()

        print("Current graph:", flush=True)
        print(f"  Entity nodes:             {entity_count:,}", flush=True)
        print(f"  RELATES_TO relationships: {relates_to_count:,}", flush=True)
        print(f"  Existing NOISE edges:     {noise_count:,}", flush=True)
        print(flush=True)

        if noise_count > 0 and not args.allow_existing_noise:
            raise RuntimeError(
                "The graph already contains "
                f"{noise_count:,} NOISE relationships.\n\n"
                "The script stopped to prevent accidentally adding "
                "noise on top of an already noisy graph.\n\n"
                "Use --allow-existing-noise only if this is intentional."
            )

        number_of_noise_edges = int(
            relates_to_count
            * args.noise_percent
            / 100.0
        )

        print("Noise calculation:", flush=True)
        print(
            f"  Original RELATES_TO:     {relates_to_count:,}",
            flush=True,
        )
        print(
            f"  Requested noise:         {args.noise_percent}%",
            flush=True,
        )
        print(
            f"  Edges to add:            {number_of_noise_edges:,}",
            flush=True,
        )
        print(flush=True)

        if number_of_noise_edges == 0:
            print(
                "Noise level results in zero additional edges.",
                flush=True,
            )
            return

        print("Loading Entity IDs...", flush=True)

        entity_ids = injector.get_entity_ids()

        print(
            f"Loaded {len(entity_ids):,} Entity nodes.",
            flush=True,
        )
        print(flush=True)

        print(
            "Loading existing Entity -> Entity pairs...",
            flush=True,
        )

        existing_pairs = (
            injector.get_existing_entity_pairs()
        )

        print(
            f"Loaded {len(existing_pairs):,} existing pairs.",
            flush=True,
        )
        print(flush=True)

        print("Generating random noise edges...", flush=True)

        noise_edges = generate_noise_edges(
            entity_ids=entity_ids,
            number_of_edges=number_of_noise_edges,
            existing_pairs=existing_pairs,
            seed=args.seed,
        )

        print(
            f"Generated {len(noise_edges):,} candidate edges.",
            flush=True,
        )
        print(flush=True)

        if args.dry_run:
            print("=" * 72, flush=True)
            print("DRY RUN", flush=True)
            print("=" * 72, flush=True)
            print(
                "No changes were made to Neo4j.",
                flush=True,
            )
            print(flush=True)
            print(
                f"Would add: {len(noise_edges):,} "
                "RELATES_TO relationships with random predicates.",
                flush=True,
            )

            print("Sample candidate edges:", flush=True)

            for source_id, target_id, relation_type in noise_edges[:10]:
                print(
                    f"  {source_id} -[{relation_type}]-> {target_id}",
                    flush=True,
                )
            return


        print("Writing noise to Neo4j...", flush=True)
        print(flush=True)

        total_created = 0

        for start in range(
            0,
            len(noise_edges),
            args.batch_size,
        ):
            batch = noise_edges[
                start:start + args.batch_size
            ]

            created = injector.add_noise_batch(
                batch
            )

            total_created += created

            print(
                f"Progress: "
                f"{min(start + len(batch), len(noise_edges)):,}"
                f"/{len(noise_edges):,} "
                f"candidates processed; "
                f"{total_created:,} created",
                flush=True,
            )

        final_relates_to_count = (
            injector.get_relates_to_count()
        )

        final_noise_count = (
            injector.get_noise_count()
        )

        print(flush=True)
        print("=" * 72, flush=True)
        print("Noise injection completed", flush=True)
        print("=" * 72, flush=True)

        print(
            f"Original RELATES_TO:       "
            f"{relates_to_count:,}",
            flush=True,
        )

        print(
            f"Requested noise:            "
            f"{number_of_noise_edges:,}",
            flush=True,
        )

        print(
            f"Actually created:           "
            f"{total_created:,}",
            flush=True,
        )

        print(
            f"Final RELATES_TO:           "
            f"{final_relates_to_count:,}",
            flush=True,
        )

        print(
            f"Total NOISE relationships:  "
            f"{final_noise_count:,}",
            flush=True,
        )

        if relates_to_count > 0:
            actual_noise_percent = (
                total_created
                / relates_to_count
                * 100.0
            )
        else:
            actual_noise_percent = 0.0

        print(
            f"Actual added noise level:   "
            f"{actual_noise_percent:.4f}%",
            flush=True,
        )

        print(flush=True)
        print("Noise relationships use:", flush=True)
        print(
            '  (:Entity)-[:RELATES_TO '
            '{type: "<predicate>", is_noise: true}]->(:Entity)',
            flush=True,
        )
        print(
            "Predicates look semantically meaningful, but the edges "
            "are randomly generated.",
            flush=True,
        )

    except Exception as exc:
        print(flush=True)
        print("=" * 72, flush=True)
        print("ERROR", flush=True)
        print("=" * 72, flush=True)
        print(str(exc), flush=True)
        print(flush=True)
        raise

    finally:
        injector.close()


if __name__ == "__main__":
    main()