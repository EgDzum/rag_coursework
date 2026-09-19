#!/usr/bin/env python3

"""
Examples:

    # FAISS only
    python -m retrieval.hybrid_rag \
        --mode naive \
        --question "Your question"

    # FAISS + BM25 + RRF
    python -m retrieval.hybrid_rag \
        --mode hybrid \
        --question "Your question"

If --question is omitted, all unique questions from chunks.jsonl
are evaluated.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from llm.qwen import QwenLLM
from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RRF_K = 60


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> str:
    p = Path(path)
    return str(PROJECT_ROOT / p)


def load_chunks(path: str) -> list[dict[str, Any]]:
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            if not item.get("text"):
                continue

            chunks.append(item)

    return chunks


def tokenize(text: str) -> list[str]:
    """
    Simple deterministic tokenizer for BM25.

    Lowercasing and extracting alphanumeric tokens makes sparse
    retrieval independent of punctuation and Markdown formatting.
    """
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


class HybridRetriever:
    """
    Retriever supporting two modes:

    naive:
        FAISS only.

    hybrid:
        FAISS + BM25 + RRF.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        faiss_store: FAISSStore,
        embedding_model: QwenEmbedding,
        mode: str = "hybrid",
        top_k_faiss: int = 8,
        top_k_bm25: int = 8,
        top_k_final: int = 4,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        if mode not in {"naive", "hybrid"}:
            raise ValueError(
                f"Unsupported retrieval mode: {mode}. "
                f"Use 'naive' or 'hybrid'."
            )

        self.chunks = chunks
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model

        self.mode = mode
        self.top_k_faiss = top_k_faiss
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

        self.bm25 = None

        # BM25 is only needed for hybrid retrieval.
        if self.mode == "hybrid":
            print("Building BM25 index...")

            tokenized_corpus = [
                tokenize(item["text"])
                for item in chunks
            ]

            self.bm25 = BM25Okapi(tokenized_corpus)

            print(
                f"BM25 index built for "
                f"{len(chunks):,} chunks."
            )

        # FAISS metadata should correspond to the same chunk corpus.
        self.chunk_by_id = {
            str(item["chunk_id"]): item
            for item in chunks
        }

    def _faiss_retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.encode([question])

        return self.faiss_store.search(
            query_embedding,
            k=self.top_k_faiss,
        )

    def _bm25_retrieve(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        if self.bm25 is None:
            return []

        query_tokens = tokenize(question)

        scores = self.bm25.get_scores(query_tokens)

        # argpartition avoids sorting the entire corpus.
        k = min(self.top_k_bm25, len(scores))

        if k == 0:
            return []

        candidate_indices = np.argpartition(
            scores,
            -k,
        )[-k:]

        candidate_indices = candidate_indices[
            np.argsort(
                scores[candidate_indices]
            )[::-1]
        ]

        results = []

        for idx in candidate_indices:
            item = dict(self.chunks[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)

        return results

    def _retrieve_naive(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        FAISS-only retrieval.
        """

        faiss_results = self._faiss_retrieve(question)

        final_results = []

        for rank, item in enumerate(
            faiss_results[: self.top_k_final],
            start=1,
        ):
            result = dict(item)

            # In naive mode there is no RRF score.
            result["rrf_score"] = None
            result["retrieval_rank"] = rank
            result["sources"] = ["faiss"]

            final_results.append(result)

        return {
            "faiss": faiss_results,
            "bm25": [],
            "hybrid": final_results,
        }

    def _retrieve_hybrid(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        FAISS + BM25 + Reciprocal Rank Fusion.
        """

        faiss_results = self._faiss_retrieve(question)
        bm25_results = self._bm25_retrieve(question)

        # RRF is rank-based, so original retrieval scores are
        # kept separately for analysis.
        rrf_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        items: dict[str, dict[str, Any]] = {}

        # --------------------------------------------------------
        # FAISS
        # --------------------------------------------------------

        for rank, item in enumerate(
            faiss_results,
            start=1,
        ):
            chunk_id = str(item["chunk_id"])

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            sources.setdefault(
                chunk_id,
                set(),
            ).add("faiss")

            items[chunk_id] = item

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        for rank, item in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = str(item["chunk_id"])

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            sources.setdefault(
                chunk_id,
                set(),
            ).add("bm25")

            items.setdefault(
                chunk_id,
                item,
            )

        # --------------------------------------------------------
        # RRF ranking
        # --------------------------------------------------------

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[: self.top_k_final]

        final_results = []

        for final_rank, chunk_id in enumerate(
            ranked_ids,
            start=1,
        ):
            item = dict(items[chunk_id])

            item["rrf_score"] = float(
                rrf_scores[chunk_id]
            )

            item["retrieval_rank"] = final_rank

            item["sources"] = sorted(
                sources[chunk_id]
            )

            final_results.append(item)

        return {
            "faiss": faiss_results,
            "bm25": bm25_results,
            "hybrid": final_results,
        }

    def retrieve(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Dispatch retrieval according to selected mode.
        """

        if self.mode == "naive":
            return self._retrieve_naive(question)

        return self._retrieve_hybrid(question)


def build_context(
    results: list[dict[str, Any]],
) -> str:
    parts = [
        "=== RETRIEVED PASSAGES ===",
        "",
    ]

    for i, item in enumerate(
        results,
        start=1,
    ):
        rrf_score = item.get("rrf_score")

        if rrf_score is None:
            rrf_score_text = "N/A"
        else:
            rrf_score_text = f"{rrf_score:.6f}"

        parts.append(
            f"""PASSAGE {i}
Title: {item.get("title", "")}
Authors: {item.get("authors", "")}
Retrieval sources: {", ".join(item.get("sources", []))}
RRF score: {rrf_score_text}

{item["text"]}
"""
        )

    return "\n".join(parts)


def build_answer_prompt(
    question: str,
    context: str,
) -> str:
    return f"""You are a retrieval-augmented question answering system.

Answer the question using ONLY the supplied retrieved passages.

Rules:
1. Use only information supported by the passages.
2. Do not use outside knowledge.
3. If the passages do not contain enough information, say that the answer
   cannot be determined from the provided context.
4. Give a concise, direct answer.
5. Do not mention the retrieval system unless necessary.

Question:
{question}

Context:
{context}

Answer:
"""


def get_unique_questions(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    The same QA question can occur in multiple chunks of one document.
    Deduplicate by question while preserving the first reference answer.
    """

    seen = set()
    questions = []

    for item in chunks:
        question = str(
            item.get("question", "")
        ).strip()

        if not question or question in seen:
            continue

        seen.add(question)

        answers = item.get(
            "answers",
            [],
        )

        reference_answer = (
            answers[0]
            if answers
            else ""
        )

        questions.append(
            {
                "question": question,
                "reference_answer": reference_answer,
            }
        )

    return questions


def make_serializable_retrieval(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for item in results:
        output.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "title": item.get("title", ""),
                "authors": item.get("authors", ""),
                "score": item.get("score"),
                "rrf_score": item.get("rrf_score"),
                "retrieval_rank": item.get(
                    "retrieval_rank"
                ),
                "sources": item.get(
                    "sources",
                    [],
                ),
            }
        )

    return output


def answer_question(
    question: str,
    reference_answer: str,
    retriever: HybridRetriever,
    llm: QwenLLM,
    top_k_final: int,
) -> dict[str, Any]:
    start = time.perf_counter()

    retrieval = retriever.retrieve(question)

    context = build_context(
        retrieval["hybrid"]
    )

    prompt = build_answer_prompt(
        question,
        context,
    )

    answer = llm.generate(
        prompt,
        max_new_tokens=llm.max_new_tokens,
    )

    latency = time.perf_counter() - start

    return {
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": answer,

        "retrieved_chunks": make_serializable_retrieval(
            retrieval["hybrid"][:top_k_final]
        ),

        "retrieval_scores": {
            "faiss": [
                {
                    "chunk_id": x["chunk_id"],
                    "score": float(x["score"]),
                }
                for x in retrieval["faiss"]
            ],

            "bm25": [
                {
                    "chunk_id": x["chunk_id"],
                    "score": float(x["score"]),
                }
                for x in retrieval["bm25"]
            ],

            "hybrid_rrf": [
                {
                    "chunk_id": x["chunk_id"],
                    "score": (
                        float(x["rrf_score"])
                        if x.get("rrf_score") is not None
                        else None
                    ),
                }
                for x in retrieval["hybrid"]
            ],
        },

        "mode": retriever.mode,

        "latency": latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAG with selectable retrieval: "
            "FAISS-only or FAISS + BM25 + RRF."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/config.yml",
    )

    parser.add_argument(
        "--question",
        default=None,
        help=(
            "Run one question. If omitted, "
            "evaluate all unique dataset questions."
        ),
    )

    parser.add_argument(
        "--output",
        default="results/rag.jsonl",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "naive",
            "hybrid",
        ],
        default="hybrid",
        help=(
            "Retrieval mode: "
            "'naive' = FAISS only; "
            "'hybrid' = FAISS + BM25 + RRF."
        ),
    )

    parser.add_argument(
        "--top-k-faiss",
        type=int,
        default=8,
        help=(
            "Number of candidates retrieved by FAISS."
        ),
    )

    parser.add_argument(
        "--top-k-bm25",
        type=int,
        default=8,
        help=(
            "Number of candidates retrieved by BM25. "
            "Used only in hybrid mode."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Final number of passages. "
            "In naive mode this is the number of FAISS passages "
            "passed to the LLM."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "RRF constant. Used only in hybrid mode. "
            "Standard value is 60."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_path(
        args.config
    )

    config = load_config(
        config_path
    )

    dataset_path = resolve_path(
        config.get(
            "dataset",
            {},
        ).get(
            "chunks_path",
            "data/processed/chunks.jsonl",
        )
    )

    # The project config currently contains the raw dataset path,
    # but the retrieval corpus is the processed chunks file.
    if not os.path.exists(dataset_path):
        dataset_path = resolve_path(
            "data/processed/chunks.jsonl"
        )

    faiss_cfg = config["faiss"]
    embedding_cfg = config["embedding"]
    llm_cfg = config["llm"]
    retrieval_cfg = config.get(
        "retrieval",
        {},
    )

    faiss_index_path = resolve_path(
        faiss_cfg["index_path"]
    )

    faiss_metadata_path = resolve_path(
        faiss_cfg["metadata_path"]
    )

    top_k_final = (
        args.top_k
        or retrieval_cfg.get(
            "top_k_vector",
            4,
        )
    )

    chunks = load_chunks(
        dataset_path
    )

    print(
        f"Loaded {len(chunks):,} chunks."
    )

    # ------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------

    faiss_store = FAISSStore(
        index_path=faiss_index_path,
        metadata_path=faiss_metadata_path,
    )

    faiss_store.load()

    print(
        f"Loaded FAISS index with "
        f"{faiss_store.index.ntotal:,} vectors."
    )

    # ------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------

    embedding_model = QwenEmbedding(
        model_name=embedding_cfg["model"],
        batch_size=embedding_cfg.get(
            "batch_size",
            4,
        ),
        normalize=embedding_cfg.get(
            "normalize",
            True,
        ),
        max_length=embedding_cfg.get(
            "max_length",
            2048,
        ),
    )

    # ------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------

    retriever = HybridRetriever(
        chunks=chunks,
        faiss_store=faiss_store,
        embedding_model=embedding_model,
        mode=args.mode,
        top_k_faiss=args.top_k_faiss,
        top_k_bm25=args.top_k_bm25,
        top_k_final=top_k_final,
        rrf_k=args.rrf_k,
    )

    # ------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------

    llm = QwenLLM(
        model_name=llm_cfg["model"],
        max_new_tokens=llm_cfg.get(
            "max_new_tokens",
            1024,
        ),
        max_input_tokens=llm_cfg.get(
            "max_input_tokens",
            3072,
        ),
        temperature=llm_cfg.get(
            "temperature",
            0.0,
        ),
    )

    # ------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------

    if args.question:
        questions = [
            {
                "question": args.question,
                "reference_answer": "",
            }
        ]
    else:
        questions = get_unique_questions(
            chunks
        )

    print(
        f"Retrieval mode: {args.mode}"
    )

    print(
        f"Questions to process: "
        f"{len(questions):,}"
    )

    print("")

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    output_path = resolve_path(
        args.output
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out:

        for i, qa in enumerate(
            questions,
            start=1,
        ):
            print(
                f"[{i}/{len(questions)}] "
                f"{qa['question'][:120]}"
            )

            result = answer_question(
                question=qa["question"],
                reference_answer=qa[
                    "reference_answer"
                ],
                retriever=retriever,
                llm=llm,
                top_k_final=top_k_final,
            )

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            print(
                f"  mode={result['mode']} "
                f"latency={result['latency']:.2f}s"
            )

    print("")
    print("==========================================")
    print("Finished")
    print("==========================================")
    print(
        f"Mode:    {args.mode}"
    )
    print(
        f"Results: {output_path}"
    )


if __name__ == "__main__":
    main()