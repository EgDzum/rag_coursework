"""
Corpus contamination for FAISS index.

Script:
  1. Reads clean chunks (data/processed/chunks.jsonl).
  2. Applies selected noise types.
  3. (Optionally) adds injection chunks from another file.
  4. Saves noisy chunks to data/processed/chunks_noisy.jsonl.
  5. Rebuilds FAISS index in indexes/faiss_noisy/.

Important: original index.faiss / metadata.json are NOT overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml
from tqdm import tqdm
import torch 

from retrieval.embeddings import QwenEmbedding
from retrieval.vector_store import FAISSStore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("noise_faiss")



SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text]


def join_sentences(sents: Iterable[str]) -> str:
    return " ".join(s.strip() for s in sents if s.strip())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class NoiseConfig:
    # Which noise types to enable. Empty = do nothing.
    types: List[str] = field(default_factory=lambda: ["delete"])

    # delete / truncate: fraction of sentences to remove/keep
    delete_ratio: float = 0.05          # fraction of sentences to delete
    truncate_keep_ratio: float = 0.9   # fraction to keep in truncate

    # inject: path to jsonl with foreign chunks and how many to add
    inject_path: Optional[str] = None
    inject_ratio: float = 0.1          # add N_extra = int(len(corpus) * ratio)

    # word_swap / char_noise: fraction of words/chars to distort
    word_swap_ratio: float = 0.01
    char_noise_ratio: float = 0.009

    # duplicate: fraction of chunks to duplicate with distortion
    duplicate_ratio: float = 0.05

    # overall probability to apply noise to a specific chunk
    apply_prob: float = 0.5

    # seed
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NoiseConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


def op_delete(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    n_drop = max(1, int(len(sents) * cfg.delete_ratio))
    idx_to_drop = set(rng.sample(range(len(sents)), min(n_drop, len(sents) - 1)))
    kept = [s for i, s in enumerate(sents) if i not in idx_to_drop]
    return join_sentences(kept)


def op_truncate(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    keep = max(1, int(len(sents) * cfg.truncate_keep_ratio))
    # keep a random continuous window (emulates context loss)
    start = rng.randint(0, max(0, len(sents) - keep))
    return join_sentences(sents[start:start + keep])


def op_shuffle_sentences(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return text
    rng.shuffle(sents)
    return join_sentences(sents)


def op_word_swap(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    n = max(1, int(len(words) * cfg.word_swap_ratio))
    idxs = rng.sample(range(len(words)), min(n, len(words)))
    for i in idxs:
        # with probability 0.5 � drop the word, otherwise corrupt it
        if rng.random() < 0.5:
            words[i] = ""
        else:
            w = words[i]
            if len(w) > 3:
                j = rng.randint(0, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(w for w in words if w)


def op_char_noise(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    if not text:
        return text
    chars = list(text)
    n = max(1, int(len(chars) * cfg.char_noise_ratio))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(n):
        i = rng.randrange(len(chars))
        action = rng.choice(["sub", "ins", "del"])
        if action == "sub":
            chars[i] = rng.choice(alphabet)
        elif action == "ins":
            chars.insert(i, rng.choice(alphabet))
        else:
            if chars:
                chars.pop(i)
    return "".join(chars)


OP_MAP = {
    "delete": op_delete,
    "truncate": op_truncate,
    "shuffle_sentences": op_shuffle_sentences,
    "word_swap": op_word_swap,
    "char_noise": op_char_noise,
}


def apply_ops(text: str, cfg: NoiseConfig, rng: random.Random) -> str:
    if rng.random() > cfg.apply_prob:
        return text
    out = text
    for name in cfg.types:
        if name in OP_MAP:
            out = OP_MAP[name](out, cfg, rng)
    return out


def build_noisy_corpus(
    chunks: List[Dict[str, Any]],
    cfg: NoiseConfig,
) -> List[Dict[str, Any]]:
    rng = random.Random(cfg.seed)
    noisy: List[Dict[str, Any]] = []

    for row in tqdm(chunks, desc="noising chunks"):
        new_row = dict(row)
        orig_text = row.get("text", "") or ""
        new_row["text"] = apply_ops(orig_text, cfg, rng)
        new_row["noise_types"] = list(cfg.types)
        new_row["is_synthetic"] = False
        noisy.append(new_row)

    if "inject" in cfg.types:
        if not cfg.inject_path:
            log.warning("inject is specified in types, but inject_path is not set")
        else:
            inject_path = Path(cfg.inject_path)
            if not inject_path.exists():
                log.warning("inject_path %s not found", inject_path)
            else:
                extra_pool = load_jsonl(inject_path)
                n_extra = max(1, int(len(chunks) * cfg.inject_ratio))
                log.info("Inject: adding %d chunks from %s", n_extra, inject_path)
                for i in range(n_extra):
                    src = rng.choice(extra_pool)
                    new_row = dict(src)
                    # ensure unique id
                    base_id = str(src.get("chunk_id", f"inject-{i}"))
                    new_row["chunk_id"] = f"inject-{i}-{base_id}"
                    new_row["is_synthetic"] = True
                    new_row["noise_types"] = ["inject"]
                    noisy.append(new_row)

    if "duplicate" in cfg.types:
        n_dup = int(len(chunks) * cfg.duplicate_ratio)
        if n_dup > 0:
            log.info("Duplicate: creating %d distorted copies", n_dup)
        for i in range(n_dup):
            src = rng.choice(noisy)
            new_row = dict(src)
            new_row["chunk_id"] = f"dup-{i}-{src.get('chunk_id', i)}"
            # slight distortion
            new_row["text"] = op_word_swap(src.get("text", ""), cfg, rng)
            new_row["is_synthetic"] = True
            new_row["noise_types"] = list(cfg.types) + ["duplicate"]
            noisy.append(new_row)

    rng.shuffle(noisy)
    return noisy


def rebuild_faiss(
    chunks: List[Dict[str, Any]],
    out_dir: Path,
    batch_size: int = 32,
) -> None:
    """Rebuilds FAISS index from noisy chunks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "10_noisy_index.faiss"
    metadata_path = out_dir / "10_metadata_noisy.json"

    for p in (index_path, metadata_path):
        if p.exists() and p.is_dir():
            raise IsADirectoryError(
                f"{p} is a directory, expected a file. "
                f"Remove it first: rm -rf {p}"
            )

    log.info("Initializing QwenEmbedding...")
    embedder = QwenEmbedding("Qwen/Qwen3-Embedding-0.6B")

    texts = [c.get("text", "") for c in chunks]
    log.info("Encoding %d chunks...", len(texts))

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
        batch = texts[i:i + batch_size]
        emb = embedder.encode(batch)
        vectors.append(emb)

    matrix = np.vstack(vectors).astype("float32")
    log.info("Embedding matrix: %s", matrix.shape)

    store = FAISSStore(index_path=str(index_path), metadata_path=str(metadata_path))
    store.build(matrix, chunks)
    store.save()
    log.info("FAISS index saved")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yml")
    parser.add_argument(
        "--chunks",
        default=None,
        help="Path to clean chunks.jsonl (default taken from config.yml).",
    )
    parser.add_argument(
        "--out-chunks",
        default="data/processed/chunks_noisy.jsonl",
    )
    parser.add_argument(
        "--out-index",
        default="indexes/faiss_noisy",
    )
    parser.add_argument(
        "--noise-types",
        nargs="+",
        default=None,
        help="List of types: delete truncate shuffle_sentences word_swap char_noise inject duplicate",
    )
    parser.add_argument("--inject-path", default=None)
    parser.add_argument("--inject-ratio", type=float, default=None)
    parser.add_argument("--delete-ratio", type=float, default=None)
    parser.add_argument("--truncate-keep-ratio", type=float, default=None)
    parser.add_argument("--word-swap-ratio", type=float, default=None)
    parser.add_argument("--char-noise-ratio", type=float, default=None)
    parser.add_argument("--duplicate-ratio", type=float, default=None)
    parser.add_argument("--apply-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
    else:
        full_cfg = {}

    chunks_path = args.chunks or (
        full_cfg.get("dataset", {}).get("chunks_path")
        or "data/processed/chunks.jsonl"
    )

    noise_cfg_dict = dict(full_cfg.get("noise", {}) or {})

    if args.noise_types is not None:
        noise_cfg_dict["types"] = args.noise_types
    if args.inject_path is not None:
        noise_cfg_dict["inject_path"] = args.inject_path
    if args.inject_ratio is not None:
        noise_cfg_dict["inject_ratio"] = args.inject_ratio
    if args.delete_ratio is not None:
        noise_cfg_dict["delete_ratio"] = args.delete_ratio
    if args.truncate_keep_ratio is not None:
        noise_cfg_dict["truncate_keep_ratio"] = args.truncate_keep_ratio
    if args.word_swap_ratio is not None:
        noise_cfg_dict["word_swap_ratio"] = args.word_swap_ratio
    if args.char_noise_ratio is not None:
        noise_cfg_dict["char_noise_ratio"] = args.char_noise_ratio
    if args.duplicate_ratio is not None:
        noise_cfg_dict["duplicate_ratio"] = args.duplicate_ratio
    if args.apply_prob is not None:
        noise_cfg_dict["apply_prob"] = args.apply_prob
    if args.seed is not None:
        noise_cfg_dict["seed"] = args.seed

    cfg = NoiseConfig.from_dict(noise_cfg_dict)
    log.info("Noise config: %s", cfg)

    chunks_file = Path(chunks_path)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    log.info("Reading clean chunks: %s", chunks_file)
    chunks = load_jsonl(chunks_file)
    log.info("Total chunks: %d", len(chunks))

    noisy = build_noisy_corpus(chunks, cfg)
    log.info("Final corpus: %d chunks", len(noisy))

    out_chunks = Path(args.out_chunks)
    save_jsonl(noisy, out_chunks)
    log.info("Noisy chunks saved: %s", out_chunks)

    rebuild_faiss(noisy, Path(args.out_index))


if __name__ == "__main__":
    main()