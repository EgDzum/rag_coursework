import json
import os

import faiss
import numpy as np


class FAISSStore:

    def __init__(
        self,
        index_path,
        metadata_path
    ):

        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

    def build(
        self,
        embeddings,
        metadata
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match "
                "number of metadata records"
            )

        if len(embeddings) == 0:
            raise ValueError(
                "Cannot build an index with zero embeddings"
            )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(embeddings)

        self.index = index
        self.metadata = list(metadata)

    def append(
        self,
        embeddings,
        metadata
    ):

        if self.index is None:
            raise RuntimeError(
                "Index is not loaded. "
                "Call load() before append()."
            )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array"
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match "
                "number of metadata records"
            )

        if len(embeddings) == 0:
            print("No embeddings to append")
            return

        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"new={embeddings.shape[1]}, "
                f"index={self.index.d}"
            )

        existing_ids = {
            item["chunk_id"]
            for item in self.metadata
            if "chunk_id" in item
        }

        new_metadata = []
        new_embeddings = []

        duplicate_count = 0

        for embedding, item in zip(
            embeddings,
            metadata
        ):

            chunk_id = item.get("chunk_id")

            if chunk_id in existing_ids:
                duplicate_count += 1
                continue

            existing_ids.add(chunk_id)

            new_embeddings.append(embedding)
            new_metadata.append(item)

        if not new_embeddings:
            print(
                "No new chunks to append. "
                f"Duplicates skipped: {duplicate_count}"
            )
            return

        new_embeddings = np.asarray(
            new_embeddings,
            dtype="float32"
        )

        self.index.add(
            new_embeddings
        )

        self.metadata.extend(
            new_metadata
        )

        print(
            f"Added vectors: {len(new_metadata)}"
        )

        print(
            f"Duplicates skipped: {duplicate_count}"
        )

        print(
            f"Total vectors: {self.index.ntotal}"
        )

        print(
            f"Total metadata records: {len(self.metadata)}"
        )

    def save(self):

        if self.index is None:
            raise RuntimeError(
                "Cannot save an empty index"
            )

        metadata_dir = os.path.dirname(
            self.metadata_path
        )

        index_dir = os.path.dirname(
            self.index_path
        )

        if index_dir:
            os.makedirs(
                index_dir,
                exist_ok=True
            )

        if metadata_dir:
            os.makedirs(
                metadata_dir,
                exist_ok=True
            )

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "Index and metadata size mismatch: "
                f"vectors={self.index.ntotal}, "
                f"metadata={len(self.metadata)}"
            )

        faiss.write_index(
            self.index,
            self.index_path
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

    def load(self):

        self.index = faiss.read_index(
            self.index_path
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise RuntimeError(
                "Index and metadata size mismatch: "
                f"vectors={self.index.ntotal}, "
                f"metadata={len(self.metadata)}"
            )

    def search(
        self,
        query_embedding,
        k=8
    ):

        if self.index is None:
            raise RuntimeError(
                "Index is not loaded"
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            item = dict(
                self.metadata[idx]
            )

            item["score"] = float(score)

            results.append(item)

        return results