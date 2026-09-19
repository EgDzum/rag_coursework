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

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(embeddings)

        self.index = index
        self.metadata = metadata

    def save(self):

        os.makedirs(
            os.path.dirname(self.index_path),
            exist_ok=True
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

    def search(
        self,
        query_embedding,
        k=8
    ):

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