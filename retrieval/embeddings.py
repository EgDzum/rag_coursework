import torch

from sentence_transformers import SentenceTransformer


class QwenEmbedding:

    def __init__(
        self,
        model_name: str,
        batch_size: int = 4,
        normalize: bool = True,
        max_length: int = 512
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.batch_size = batch_size
        self.normalize = normalize

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        self.model.max_seq_length = max_length

    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True
            )
        return embeddings