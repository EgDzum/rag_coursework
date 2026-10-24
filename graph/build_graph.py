import json
import os
import re
import hashlib

from tqdm import tqdm


INPUT_PATH = "data/raw/agriculture.jsonl"
OUTPUT_PATH = "$SCRATCH/index_data/processed/agriculture_chunks.jsonl"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def make_chunks(text: str, chunk_size=1200, overlap=150):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        chunks.append({
            "text": chunk,
            "start_word": start,
            "end_word": end
        })

        if end == len(words):
            break

        start = end - overlap

    return chunks


def stable_id(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:32]


def process_dataset():

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):

            row = json.loads(line)

            question = normalize_text(
                row.get("input", "")
            )

            context = normalize_text(
                row.get("context", "")
            )

            answers = row.get("answers", [])

            if not isinstance(answers, list):
                answers = [answers]

            meta = row.get("meta") or {}

            title = meta.get("title", "")
            authors = meta.get("authors", "")

            document_id = (
                row.get("context_id")
                or row.get("_id")
                or stable_id(context)
            )

            chunks = make_chunks(
                context,
                CHUNK_SIZE,
                CHUNK_OVERLAP
            )

            for i, chunk in enumerate(chunks):

                chunk_id = f"{document_id}_{i}"

                output = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,

                    "question": question,
                    "answers": answers,

                    "text": chunk["text"],

                    "title": title,
                    "authors": authors,

                    "label": row.get(
                        "label",
                        "agriculture"
                    ),

                    "chunk_index": i,
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"]
                }

                fout.write(
                    json.dumps(
                        output,
                        ensure_ascii=False
                    ) + "\n"
                )


if __name__ == "__main__":
    process_dataset()