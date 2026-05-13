import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data" / "docs"

CHROMA_DIR = Path("chroma_store")
COLLECTION_NAME = "orbitops_assignment"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any]


def parse_metadata_and_text(raw_text: str) -> dict:
    """Parse the metadata header and body from a document string.

    Returns a dictionary with keys:
    doc_id, title, date, team, tags, security_clearance, text
    """
    metadata = {}
    if "\n\n" not in raw_text:
        metadata["text"] = raw_text
        return metadata
    metatext, text = raw_text.split("\n\n", 1)

    for line in metatext.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalised_key = key.strip().lower()

        if normalised_key == "doc id":
            metadata["doc_id"] = value.strip().lower()
        elif normalised_key == "date":
            date_match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
            if date_match:
                year, month, day = date_match.groups()
                metadata["date"] = f"{year}-{int(month):02d}-{int(day):02d}"
            else:
                metadata["date"] = value.strip()
        elif normalised_key == "security clearance":
            metadata["security_clearance"] = value.strip().upper()
        else:
            metadata[normalised_key] = value.strip()

    metadata["text"] = text
    return metadata


def load_documents(data_dir: Path) -> list[Document]:
    """Load and parse all .txt files from data_dir into Document objects."""
    documents = []
    for file in sorted(data_dir.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        metadata = parse_metadata_and_text(text)
        metadata["source_file"] = file.name
        documents.append(Document(metadata["text"], metadata))
    return documents


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def chunk_text(text: str, chunk_size: int = 180, chunk_overlap: int = 40) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: List[str] = []
    words = text.split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        if chunk:
            chunks.append(" ".join(chunk))
        start += chunk_size - chunk_overlap
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 180, chunk_overlap: int = 40
) -> list[Document]:
    """Convert Document objects into overlapping chunk Documents.

    Each returned Document has metadata keys:
    doc_id, title, source_file, chunk_id, chunk_size, chunk_overlap,
    date, team, tags, security_clearance
    """
    chunked_docs = []
    for document in documents:
        cleaned = clean_text(document.text)
        chunks = chunk_text(cleaned, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            metadata = dict(document.metadata)
            metadata["chunk_id"] = f"{metadata['doc_id']}_chunk_{i}"
            metadata["chunk_size"] = chunk_size
            metadata["chunk_overlap"] = chunk_overlap
            metadata.pop("text", None)
            chunked_docs.append(Document(chunk, metadata))
    return chunked_docs


def build_dense_index(chunks: list[Document], model: SentenceTransformer) -> np.ndarray:
    """Encode chunk texts and return a 2-D numpy array of L2-normalised embeddings."""
    texts = [chunk.text for chunk in chunks]
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def dense_retrieve(
    query: str,
    chunks: list[Document],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[Document]:
    """Return the top_k chunks most similar to query by cosine similarity."""
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        Document(
            text=chunks[i].text,
            metadata={**chunks[i].metadata, "dense_score": float(scores[i])},
        )
        for i in top_indices
    ]


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    embeddings = build_dense_index(chunks, embed_model)
    print(f"Built index: {embeddings.shape}")
