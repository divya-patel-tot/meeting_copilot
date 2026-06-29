from __future__ import annotations

from pathlib import Path

from app.core.rag.chunking import chunk_text
from app.core.rag.embedder import Embedder
from app.core.rag.ingestion import parse_docx, parse_pdf
from app.core.rag.vectorstore import VectorStore


class KnowledgeBase:
    """Ingest documents and query the persistent vector store."""

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._store = VectorStore()

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = parse_pdf(str(path))
        elif suffix == ".docx":
            text = parse_docx(str(path))
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks = chunk_text(text)
        if not chunks:
            return 0

        source = path.name
        self._store.delete_by_source(source)

        ids = [f"{source}::{index}" for index in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": index}
            for index in range(len(chunks))
        ]
        embeddings = self._embedder.embed(chunks)
        self._store.add_documents(
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query_candidates(self, text: str, top_k: int = 4) -> list[dict]:
        """Return top_k matches ranked by score (higher = better), no threshold filter."""
        embedding = self._embedder.embed([text])[0]
        results = self._store.query(embedding, top_k=top_k)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches: list[dict] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + float(distance))
            matches.append(
                {
                    "text": doc,
                    "source": (meta or {}).get("source", "unknown"),
                    "chunk_index": (meta or {}).get("chunk_index"),
                    "score": score,
                    "distance": float(distance),
                }
            )
        return matches

    def query(
        self,
        text: str,
        top_k: int = 4,
        relevance_threshold: float = 0.55,
    ) -> list[dict]:
        """Return chunks with score >= relevance_threshold (higher score = better match)."""
        return [
            match
            for match in self.query_candidates(text, top_k=top_k)
            if match["score"] >= relevance_threshold
        ]

    def get_stats(self) -> dict:
        return {
            "total_chunks": self._store.count(),
            "sources": self._store.list_sources(),
        }

    def clear(self) -> None:
        self._store.clear()
