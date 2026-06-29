import chromadb
from chromadb.config import Settings

from app.core.rag.chroma_telemetry import NoOpTelemetry
from app.utils.paths import KB_DIR

_CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    chroma_product_telemetry_impl=f"{NoOpTelemetry.__module__}.{NoOpTelemetry.__name__}",
)


class VectorStore:
    """Persistent ChromaDB vector store for knowledge base documents."""

    def __init__(self, collection_name: str = "documents") -> None:
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(KB_DIR),
            settings=_CHROMA_SETTINGS,
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        kwargs = {
            "ids": ids,
            "documents": texts,
            "embeddings": embeddings,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        self._collection.add(**kwargs)

    def query(self, query_embedding: list[float], top_k: int = 3) -> dict:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def delete_by_source(self, source: str) -> int:
        """Remove all chunks whose metadata source matches the given filename."""
        result = self._collection.get(
            where={"source": source},
            include=[],
        )
        ids = result.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(where={"source": source})
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[str]:
        if self.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        sources = {
            meta.get("source")
            for meta in (result.get("metadatas") or [])
            if meta and meta.get("source")
        }
        return sorted(sources)

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
