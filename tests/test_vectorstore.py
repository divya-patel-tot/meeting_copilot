import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.rag.embedder import Embedder
        from app.core.rag.vectorstore import VectorStore

        suffix = uuid.uuid4().hex[:8]
        store = VectorStore(collection_name=f"test_{suffix}")
        embedder = Embedder()

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a great language for building tools.",
            "Vector databases enable semantic search.",
        ]
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = embedder.embed(texts)

        store.add_documents(ids=ids, texts=texts, embeddings=embeddings)

        query_embedding = embeddings[1]
        results = store.query(query_embedding, top_k=1)

        if not results.get("ids") or not results["ids"][0]:
            return False, "Query returned no results"

        top_id = results["ids"][0][0]
        if top_id != "doc_1":
            return False, f"Expected top match doc_1, got {top_id}"

        return True, f"Top match: {top_id} — {results['documents'][0][0][:50]}..."
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
