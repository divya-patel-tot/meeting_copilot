import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.rag.embedder import Embedder

        embedder = Embedder()
        texts = ["hello world", "testing embeddings"]
        vectors = embedder.embed(texts)

        if len(vectors) != 2:
            return False, f"Expected 2 vectors, got {len(vectors)}"
        if len(vectors[0]) != len(vectors[1]):
            return False, "Vector dimensions differ"
        if len(vectors[0]) == 0:
            return False, "Empty embedding vector"
        if all(v == 0.0 for v in vectors[0]):
            return False, "Embedding vector is all zeros"
        return True, f"Generated 2 vectors of dimension {len(vectors[0])}"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
