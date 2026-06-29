from fastembed import TextEmbedding


class Embedder:
    """ONNX text embedding wrapper using fastembed."""

    def __init__(self) -> None:
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [embedding.tolist() for embedding in self._model.embed(texts)]
