import re


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(
    text: str,
    target_words: int = 300,
    overlap_words: int = 40,
) -> list[str]:
    """Split text into sentence-aware chunks with trailing overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if current and current_words + sentence_words > target_words:
            chunks.append(" ".join(current))

            overlap: list[str] = []
            overlap_count = 0
            for prior in reversed(current):
                overlap_words_count = _word_count(prior)
                if overlap and overlap_count + overlap_words_count > overlap_words:
                    break
                overlap.insert(0, prior)
                overlap_count += overlap_words_count

            current = overlap + [sentence]
            current_words = sum(_word_count(s) for s in current)
        else:
            current.append(sentence)
            current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks
