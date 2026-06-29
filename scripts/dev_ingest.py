#!/usr/bin/env python3
"""Dev runner: ingest documents into the knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.rag.chunking import chunk_text
from app.core.rag.knowledge_base import KnowledgeBase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a document into the knowledge base.")
    parser.add_argument("file_path", nargs="?", help="Path to .pdf, .docx, or .txt file")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print knowledge base stats and exit",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the knowledge base (asks for confirmation)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    kb = KnowledgeBase()

    if args.stats:
        stats = kb.get_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print("Sources:")
        for source in stats["sources"]:
            print(f"  - {source}")
        return 0

    if args.clear:
        confirm = input("Clear the entire knowledge base? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return 0
        kb.clear()
        print("Knowledge base cleared.")
        return 0

    if not args.file_path:
        print("Provide a file path or use --stats / --clear.", file=sys.stderr)
        return 1

    path = Path(args.file_path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    chunk_count = kb.ingest_file(str(path))
    print(f"Ingested {chunk_count} chunk(s) from {path.name}")

    if chunk_count == 0:
        return 0

    suffix = path.suffix.lower()
    if suffix == ".txt":
        preview_source = path.read_text(encoding="utf-8")
    else:
        from app.core.rag.ingestion import parse_docx, parse_pdf

        if suffix == ".pdf":
            preview_source = parse_pdf(str(path))
        else:
            preview_source = parse_docx(str(path))

    sample_chunks = chunk_text(preview_source)[:2]
    print("\nSample chunks:")
    for index, chunk in enumerate(sample_chunks, start=1):
        preview = chunk[:150].replace("\n", " ")
        if len(chunk) > 150:
            preview += "..."
        print(f"  {index}. {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
