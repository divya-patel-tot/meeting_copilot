#!/usr/bin/env python3
"""Dev runner: query the knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.rag.knowledge_base import KnowledgeBase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the knowledge base.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text (prompts if omitted)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        help="Number of results to return (default: 4)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    query_text = " ".join(args.query).strip()
    if not query_text:
        query_text = input("Enter query: ").strip()
    if not query_text:
        print("No query provided.", file=sys.stderr)
        return 1

    kb = KnowledgeBase()
    results = kb.query(query_text, top_k=args.top_k)
    if not results:
        print("No results found.")
        return 0

    print(f'Query: "{query_text}"\n')
    for rank, match in enumerate(results, start=1):
        print(
            f"{rank}. score={match['score']:.3f} "
            f"source={match['source']} chunk={match['chunk_index']}"
        )
        print(f"   {match['text']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
