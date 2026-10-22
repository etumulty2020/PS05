"""Transparent word-overlap retrieval for the six Cedar Lab passages."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import re

STOPWORDS = set("a an the is are to of in at for and or may can i it be".split())
DOCS = Path(__file__).resolve().parent / "docs"


@dataclass(frozen=True)
class Chunk:
    id: str
    filename: str
    text: str


def load_chunks():
    chunks = []
    for filename in ("access.txt", "training.txt", "booking.txt"):
        source = (DOCS / filename).read_text(encoding="utf-8")
        for paragraph in re.split(r"\n\s*\n", source.strip()):
            match = re.fullmatch(r"\[([ATB]\d+)\]\s+(.+)", paragraph, re.DOTALL)
            if not match:
                raise ValueError(f"Invalid labeled paragraph in {filename}")
            chunk_id, text = match.groups()
            chunks.append(Chunk(chunk_id, filename, text))
    if len(chunks) != 6 or {c.id for c in chunks} != {"A1", "A2", "T1", "T2", "B1", "B2"}:
        raise ValueError("Expected exactly six unique chunks: A1, A2, T1, T2, B1, B2")
    return chunks


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def retrieve(question, chunks, k=2):
    query_words = words(question)
    scored = []
    for chunk in chunks:
        matched = query_words & words(chunk.text)
        scored.append((chunk, len(matched), sorted(matched)))
    ranked = sorted(scored, key=lambda item: (-item[1], item[0].id))
    selected = [item for item in ranked if item[1] > 0][:k]
    return ranked, selected


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("k must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("-k", "--k", type=positive_integer, default=2)
    args = parser.parse_args()
    ranked, selected = retrieve(args.question, load_chunks(), args.k)
    print(f"Question: {args.question}")
    print(f"k: {args.k}")
    print("\nAll chunk scores (descending score, then ascending ID):")
    for chunk, score, matched in ranked:
        print(f"{chunk.id} | score={score} | matched words: {', '.join(matched) or '(none)'}")
    if not selected:
        print("\nNo matching passages.")
        return
    print("\nSelected passages:")
    for chunk, score, _ in selected:
        print(f"[{chunk.id}] {chunk.filename} | score={score}")
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
