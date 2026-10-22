"""Inspect real local embeddings and retrieve Cedar Lab passages on CPU."""
import argparse
from importlib.metadata import version
import platform

import numpy as np
from sentence_transformers import SentenceTransformer

from keyword_retriever import load_chunks, positive_integer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOLERANCE = 1e-6


def show_vector(label, vector):
    length = np.linalg.norm(vector)
    if not np.isclose(length, 1.0, rtol=0, atol=TOLERANCE):
        raise ValueError(f"{label}: vector length is not approximately 1: {length}")
    print(f"{label} first 12: " + " ".join(f"{x:.4f}" for x in vector[:12]))
    print(f"{label} length: {length:.10f} (approximately 1: PASS)")


def retrieve(question, model, chunks, document_vectors, k=2):
    # Only the question is encoded here; document vectors are reused.
    query = model.encode_query(
        question, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float64)
    if query.shape != (document_vectors.shape[1],):
        raise ValueError(f"Incompatible query shape: {query.shape}")
    if not np.isfinite(query).all():
        raise ValueError("Non-finite question coordinates")
    scores = document_vectors @ query
    cosine = scores / (np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(query))
    if not np.allclose(scores, cosine, rtol=0, atol=TOLERANCE):
        raise ValueError("Dot products and direct cosine scores disagree")
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].id))
    return query, scores, cosine, order, order[:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+", help="One or more quoted questions")
    parser.add_argument("-k", "--k", type=positive_integer, default=2)
    args = parser.parse_args()
    chunks = load_chunks()
    # The prepared model is cached locally. No network requests or API calls.
    model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    if str(model.device) != "cpu":
        raise ValueError(f"Expected CPU, got {model.device}")
    # Encode only paragraph text, once per invocation, before the question loop.
    document_vectors = model.encode_document(
        [chunk.text for chunk in chunks], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float64)
    if document_vectors.shape != (6, 384) or not np.isfinite(document_vectors).all():
        raise ValueError(f"Invalid document embeddings: {document_vectors.shape}")

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")
    print(f"Python: {platform.python_version()}")
    for package in ("sentence-transformers", "numpy", "torch", "transformers", "huggingface-hub"):
        print(f"{package}: {version(package)}")
    print(f"Document matrix shape: {document_vectors.shape}")
    print("Document embeddings computed once and reused for every question in this run.")
    print("All 384 coordinates, not just the displayed 12, are used in retrieval.")
    print("Individual coordinates have no assigned semantic names.")
    print("Calculations use unrounded model output, promoted to float64; rounding is display only.")
    print(f"Numerical checks use absolute tolerance {TOLERANCE} (relative tolerance 0).")
    for chunk, vector in zip(chunks, document_vectors):
        show_vector(f"Document {chunk.id} ({chunk.filename})", vector)

    for question in args.questions:
        print(f"\nQuestion: {question}")
        print(f"k: {args.k}")
        query, scores, cosine, order, selected = retrieve(
            question, model, chunks, document_vectors, args.k,
        )
        print(f"Question vector shape: {query.shape}")
        show_vector("Question", query)
        print("\nAll six scores (descending, ties by ascending ID):")
        for i in order:
            print(f"{chunks[i].id} | dot={scores[i]:.10f} | direct cosine={cosine[i]:.10f}")
        print(f"Dot/cosine agreement: PASS; max difference={np.max(np.abs(scores - cosine)):.12g}")

        top = order[0]
        products = query * document_vectors[top]
        print(f"\nCoordinate calculation for top passage {chunks[top].id}:")
        for j in range(12):
            print(f"{j + 1:2d}: {query[j]:.4f} * {document_vectors[top, j]:.4f} = {products[j]:.4f}")
        print(f"Subtotal coordinates 1-12: {products[:12].sum():.10f}")
        print(f"Subtotal coordinates 13-384: {products[12:].sum():.10f}")
        print(f"Full similarity (all 384 products): {products.sum():.10f}")
        if not np.isclose(products.sum(), scores[top], rtol=0, atol=TOLERANCE):
            raise ValueError("Coordinate products do not sum to the retrieval score")
        print("Coordinate-sum/dot-product agreement: PASS")
        print("\nSelected passages (top k, no relevance threshold):")
        for i in selected:
            chunk = chunks[i]
            print(f"[{chunk.id}] {chunk.filename} | score={scores[i]:.10f}")
            print(chunk.text)
            print()


if __name__ == "__main__":
    main()
