"""Load the assignment model on CPU and record actual environment details."""
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device="cpu")
vectors = model.encode_document(
    ["Cedar Lab setup verification."],
    normalize_embeddings=True,
    convert_to_numpy=True,
)
assert str(model.device) == "cpu", model.device
assert vectors.shape == (1, 384), vectors.shape
assert np.isfinite(vectors).all()
assert np.isclose(np.linalg.norm(vectors[0]), 1.0)

lines = [
    "Setup verification: PASS",
    f"Python: {platform.python_version()}",
    f"Platform: {platform.platform()}",
    f"Model: {model_name}",
    f"Device: {model.device}",
]
for package in ("sentence-transformers", "numpy", "torch", "transformers", "huggingface-hub"):
    lines.append(f"{package}: {version(package)}")
lines.extend([
    f"Smoke-test embedding shape: {vectors.shape}",
    f"Smoke-test vector norm: {np.linalg.norm(vectors[0]):.6f}",
    "This smoke test is setup evidence, not an assignment retrieval run.",
])
report = "\n".join(lines) + "\n"
Path("evidence").mkdir(exist_ok=True)
Path("evidence/setup-verification.txt").write_text(report)
print(report)
