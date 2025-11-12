import os
from sentence_transformers import CrossEncoder

# Maps RERANKER_MODEL env var values to HuggingFace model IDs
RERANKER_REGISTRY: dict[str, str] = {
    "ms-marco-minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}

DEFAULT_RERANKER = "ms-marco-minilm-l6"


def create_reranker() -> tuple[CrossEncoder, str]:
    """Load the reranker selected by RERANKER_MODEL env var.

    Returns (model, key) so callers can report which model is active.
    """
    key = os.environ.get("RERANKER_MODEL", DEFAULT_RERANKER).strip().lower()
    model_id = RERANKER_REGISTRY.get(key)
    if model_id is None:
        valid = ", ".join(RERANKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown RERANKER_MODEL='{key}'. Valid options: {valid}"
        )
    print(f"[reranker] Loading '{key}' ({model_id})")
    return CrossEncoder(model_id), key
