"""
Weighted Fusion
E_{final} = Normalize(alpha*E_{text} + beta*E_{image})
- Initially alpha = beta = 0.5
- We can change them and see the results
"""

import numpy as np

def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Apply L2 normalization to embeddings.
    Args:
        embeddings: Array of shape (N, D) or (D,).
        eps: Small value to prevent division by zero.
    Returns:
        Normalized float32 embeddings.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)

def fuse_embeddings(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray | None = None,
    image_mask: np.ndarray | None = None,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> np.ndarray:
    """
    Fuse text and image embeddings using weighted addition.
    Missing images are handled using image_mask.
    Text-only books retain their text embeddings.
    Args:
        text_embeddings: Shape (N, D).
        image_embeddings: Shape (N, D), optional.
        image_mask: Boolean array of shape (N,).
                    True means the book has a valid image.
        text_weight: Text contribution.
        image_weight: Image contribution.
    Returns:
        Final normalized embeddings of shape (N, D).
    """
    if text_weight < 0 or image_weight < 0:
        raise ValueError("Weights must be non-negative.")

    if text_weight == 0 and image_weight == 0:
        raise ValueError("At least one weight must be positive.")

    # Validate shape BEFORE normalizing: normalize_embeddings() calls
    # norm(axis=1), which raises a cryptic error on 1-D / empty input.
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must have shape (N, D).")
    text_embeddings = normalize_embeddings(text_embeddings)

    n_samples, embedding_dim = text_embeddings.shape
    if image_embeddings is None:
        return text_embeddings.copy()

    image_embeddings = np.asarray(image_embeddings, dtype=np.float32)

    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("Text and image embeddings must have the same shape.")

    if image_mask is None:
        image_mask = np.ones(n_samples, dtype=bool)
    else:
        image_mask = np.asarray(image_mask, dtype=bool)

    if image_mask.shape != (n_samples,):
        raise ValueError("image_mask must have shape (N,).")

    image_embeddings = normalize_embeddings(image_embeddings)

    # Start with text-only representations.
    final_embeddings = text_embeddings.copy()

    # Fuse only books that have valid images.
    valid_indices = np.flatnonzero(image_mask)

    if len(valid_indices) > 0:
        fused = (text_weight * text_embeddings[valid_indices] + image_weight * image_embeddings[valid_indices])
        final_embeddings[valid_indices] = fused

    # Normalize final vectors.
    final_embeddings = normalize_embeddings(final_embeddings)

    return final_embeddings.astype(np.float32)
