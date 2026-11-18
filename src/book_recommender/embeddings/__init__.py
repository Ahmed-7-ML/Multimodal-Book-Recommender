"""Embeddings package: generator + fusion."""
from book_recommender.embeddings.generator import EmbeddingGenerator
from book_recommender.embeddings.fusion import fuse_embeddings, normalize_embeddings

__all__ = ["EmbeddingGenerator", "fuse_embeddings", "normalize_embeddings"]

