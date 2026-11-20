"""Search the most relevant books using cosine similarity
Steps:
1. Convert the user query into Embedding Vector
2. Select the suitable Named Vector to search
3. Retrieve Top-K Results

Supported retrieval modes:
    1. Text-to-Book: Search using a text query.
    2. Image-to-Book: Search using a book cover image.
    3. Multimodal-to-Book: Search using both text and image.

All vector searches are performed on Qdrant Cloud.
CLIP embeddings are generated locally.
"""
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from qdrant_client.models import Filter

from book_recommender.config import settings
from book_recommender.models.clip_encoder import CLIPEncoder
from book_recommender.embeddings.fusion import fuse_embeddings
from book_recommender.vector_store.qdrant_store import QdrantStore

class BookRetriever:
    """Retrieve similar books from Qdrant using CLIP Embeddings""" 
    def __init__(self, encoder: CLIPEncoder | None = None, store: QdrantStore | None = None):
        self.encoder = encoder or CLIPEncoder()
        self.store = store or QdrantStore()
        self.embedding_dim = self.encoder.embedding_dim
        if self.embedding_dim != settings.QDRANT_VECTOR_SIZE:
            raise ValueError("CLIP embedding dimension does not match Qdrant: "
                             f"{self.embedding_dim} != {settings.QDRANT_VECTOR_SIZE}")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be a positive integer.")

    @staticmethod
    def _load_image(image: str | Path | Image.Image) -> Image.Image:
        """Load an image and convert it to RGB."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Validate and prepare a single query embedding."""
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected embedding shape ({self.embedding_dim},), "
                f"got {embedding.shape}"
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or infinite values.")
        return embedding

    @staticmethod
    def _format_results(points: list[Any]) -> list[dict]:
        """Convert Qdrant ScoredPoints into plain dictionaries."""
        results = []
        for point in points:
            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": point.payload or {},
                }
            )
        return results

    def _search(
        self, 
        query_embedding: np.ndarray, 
        vector_name: str,
        limit: int,
        query_filter: Filter | None = None
        ) -> list[dict]:
        """Execute vector search on Qdrant"""
        self._validate_limit(limit=limit)
        query_embedding = self._validate_embedding(embedding=query_embedding)
        points = self.store.search(
            query_vector=query_embedding.tolist(),
            vector_name=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True
        )
        return self._format_results(points=points)

    def retrieve_by_text(
        self,
        query: str,
        limit: int = 10,
        query_filter: Filter | None = None,
        ) -> list[dict]:
        """
        Retrieve books using a natural-language text query.
        Searches against `text_vector`.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Text query cannot be empty.")
        embeddings = self.encoder.encode_texts([query.strip()])
        query_embedding = embeddings[0]
        return self._search(
            query_embedding=query_embedding,
            vector_name="text_vector",
            limit=limit,
            query_filter=query_filter
        )

    def retrieve_by_image(
        self,
        image: str | Path | Image.Image,
        limit: int = 10,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """
        Retrieve books using a book cover image.
        Searches against `image_vector`.
        Only points containing image_vector can match.
        """
        image = self._load_image(image=image)
        embeddings = self.encoder.encode_images([image])
        query_embedding = embeddings[0]
        return self._search(
            query_embedding=query_embedding,
            vector_name="image_vector",
            limit=limit,
            query_filter=query_filter
        )

    def retrieve_multimodal(
        self,
        text: str,
        image: str | Path | Image.Image,
        limit: int = 10,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        query_filter: Filter | None = None,
    ) -> list[dict]:
        """
        Retrieve books using both text and image queries.
        The query embeddings are fused, then compared against `fused_vector` in Qdrant.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text query cannot be empty.")
        image = self._load_image(image=image)
        
        text_embedding = self.encoder.encode_texts([text])
        image_embedding = self.encoder.encode_images([image])
        query_embedding = fuse_embeddings(
            text_embeddings=text_embedding,
            image_embeddings=image_embedding,
            text_weight=text_weight,
            image_weight=image_weight,
            image_mask=np.ones(1, dtype=bool)
        )
        query_embedding = query_embedding[0]
        return self._search(
            query_embedding=query_embedding,
            vector_name="fused_vector",
            limit=limit,
            query_filter=query_filter
        )

    def close(self) -> None:
        """Close Qdrant connection."""
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
