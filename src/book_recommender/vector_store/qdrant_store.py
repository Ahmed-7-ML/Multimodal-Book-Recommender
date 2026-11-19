"""
Qdrant Cloud vector database operations.
Collection of 3 Named Vectors:
    - Text Vector
    - Image Vector
    - Fused Vector

Responsibilities:
- Initialize Qdrant Cloud client.
- Create collection with named vectors.
- Upsert points in batches.
- Search using named vectors.
- Retrieve collection information.
- Delete collection when explicitly requested.
"""
from collections.abc import Iterable, Mapping
from typing import Any
import math
from book_recommender.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter
)

class QdrantStore:
    """Cloud Qdrant Vector Store"""
    VECTOR_NAMES = {
        'text_vector',
        'image_vector',
        'fused_vector'
    }
    
    # If the Book doesnot contain Covers, Use the Text & Fused Only
    REQUIRED_VECTORS = {
        'text_vector',
        'fused_vector'
    }
    
    def __init__(self, client: QdrantClient | None = None) -> None:
        """Connect to Qdrant Cloud"""
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        self.batch_size = settings.QDRANT_UPSERT_BATCH_SIZE
        self.client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            # cloud_inference=True, # Commented as we use the CLIP Encoder Locally
            timeout=60
        )

    def create_collection(self) -> None:
        """Create our Collection"""
        if self.client.collection_exists(self.collection_name):
            """Check if the Collection is Existed Before"""
            return
        
        vectors_config = {
            vector_name: VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
            for vector_name in self.VECTOR_NAMES
        }
        self.client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=vectors_config
        )

    def delete_collection(self) -> None:
        """Delete the collection and all its points"""
        if self.client.collection_exists(self.collection_name):
            """Check if the Collection is Existed or not"""
            self.client.delete_collection(
                collection_name=self.collection_name
            )

    def get_collection_info(self) -> Any:
        """Return Collection Information"""
        return self.client.get_collection(
            collection_name=self.collection_name
        )

    def _validate_vector(self, vector: list[float], vector_name: str) -> None:
        if vector_name not in self.VECTOR_NAMES:
            raise ValueError(f"Unknown vector name: {vector_name}")
        
        if len(vector) != self.vector_size:
            raise ValueError(
                f"{vector_name} must have dimension "
                f"{self.vector_size}, got {len(vector)}"
            )
        
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError(f"{vector_name} contains NaN or infinity.")

    def _validate_point(self, point: PointStruct) -> None:
        vectors = point.vector
        if not isinstance(vectors, Mapping):
            raise ValueError("Expected named vectors as a dictionary.")
        
        unknown_vectors = set(vectors) - self.VECTOR_NAMES
        if unknown_vectors:
            raise ValueError(f"Unknown vectors: {unknown_vectors}")

        missing_vectors = self.REQUIRED_VECTORS - set(vectors)
        if missing_vectors:
            raise ValueError(f"Missing required vectors: {missing_vectors}")
        
        for name, vector in vectors.items():
            self._validate_vector(vector, name)

    def upsert_points(self, points: Iterable[PointStruct]) -> None:
        """
        - Upload points to Qdrant Cloud in batches.
        - Existing points with the same ID are updated.
        """
        if not self.client.collection_exists(self.collection_name):
            """If the Collection is not existed -> Create it"""
            self.create_collection()
        
        # Add Points in Collection by Batches
        batch = []
        for point in points:
            self._validate_point(point=point)
            batch.append(point)

            if len(batch) >= self.batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True # Wait until Successful Upload
                )
                batch.clear()
        
        if batch:
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,  # Wait until Successful Upload
            )

    def search(
        self,
        query_vector: list[float],
        vector_name: str = "fused_vector",
        limit: int = 10,
        query_filter: Filter | None = None,
        with_payload: bool = True
    ) -> list[Any]:
        """Search using a specific Named Vector"""
        self._validate_vector(query_vector, vector_name)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=False
        )
        return response.points

    # Lifecycle: Enter, Exit, Close
    def close(self) -> None:
        """Close the Qdrant Client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
