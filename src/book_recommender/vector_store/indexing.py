"""
Build Qdrant points from book metadata and embeddings.
- Use ISBN as UUID
Pipeline:
    Load Books
        -> Clean Books
        -> Resolve Cover Images
        -> Generate CLIP Embeddings
        -> Fuse Embeddings
        -> Build Qdrant Points
        -> Upload to Qdrant Cloud

Storage:
    Qdrant Cloud only.
"""

# UUID objects (universally unique identifiers) according to RFC 4122.
from uuid import uuid5, NAMESPACE_URL
import numpy as np
import pandas as pd
from qdrant_client.models import PointStruct
import logging
from pathlib import Path

from book_recommender.config import settings
from book_recommender.data.loader import load_books
from book_recommender.data.preprocessing import clean_books
from book_recommender.embeddings.generator import EmbeddingGenerator
from book_recommender.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
INDEX_BATCH_SIZE = settings.INDEX_BATCH_SIZE
# Supported cover image extensions.
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)

def make_point_id(isbn: str) -> str:
    """
    Generate a deterministic UUID from ISBN.
    The same ISBN always produces the same Point ID.
    """
    isbn = str(isbn).strip()
    if not isbn:
        raise ValueError("ISBN cannot be empty.")
    return str(uuid5(NAMESPACE_URL, f"book:{isbn}"))

def build_image_paths(books: pd.DataFrame) -> dict[str, Path]:
    """
    Build ISBN -> local cover image path mapping.
    Expected structure:
        artifacts/
            covers/
                ISBN.jpg
                ISBN.png
                ISBN.webp
    Only existing image files are included.
    """
    covers_dir = settings.COVERS_DIR
    image_paths = {}
    if not covers_dir.exists():
        logger.warning("Covers directory does not exist: %s", covers_dir,)
        return image_paths
    for isbn in books['ISBN']:
        isbn = str(isbn).strip()
        for extension in IMAGE_EXTENSIONS:
            image_path = covers_dir / f"{isbn}{extension}"
            if image_path.is_file():
                image_paths[isbn] = image_path
                break
    logger.info("Found %d cover images for %d books.",len(image_paths),len(books),)
    return image_paths

def _safe_payload_value(value):
    """Convert pandas values into JSON-compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

def build_qdrant_points(
    books: pd.DataFrame,
    text_embeddings: np.ndarray,
    fused_embeddings: np.ndarray,
    image_embeddings: np.ndarray | None = None,
    image_mask: np.ndarray | None = None,
) -> list[PointStruct]:
    """
    Convert book metadata and embeddings into Qdrant points.

    Expected:
        books: DataFrame with book metadata.
        text_embeddings: (n_books, 512)
        fused_embeddings: (n_books, 512)
        image_embeddings: (n_books, 512), optional.
        image_mask: (n_books,), True if image exists.
    
    Required vectors:
        text_vector
        fused_vector
    Optional vector:
        image_vector

    All embeddings must follow the same book order.
    """
    n_books = len(books)
    vector_size=settings.QDRANT_VECTOR_SIZE
    
    # Validate embedding dimensions.
    
    # Validate text embeddings.
    if text_embeddings.shape != (n_books, vector_size):
        raise ValueError("Invalid text embeddings shape.")

    # Validate fused embeddings.
    if fused_embeddings.shape != (n_books, vector_size):
        raise ValueError("Invalid fused embeddings shape.")

    # Validate optional image embeddings.
    if image_embeddings is not None:
        if image_embeddings.shape != (n_books, vector_size):
            raise ValueError(f"Invalid image embeddings shape: {image_embeddings.shape}")

    if image_mask is None:
        if image_embeddings is not None:
            raise ValueError("image_mask is required with image_embeddings.")
        image_mask = np.zeros(n_books, dtype=bool)

    if image_mask.shape != (n_books,):
        raise ValueError("Invalid image_mask shape.")

    if image_embeddings is None and image_mask.any():
        raise ValueError("Cannot mark images available without image embeddings.")
    
    # Validate ISBNs
    if books["ISBN"].isna().any():
        raise ValueError("ISBN cannot contain missing values.")

    if books["ISBN"].duplicated().any():
        raise ValueError("ISBN values must be unique.")
    
    points = []
    for idx, (_, book) in enumerate(books.iterrows()):
        isbn = str(book["ISBN"]).strip()
        if not isbn:
            raise ValueError(f"Missing ISBN at row {idx}.")
        
        # Required named vectors.
        vectors = {
            "text_vector": text_embeddings[idx].tolist(),
            "fused_vector": fused_embeddings[idx].tolist(),
        }
        # Optional image vector.
        has_image = bool(image_mask[idx])
        if has_image:
            vectors["image_vector"] = (image_embeddings[idx].tolist())
        
        # Book Metadata
        payload = {
            "isbn": isbn,
            "title": _safe_payload_value(book.get("Title")),
            "author": _safe_payload_value(book.get("Author")),
            "year": _safe_payload_value(book.get("Year")),
            "publisher": _safe_payload_value(book.get("Publisher")),
            "image_available": has_image,
        }

        point = PointStruct(
            id=make_point_id(isbn),
            vector=vectors,
            payload=payload
        )
        points.append(point)
    return points

def run_indexing(index_batch_size: int = INDEX_BATCH_SIZE) -> None:
    """
    Execute the complete book indexing pipeline.
    Steps:
        1. Load Books
        2. Clean Books
        3. Resolve cover images
        4. Initialize CLIP Encoder
        5. Create Qdrant Collection
        6. Generate Embeddings in Batches
        7. Build Qdrant Points
        8. Upload Points to Qdrant Cloud
    """
    if index_batch_size < 1:
        raise ValueError("index_batch_size must be positive.")

    logger.info("=" * 60)
    logger.info("Starting Book Indexing Pipeline")
    logger.info("=" * 60)

    # --------------------------------------------------
    # STEP 1: Load Books
    # --------------------------------------------------
    logger.info("[STEP 1] Loading Books...")
    books = load_books()
    logger.info("Loaded %d raw books.", len(books))

    # --------------------------------------------------
    # STEP 2: Clean Books
    # --------------------------------------------------
    logger.info("[STEP 2] Cleaning Books...")
    books = clean_books(books)
    if books.empty:
        raise ValueError("No books available for indexing.")
    logger.info("Books after cleaning: %d", len(books))

    # --------------------------------------------------
    # STEP 3: Resolve Cover Images
    # --------------------------------------------------
    logger.info("[STEP 3] Resolving Cover Images...")
    image_paths = build_image_paths(books)

    # --------------------------------------------------
    # STEP 4: Initialize Embedding Generator
    # --------------------------------------------------
    logger.info("[STEP 4] Initializing CLIP Generator...")
    generator = EmbeddingGenerator(
        text_batch_size=settings.TEXT_BATCH_SIZE,
        image_batch_size=settings.IMAGE_BATCH_SIZE,
    )

    if generator.embedding_dim != settings.QDRANT_VECTOR_SIZE:
        raise ValueError(
            "CLIP embedding dimension does not match Qdrant "
            f"configuration: {generator.embedding_dim} != "
            f"{settings.QDRANT_VECTOR_SIZE}"
        )
    logger.info("CLIP embedding dimension: %d", generator.embedding_dim,)

    # --------------------------------------------------
    # STEP 5: Connect to Qdrant Cloud
    # --------------------------------------------------
    logger.info("[STEP 5] Connecting to Qdrant Cloud...")
    store = QdrantStore()
    try:
        store.create_collection()
        logger.info("Connected to collection: %s", settings.QDRANT_COLLECTION_NAME)

        # --------------------------------------------------
        # STEP 6: Generate & Upload Embeddings in Batches
        # --------------------------------------------------
        total_books = len(books)
        total_batches = (total_books + index_batch_size - 1) // index_batch_size
        total_images = 0
        logger.info("[STEP 6] Indexing %d books in %d batches...", total_books, total_batches)
        for batch_number, start in enumerate(range(0, total_books, index_batch_size), start=1,):
            end = min(start + index_batch_size, total_books)
            
            # Preserve original row ordering.
            batch_books = books.iloc[start:end].copy()
            
            # Resolve paths for this batch.
            batch_image_paths = {
                str(isbn): image_paths[str(isbn)]
                for isbn in batch_books["ISBN"]
                if str(isbn) in image_paths
            }
            
            logger.info(
                "Batch %d/%d: Processing books %d-%d",
                batch_number,
                total_batches,
                start + 1,
                end,
            )
            
            # Generate text, image and fused embeddings.
            embedding_results = generator.generate(books=batch_books, image_paths=batch_image_paths)
            text_embeddings = embedding_results["text_embeddings"]
            image_embeddings = embedding_results["image_embeddings"]
            image_mask = embedding_results["image_mask"]
            fused_embeddings = embedding_results["book_embeddings"]
            total_images += int(image_mask.sum())

            # Build Qdrant points.
            points = build_qdrant_points(
                books=batch_books,
                text_embeddings=text_embeddings,
                fused_embeddings=fused_embeddings,
                image_embeddings=image_embeddings,
                image_mask=image_mask,
            )

            # Upload to Qdrant Cloud.
            store.upsert_points(points)

            logger.info(
                "Batch %d/%d uploaded successfully. "
                "Books: %d | Images: %d",
                batch_number,
                total_batches,
                len(points),
                int(image_mask.sum()),
            )

            # Release batch references.
            del embedding_results
            del text_embeddings
            del image_embeddings
            del fused_embeddings
            del points

        # --------------------------------------------------
        # STEP 7: Collection Information
        # --------------------------------------------------
        logger.info("[STEP 7] Retrieving Collection Info...")
        collection_info = store.get_collection_info()
        logger.info("Collection points: %s", collection_info.points_count)

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETED SUCCESSFULLY")
        logger.info("Total books processed: %d", total_books)
        logger.info("Books with covers: %d", total_images)
        logger.info("Books without covers: %d", total_books - total_images)
        logger.info("Qdrant collection: %s", settings.QDRANT_COLLECTION_NAME)
        logger.info("Points in collection: %s", collection_info.points_count)
        logger.info("=" * 60)
    finally:
        store.close()


# ==================================================
# 5. Entry Point
# ==================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )
    run_indexing()
