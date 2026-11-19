"""
Generate Embeddings for Text & Images using CLIP Encoder
"""
from pathlib import Path
from typing import Mapping
import logging
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from book_recommender.data.preprocessing import create_book_description
from book_recommender.models.clip_encoder import CLIPEncoder
from book_recommender.embeddings.fusion import fuse_embeddings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate multimodal embeddings for a collection of books.
    Pipeline:
        Books DataFrame
            -> Text descriptions
            -> CLIP text embeddings
            -> Load available cover images
            -> CLIP image embeddings
            -> Fusion
            -> Final book embeddings
    """

    def __init__(
        self,
        encoder: CLIPEncoder | None = None,
        text_batch_size: int = 32,
        image_batch_size: int = 16,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ):
        if text_batch_size < 1 or image_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")

        if text_weight < 0 or image_weight < 0:
            raise ValueError("Weights must be non-negative.")

        if text_weight == 0 and image_weight == 0:
            raise ValueError("At least one weight must be positive.")

        self.encoder = encoder or CLIPEncoder()

        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size

        self.text_weight = text_weight
        self.image_weight = image_weight

        self.embedding_dim = self.encoder.embedding_dim

    @staticmethod
    def _validate_books(books: pd.DataFrame) -> None:
        """Validate the input DataFrame."""
        # clean_books() outputs Capitalized columns (Title/Author/Year),
        # but some callers/tests use lowercase (title/author/year/publisher).
        # Accept both: validate on ISBN + that at least one naming variant
        # of each metadata column is present.
        required_groups = {
            "ISBN": ("ISBN",),
            "Title": ("Title", "title"),
            "Author": ("Author", "author"),
            "Year": ("Year", "year"),
            "Publisher": ("Publisher", "publisher"),
        }
        missing = [key for key, variants in required_groups.items()
                   if not any(v in books.columns for v in variants)]
        if missing:
            missing_variants = {m: required_groups[m] for m in missing}
            raise ValueError(
                f"Missing required columns: {sorted(missing_variants)}; "
                f"accepted variants: {missing_variants}"
            )

        if books.empty:
            raise ValueError("Books DataFrame is empty.")

        if books["ISBN"].isna().any():
            raise ValueError("ISBN cannot contain missing values.")

        if books["ISBN"].duplicated().any():
            raise ValueError("ISBN values must be unique.")

    @staticmethod
    def build_texts(books: pd.DataFrame) -> list[str]:
        """
        Build text descriptions using preprocessing.py.
        """
        return [
            create_book_description(row)
            for _, row in books.iterrows()
        ]

    @staticmethod
    def load_image(image_path) -> Image.Image | None:
        """
        Load a book cover as an RGB PIL image.

        Returns None if the image is missing or invalid.
        """
        if image_path is None:
            return None
        try:
            path = Path(image_path)
            if not path.is_file():
                logger.warning("Image not found: %s", path)
                return None
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as exc:
            logger.warning(
                "Could not load image %s: %s",
                image_path,
                exc,
            )
            return None

    def generate_text_embeddings(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate text embeddings in batches.

        Returns:
            Array of shape (N, embedding_dim).
        """
        if not texts:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )
        batches = []
        for start in range(0, len(texts), self.text_batch_size):
            batch = texts[start:start + self.text_batch_size]
            embeddings = self.encoder.encode_texts(batch,batch_size=self.text_batch_size,)
            batches.append(
                np.asarray(embeddings, dtype=np.float32)
            )

        return np.concatenate(batches, axis=0)

    def generate_image_embeddings(
        self,
        image_paths: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate image embeddings only for valid images.

        Missing images are represented by zero rows and
        identified through image_mask.

        Returns:
            image_embeddings: (N, embedding_dim)
            image_mask: (N,)
        """
        n_samples = len(image_paths)
        image_embeddings = np.zeros((n_samples, self.embedding_dim),dtype=np.float32,)

        image_mask = np.zeros(n_samples,dtype=bool,)
        valid_indices = []
        valid_images = []
        # Load only available images.
        for index, image_path in enumerate(image_paths):
            image = self.load_image(image_path)
            if image is None:
                continue
            valid_indices.append(index)
            valid_images.append(image)

        if not valid_images:
            return image_embeddings, image_mask

        # Encode images in batches.
        for start in range(0, len(valid_images),self.image_batch_size,):
            end = start + self.image_batch_size
            batch_images = valid_images[start:end]
            batch_indices = valid_indices[start:end]
            embeddings = self.encoder.encode_images(
                batch_images,
                batch_size=self.image_batch_size,
            )
            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )
            image_embeddings[batch_indices] = embeddings
            image_mask[batch_indices] = True

        return image_embeddings, image_mask

    def generate(self, books: pd.DataFrame, image_paths: Mapping[str, str | Path] | None = None,) -> dict:
        """
        Generate text, image, and final book embeddings.
        Args:
            books:
                Cleaned books DataFrame.
            image_paths:
                Mapping of ISBN -> local cover image path.
        Returns:
            Dictionary containing:
                ISBN
                texts
                text_embeddings
                image_embeddings
                image_mask
                book_embeddings
        """
        self._validate_books(books)
        image_paths = image_paths or {}
        # 1. Build text representations.
        texts = self.build_texts(books)
        logger.info(
            "Generating text embeddings for %d books",
            len(books),
        )
        
        # 2. Generate text embeddings.
        text_embeddings = self.generate_text_embeddings(
            texts
        )
        
        # 3. Resolve local image paths by ISBN.
        paths = [
            image_paths.get(str(isbn))
            for isbn in books["ISBN"]
        ]
        logger.info("Generating image embeddings")

        # 4. Generate image embeddings.
        image_embeddings, image_mask = (
            self.generate_image_embeddings(paths)
        )

        # 5. Fuse text and image embeddings.
        logger.info("Fusing multimodal embeddings")

        book_embeddings = fuse_embeddings(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            image_mask=image_mask,
            text_weight=self.text_weight,
            image_weight=self.image_weight,
        )
        logger.info(
            "Embedding generation completed: %d books, %d dimensions, %d images",
            len(books),
            self.embedding_dim,
            int(image_mask.sum()),
        )

        # 6. Return aligned results.
        return {
            "ISBN": books["ISBN"].astype(str).to_numpy(),
            "texts": texts,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "image_mask": image_mask,
            "book_embeddings": book_embeddings,
        }
