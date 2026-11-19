
"""
Download book cover images.
Sources:
1. Open Library Covers API using ISBN.
2. Image URLs from Books.csv (ImgL, ImgM, ImgS).

Output:
    artifacts/covers/{ISBN}.jpg

Usage:
    python -m book_recommender.data.download_covers
    python -m book_recommender.data.download_covers --limit 1000
    python -m book_recommender.data.download_covers --limit 0
"""

import argparse
import logging
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from book_recommender.config import settings
from book_recommender.data.loader import load_books
from book_recommender.data.preprocessing import clean_books

logger = logging.getLogger(__name__)

DEFAULT_MAX_BOOKS = 500
REQUEST_TIMEOUT = 15
MIN_IMAGE_BYTES = 2500
MIN_IMAGE_SIZE = 60
IMAGE_COLUMNS = ["ImgL", "ImgM", "ImgS"]

def normalize_isbn(isbn: str) -> str:
    """Normalize ISBN for consistent filenames."""
    return str(isbn).strip().replace("-", "").replace(" ", "")

def safe_filename(isbn: str) -> str:
    """Remove characters that are unsafe for filenames."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", normalize_isbn(isbn))

def build_cover_urls(isbn: str, row: pd.Series) -> list[str]:
    """Build ordered cover URL candidates."""
    urls = [
        f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg?default=false",
        f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg?default=false",
    ]

    for column in IMAGE_COLUMNS:
        url = row.get(column)
        if isinstance(url, str) and url.startswith("http"):
            urls.append(url)
    # Remove duplicate URLs while preserving order.
    return list(dict.fromkeys(urls))

def is_valid_image(image_path: Path) -> bool:
    """Check whether an existing image is valid."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False

def download_cover(
    row: pd.Series,
    session: requests.Session,
    covers_dir: Path,
) -> Path | None:
    """Download one book cover and save it as JPEG."""
    isbn = normalize_isbn(row["ISBN"])
    if not isbn or isbn.lower() == "nan":
        return None
    filename = safe_filename(isbn)
    cache_path = covers_dir / f"{filename}.jpg"

    # Reuse an existing valid image.
    if cache_path.exists():
        if is_valid_image(cache_path):
            return cache_path
        cache_path.unlink(missing_ok=True)

    urls = build_cover_urls(isbn, row)
    for url in urls:
        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            if len(response.content) < MIN_IMAGE_BYTES:
                continue

            # Decode and validate the downloaded image.
            with Image.open(BytesIO(response.content)) as image:
                image.load()
                if (
                    image.width < MIN_IMAGE_SIZE
                    or image.height < MIN_IMAGE_SIZE
                ):
                    continue

                image = image.convert("RGB")
                image.save(
                    cache_path,
                    format="JPEG",
                    quality=90,
                )

            return cache_path

        except Exception as exc:
            logger.debug(
                "Failed to download cover for ISBN %s from %s: %s",
                isbn,
                url,
                exc,
            )
            continue

    return None

def download_covers(max_books: int = DEFAULT_MAX_BOOKS) -> None:
    """Load books, select a catalog, and download available covers."""

    if max_books < 0:
        raise ValueError("max_books must be zero or a positive integer.")

    covers_dir = Path(settings.COVERS_DIR)
    covers_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading books...")
    books = clean_books(load_books())

    logger.info("Loaded %d books.", len(books))

    # Keep books that have at least one cover URL.
    has_cover_url = books[IMAGE_COLUMNS].notna().any(axis=1)
    catalog = books.loc[has_cover_url].copy()

    logger.info(
        "%d books have at least one cover URL.",
        len(catalog),
    )

    if catalog.empty:
        logger.warning("No books with cover URLs were found.")
        return

    # Reproducible sampling. A limit of 0 means all books.
    if max_books > 0 and len(catalog) > max_books:
        catalog = catalog.sample(
            n=max_books,
            random_state=settings.SEED,
        ).reset_index(drop=True)
    else:
        catalog = catalog.reset_index(drop=True)

    logger.info("Books selected for download: %d", len(catalog))
    logger.info("Covers directory: %s", covers_dir)

    downloaded = 0
    failed = 0

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; BookCoverDownloader/1.0)"
        })

        for _, row in tqdm(
            catalog.iterrows(),
            total=len(catalog),
            desc="Downloading covers",
        ):
            result = download_cover(
                row=row,
                session=session,
                covers_dir=covers_dir,
            )

            if result is not None:
                downloaded += 1
            else:
                failed += 1

    logger.info("=" * 55)
    logger.info("COVER DOWNLOAD COMPLETED")
    logger.info("Selected books: %d", len(catalog))
    logger.info("Available covers: %d", downloaded)
    logger.info("Failed / unavailable: %d", failed)
    logger.info("Output directory: %s", covers_dir)
    logger.info("=" * 55)

    if downloaded == 0:
        raise RuntimeError(
            "No covers were downloaded. Check internet access "
            "and the image URLs."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Download book covers for the recommender system."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_MAX_BOOKS,
        help="Maximum number of books to process. Use 0 for all books.",
    )

    args = parser.parse_args()
    download_covers(max_books=args.limit)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
