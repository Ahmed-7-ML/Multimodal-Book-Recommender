"""Data pipeline: load raw data -> preprocess -> save to data/processed.

Run from the project root (multimodal-book-recommender/):
    uv run python -m book_recommender.data.pipeline
"""

from book_recommender.config import settings
from book_recommender.data.loader import load_books, load_ratings, load_users
from book_recommender.data.preprocessing import (
    add_book_description,
    clean_books,
    clean_ratings,
    clean_users,
    save_processed,
)


def run_pipeline() -> dict:
    """Run the full pipeline and return the saved file paths."""
    print("[STEP 1] Loading the Books Catalog...")
    books = load_books()
    print(f"  -> {len(books):,} raw rows")

    print("[STEP 2] Loading the Users Records...")
    users = load_users()
    print(f"  -> {len(users):,} raw rows")

    print("[STEP 3] Loading the Ratings...")
    ratings = load_ratings()
    print(f"  -> {len(ratings):,} raw rows")

    print("[STEP 4] Cleaning & Pre-processing Datasets...")
    books = clean_books(books)
    users = clean_users(users)
    ratings = clean_ratings(ratings)
    print(f"  -> Books: {len(books):,} rows")
    print(f"  -> Users: {len(users):,} rows")
    print(f"  -> Ratings: {len(ratings):,} rows")

    print("[STEP 5] Create Book Description...")
    books = add_book_description(books)

    print("[STEP 6] Saving Preprocessed Datasets...")
    paths = save_processed(
        books, users, ratings, output_dir=settings.PROCESSED_DATA_DIR
    )
    for name, path in paths.items():
        print(f"  -> {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    print("Pipeline finished.")
    return paths


if __name__ == "__main__":
    run_pipeline()
