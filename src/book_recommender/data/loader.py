import pandas as pd
from book_recommender.config import settings

def load_books(path=None) -> pd.DataFrame:
    """
    Load the Book-Crossing books dataset
    Args:
        path (Optional): Custom CSV Path
    Returns:
        Pandas DataFrame containing raw book records
    """
    file_path = path or settings.BOOKS_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Books dataset not found: {file_path}")

    books = pd.read_csv(
        file_path,
        sep=settings.CSV_SEPARATOR,
        encoding=settings.CSV_ENCODING,
        dtype={"ISBN": "string"},
        low_memory=False
    )
    return books

def load_users(path=None) -> pd.DataFrame:
    """
    Load the Book-Crossing users dataset
    Args:
        path (Optional): Custom CSV Path
    Returns:
        Pandas DataFrame containing raw user records
    """
    file_path = path or settings.USERS_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Users dataset not found: {file_path}")

    users = pd.read_csv(
        file_path,
        sep=settings.CSV_SEPARATOR,
        encoding=settings.CSV_ENCODING,
        low_memory=False
    )
    return users

def load_ratings(path=None) -> pd.DataFrame:
    """
    Load the Book-Crossing ratings dataset
    Args:
        path (Optional): Custom CSV Path
    Returns:
        Pandas DataFrame containing raw rating records
    """
    file_path = path or settings.RATINGS_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Ratings dataset not found: {file_path}")

    ratings = pd.read_csv(
        file_path,
        sep=settings.CSV_SEPARATOR,
        encoding=settings.CSV_ENCODING,
        low_memory=False
    )
    return ratings
