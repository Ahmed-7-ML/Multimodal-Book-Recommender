import pandas as pd
from book_recommender.config import settings

def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename the columns to be short, descriptive and readable.
    - Fix Data Types.
    - Apply Filters.
    - Handling Missing Values.
    - Handling Duplicates.
    """
    books = books.copy()

    books = books.rename(columns={
        "Book-Title": "Title",
        "Book-Author": "Author",
        "Year-Of-Publication": "Year",
        "Image-URL-S": "ImgS",
        "Image-URL-M": "ImgM",
        "Image-URL-L": "ImgL",
    })

    books["ISBN"] = books["ISBN"].astype(str).str.strip()

    books["Year"] = pd.to_numeric(
        books["Year"],
        errors="coerce"
    )

    books.loc[
        ~books["Year"].between(settings.MIN_YEAR, settings.MAX_YEAR),
        "Year"
    ] = None

    books["Title"] = books["Title"].fillna("Unknown Title")
    books["Author"] = books["Author"].fillna("Unknown Author")
    books["Publisher"] = books["Publisher"].fillna("Unknown Publisher")

    books = books.drop_duplicates(subset=["ISBN"])

    return books.reset_index(drop=True)

def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename User-ID -> UserID.
    - Fix Data Types: UserID numeric, Age numeric.
    - Ages outside [MIN_AGE, MAX_AGE] -> missing (same idea as notebooks).
    - Drop rows without UserID and duplicate UserIDs.
    """
    users = users.copy()

    users = users.rename(columns={"User-ID": "UserID"})

    users["UserID"] = pd.to_numeric(users["UserID"], errors="coerce")
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")

    users.loc[
        ~users["Age"].between(settings.MIN_AGE, settings.MAX_AGE),
        "Age"
    ] = None

    users = users.dropna(subset=["UserID"])
    users["UserID"] = users["UserID"].astype(int)

    users = users.drop_duplicates(subset=["UserID"])

    return users.reset_index(drop=True)

def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename User-ID -> UserID and Book-Rating -> Rating.
    - Fix Data Types: UserID/Rating numeric, ISBN stripped text.
    - Keep valid ratings only (0 = implicit, 1-10 = explicit).
    - Drop rows with missing values/duplicates.
    """
    ratings = ratings.copy()

    ratings = ratings.rename(columns={
        "User-ID": "UserID",
        "Book-Rating": "Rating",
    })

    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip()
    ratings["UserID"] = pd.to_numeric(ratings["UserID"], errors="coerce")
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")

    ratings = ratings.dropna(subset=["UserID", "Rating"])
    ratings = ratings[ratings["Rating"].between(0, 10)]

    ratings["UserID"] = ratings["UserID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(int)

    ratings = ratings.drop_duplicates()

    return ratings.reset_index(drop=True)

def create_book_description(row: pd.Series) -> str:
    """
    - Combine the book metadata into one text.
    - (Title, Author, Publisher, Year) --> Metadata
    """
    parts = []

    title = row.get("Title")
    author = row.get("Author")
    publisher = row.get("Publisher")
    year = row.get("Year")

    if pd.notna(title):
        parts.append(f"Book Title: {title}.")

    if pd.notna(author):
        parts.append(f"Author: {author}.")

    if pd.notna(year):
        parts.append(f"Published in {int(year)}.")

    if pd.notna(publisher):
        parts.append(f"Publisher: {publisher}.")

    return " ".join(parts) if parts else "A book."

def add_book_description(books: pd.DataFrame) -> pd.DataFrame:
    """Add a Description column built row-by-row from clean_books output."""
    books = books.copy()
    books["Description"] = books.apply(create_book_description, axis=1)
    return books

def save_processed(
    books: pd.DataFrame,
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    output_dir=None,
) -> dict:
    """Save preprocessed DataFrames to data/processed (index=False)."""
    output_dir = output_dir or settings.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "books": output_dir / settings.BOOKS_PROCESSED_FILE.name,
        "users": output_dir / settings.USERS_PROCESSED_FILE.name,
        "ratings": output_dir / settings.RATINGS_PROCESSED_FILE.name,
    }

    books.to_csv(paths["books"], index=False)
    users.to_csv(paths["users"], index=False)
    ratings.to_csv(paths["ratings"], index=False)

    return paths
