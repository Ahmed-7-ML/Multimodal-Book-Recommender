import pandas as pd
from book_recommender.config import settings

def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """
    - Rename the columns to be short, decriptive and readable.
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

    return books

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
