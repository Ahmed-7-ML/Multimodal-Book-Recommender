from data.loader import load_books, load_ratings, load_users
from data.preprocessing import clean_books, create_book_description

print(f"[STEP 1] Loading the Books Catalog...")
books = load_books()

print(f"[STEP 2] Loading the Users Records...")
users = load_users()

print(f"[STEP 3] Loading the Ratings...")
ratings = load_ratings()

print(f"[STEP 4] Cleaning & Pre-processing Books Dataset...")
books = clean_books(books)

print(f"[STEP 5] Create Book Description...")
book_descriptions = create_book_description()

