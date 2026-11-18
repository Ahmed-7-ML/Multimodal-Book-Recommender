multimodal-book-recommender/
│
├── data/
│   ├── raw/
│   │   ├── Books.csv
│   │   ├── Users.csv
│   │   └── Ratings.csv
│   │
│   └── processed/
│
├── artifacts/
│   └── covers/
│
├── src/
│   └── book_recommender/
│       │
│       ├── __init__.py
│       │
│       ├── config.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── preprocessing.py
│       │
│       ├── models/
│       │   └── clip_encoder.py
│       │
│       ├── embeddings/
│       │   ├── generator.py
│       │   └── fusion.py
│       │
│       ├── vector_store/
│       │   └── qdrant_store.py
│       │
│       ├── retrieval/
│       │   └── retriever.py
│       │
│       ├── recommendation/
│       │   └── ranker.py
│       │
│       └── api/
│           ├── main.py
│           ├── schemas.py
│           └── routes.py
│
├── scripts/
│   ├── download_covers.py
│   └── index_books.py
│
├── frontend/
│   └── app.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_embeddings.py

│   └── test_clip_model.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
