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

│       │   └──download_covers.py
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

│       │   └── indexing.py
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
├── frontend/
│   └── app.py
│
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
