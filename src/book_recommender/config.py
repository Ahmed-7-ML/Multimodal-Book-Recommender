# Path provides operating-system-aware path handling. 
# You don't have to manually worry about Windows \ versus Linux /.
from pathlib import Path

# We use pydantic-settings to allow environment variables to override defaults
from pydantic_settings import SettingsConfigDict, BaseSettings

from dotenv import load_dotenv
import os

# Load Environment Variables
load_dotenv()

# Project Root
# config.py -> multimodal_book_recommender -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# print(PROJECT_ROOT) # .../multimodal-book-recommender/

class Settings(BaseSettings):
    """Central Configuration"""
    
    # Dataset Paths & Configurations
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    BOOKS_FILE: Path = RAW_DATA_DIR / "Books.csv"
    USERS_FILE: Path = RAW_DATA_DIR / "Users.csv"
    RATINGS_FILE: Path = RAW_DATA_DIR / "Ratings.csv"
    BOOKS_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "Books_processed.csv"
    USERS_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "Users_processed.csv"
    RATINGS_PROCESSED_FILE: Path = PROCESSED_DATA_DIR / "Ratings_processed.csv"
    CSV_SEPARATOR: str = ","
    CSV_ENCODING: str = "latin-1"
    
    # Filtering and Pre-processing
    MIN_YEAR: int = 1800
    MAX_YEAR: int = 2026
    MIN_AGE: int = 5
    MAX_AGE: int = 100
    
    # Image Storage
    COVERS_DIR: Path = PROJECT_ROOT / "artifacts" / "covers"
    
    # Reproducibility
    SEED: int = 42
    
    # CLIP Model Configurations
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    TEXT_BATCH_SIZE: int = 32
    IMAGE_BATCH_SIZE: int = 16
    DEVICE: str = 'auto'
    
    # Qdrant Vector Store
    QDRANT_URL: str = "https://846de8f8-6a87-4124-9c62-c7da86bbba0c.ap-northeast-2-0.aws.cloud.qdrant.io:6333"    
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "books"
    QDRANT_VECTOR_SIZE: int = 512
    QDRANT_UPSERT_BATCH_SIZE: int = 128

    # Number of books processed in each indexing iteration.
    INDEX_BATCH_SIZE: int = 256

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
