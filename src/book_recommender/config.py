# Path provides operating-system-aware path handling. 
# You don't have to manually worry about Windows \ versus Linux /.
from pathlib import Path

# We use pydantic-settings to allow environment variables to override defaults
from pydantic_settings import BaseSettings

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
    CSV_SEPARATOR: str = ";"
    CSV_ENCODING: str = "latin-1"
    
    # Filtering and Pre-processing
    MIN_YEAR: int = 1800
    MAX_YEAR: int = 2026
    
    # Image Storage
    COVERS_DIR: Path = PROJECT_ROOT / "artifacts" / "covers"
    
    # CLIP Model Configurations
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    TEXT_BATCH_SIZE: int = 32
    IMAGE_BATCH_SIZE: int = 16
    
    class Config:
        env_file = '.env'
        extra = 'ignore'

settings = Settings()
