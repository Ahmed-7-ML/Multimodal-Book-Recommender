"""
Resource Management (Lifespan) = Lifecycle
- Startup: Create Book Retriever, Load CLIP, Connect to Qdrant (Only ONCE)
- While Running: Use the same instance not reloading them
- Shutdown: Close the Retriever
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from book_recommender.api.routes import router
from book_recommender.recommendation.ranker import BookRanker
from book_recommender.retrieval.retriever import BookRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared services when the application starts.
    Close resources when the application shuts down.
    """
    logger.info("Initializing recommendation services...")
    retriever = BookRetriever()
    ranker = BookRanker()
    app.state.retriever = retriever
    app.state.ranker = ranker
    logger.info("Recommendation services initialized successfully.")
    try:
        yield
    finally:
        logger.info("Shutting down recommendation services...")
        retriever.close()
        logger.info("Recommendation services closed.")

app = FastAPI(
    title="Multimodal Book Recommender API",
    description=(
        "A multimodal content-based book recommendation system "
        "powered by CLIP and Qdrant."
    ),
    version='0.1.0',
    lifespan=lifespan
)

app.include_router(router=router)
