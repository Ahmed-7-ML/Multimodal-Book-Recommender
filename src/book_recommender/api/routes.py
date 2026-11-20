import logging
from io import BytesIO
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import Image, UnidentifiedImageError
from book_recommender.api.schemas import (
    BookRecommendation,
    HealthResponse,
    MultimodalRequest,
    RecommendationResponse,
    TextRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def get_services(request: Request):
    """Retrieve shared application services."""
    return request.app.state.retriever, request.app.state.ranker

def load_uploaded_image(file: UploadFile) -> Image.Image:
    """
    Validate and load an uploaded image into memory.
    The image is converted to RGB and is not saved to disk.
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPEG, PNG, or WebP.",
        )
    image_bytes = file.file.read(MAX_IMAGE_SIZE + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size exceeds the 5 MB limit.",
        )
    try:
        # Verify that the uploaded file contains a valid image.
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Reopen after verify(), then convert to RGB.
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        ) from exc

def build_response(
    query_type: str,
    candidates: list[dict],
    ranker,
    top_k: int,
    min_score: float | None,
    exclude_isbns: list[str] | None,
    max_per_author: int | None,
) -> RecommendationResponse:
    """Rank retrieved candidates and build the API response."""
    recommendations = ranker.rank(
        candidates=candidates,
        top_k=top_k,
        min_score=min_score,
        exclude_isbns=exclude_isbns,
        max_per_author=max_per_author,
    )

    return RecommendationResponse(
        query_type=query_type,
        total=len(recommendations),
        recommendations=[
            BookRecommendation(**book)
            for book in recommendations
        ],
    )

@router.get("/health", response_model=HealthResponse, tags=['Health'])
def health_check():
    """Check whether the API process is running."""
    return HealthResponse(status="ok")

@router.post("/recommend/text", response_model=RecommendationResponse, tags=['Recommendations', 'Text'])
def recommend_by_text(data: TextRequest, request: Request):
    """Recommend books using a text description."""
    retriever, ranker = get_services(request)
    try:
        # Retrieve a larger candidate pool before ranking.
        candidate_limit = data.top_k * 3
        candidates = retriever.retrieve_by_text(
            query=data.query,
            limit=candidate_limit,
        )

        return build_response(
            query_type="text",
            candidates=candidates,
            ranker=ranker,
            top_k=data.top_k,
            min_score=data.min_score,
            exclude_isbns=data.exclude_isbns,
            max_per_author=data.max_per_author,
        )
    except Exception as exc:
        logger.exception("Text recommendation failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate text recommendations.",
        ) from exc

@router.post("/recommend/image", response_model=RecommendationResponse, tags=["Recommendations", "Image"],)
def recommend_by_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(default=10, ge=1, le=100),
    min_score: float | None = Form(default=None, ge=-1.0, le=1.0),
    exclude_isbns: list[str] | None = Form(default=None),
    max_per_author: int | None = Form(default=None, ge=1),
):
    """Recommend books using an uploaded image."""
    retriever, ranker = get_services(request)
    try:
        image = load_uploaded_image(file)
        candidate_limit = top_k * 3
        candidates = retriever.retrieve_by_image(
            image=image,
            limit=candidate_limit,
        )
        return build_response(
            query_type="image",
            candidates=candidates,
            ranker=ranker,
            top_k=top_k,
            min_score=min_score,
            exclude_isbns=exclude_isbns,
            max_per_author=max_per_author,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image recommendation failed.")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate image recommendations.",
        ) from exc
    finally:
        file.file.close()

@router.post("/recommend/multimodal", response_model=RecommendationResponse, tags=["Recommendations", "Text", "Image", "Multimodal"])
def recommend_multimodal(
    request: Request,
    text: str = Form(..., min_length=1, max_length=1000),
    file: UploadFile = File(...),
    top_k: int = Form(default=10, ge=1, le=100),
    min_score: float | None = Form(default=None, ge=-1.0, le=1.0),
    text_weight: float = Form(default=0.5, ge=0.0, le=1.0),
    image_weight: float = Form(default=0.5, ge=0.0, le=1.0),
    exclude_isbns: list[str] | None = Form(default=None),
    max_per_author: int | None = Form(default=None, ge=1),
):
    """Recommend books using both text and image."""
    retriever, ranker = get_services(request)
    try:
        # Validate multimodal weights.
        if abs(text_weight + image_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=422,
                detail="text_weight and image_weight must sum to 1.0.",
            )
        image = load_uploaded_image(file)
        candidate_limit = top_k * 3
        candidates = retriever.retrieve_multimodal(
            text=text,
            image=image,
            limit=candidate_limit,
            text_weight=text_weight,
            image_weight=image_weight,
        )
        return build_response(
            query_type="multimodal",
            candidates=candidates,
            ranker=ranker,
            top_k=top_k,
            min_score=min_score,
            exclude_isbns=exclude_isbns,
            max_per_author=max_per_author,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multimodal recommendation failed.")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate multimodal recommendations.",
        ) from exc
    finally:
        file.file.close()
