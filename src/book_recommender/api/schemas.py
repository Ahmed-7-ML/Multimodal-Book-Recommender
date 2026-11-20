"""
Define Request & Response Models
"""

from typing import Literal
from pydantic import BaseModel, Field, model_validator

class TextRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=1000,
        description='Text description used to search for similar books'
    )
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    exclude_isbns: list[str] | None = None
    max_per_author: int | None = Field(default=None, ge=1)

class MultimodalRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=1000,
        description="Text description used alongside the image.",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)

    text_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    image_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    exclude_isbns: list[str] | None = None
    max_per_author: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_weights(self):
        total = self.text_weight + self.image_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("text_weight and image_weight must sum to 1.0.")
        return self

class BookRecommendation(BaseModel):
    rank: int
    id: str
    isbn: str | None = None
    title: str | None = None
    author: str | None = None
    year: int | str | None = None
    publisher: str | None = None
    image_available: bool = False
    score: float

class RecommendationResponse(BaseModel):
    query_type: Literal["text", "image", "multimodal"]
    total: int
    recommendations: list[BookRecommendation]

class HealthResponse(BaseModel):
    status: Literal["ok"]
