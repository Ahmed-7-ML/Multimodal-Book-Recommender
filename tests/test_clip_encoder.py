from PIL import Image
import numpy as np
import pytest
from book_recommender.models.clip_encoder import CLIPEncoder

@pytest.fixture(scope='module')
def encoder():
    """Load the Model once"""
    return CLIPEncoder()

# Test 1: Single text
def test_single_text(encoder):
    text_embedding = encoder.encode_text("A fantasy book about dragons and magic")
    assert text_embedding.shape == (encoder.embedding_dim,)
    assert np.isclose(np.linalg.norm(text_embedding), 1.0, atol=1e-5)


# Test 2: Multiple texts
def test_multiple_texts(encoder):
    texts = [
        "A book about artificial intelligence",
        "A historical novel about World War II",
        "A fantasy story about dragons",
    ]

    text_embeddings = encoder.encode_texts(
        texts,
        batch_size=2,
    )
    assert text_embeddings.shape == (3, encoder.embedding_dim)
    assert np.allclose(np.linalg.norm(text_embeddings, axis=1), 1.0, atol=1e-5)

# Test 3: Single image
def test_single_image(encoder):
    image = Image.new("RGB", (224, 224), color="white")
    image_embedding = encoder.encode_image(image)
    assert image_embedding.shape == (encoder.embedding_dim,)
    assert np.isclose(np.linalg.norm(image_embedding), 1.0, atol=1e-5)


# Test 4: Multiple images
def test_multiple_images(encoder):
    images = [
        Image.new("RGB", (224, 224), color="white"),
        Image.new("RGB", (224, 224), color="black"),
    ]
    image_embeddings = encoder.encode_images(
        images,
        batch_size=2,
    )
    assert image_embeddings.shape == (2, encoder.embedding_dim)
    assert np.allclose(np.linalg.norm(image_embeddings, axis=1), 1.0, atol=1e-5)
