
import numpy as np
import pandas as pd
import pytest
from PIL import Image

class MockCLIPEncoder:
    """
    Lightweight CLIP mock for unit testing.
    Does not load a real model or use a GPU.
    """
    def __init__(self):
        self.embedding_dim = 4
        self.text_batches = []
        self.image_batches = []

    def encode_texts(self, texts, batch_size=32):
        self.text_batches.append(len(texts))
        embeddings = []
        for text in texts:
            value = len(text)
            embeddings.append([
                float(value),
                1.0,
                2.0,
                3.0,
            ])
        return np.asarray(embeddings, dtype=np.float32)

    def encode_images(self, images, batch_size=16):
        self.image_batches.append(len(images))
        embeddings = []
        for _ in images:
            embeddings.append([
                1.0,
                2.0,
                3.0,
                4.0,
            ])
        return np.asarray(embeddings, dtype=np.float32)

@pytest.fixture
def mock_encoder():
    return MockCLIPEncoder()

@pytest.fixture
def sample_books():
    return pd.DataFrame({
        "ISBN": [
            "1001",
            "1002",
            "1003",
            "1004",
            "1005",
        ],
        "title": [
            "Machine Learning",
            "Deep Learning",
            "Python Programming",
            "Data Science",
            "Artificial Intelligence",
        ],
        "author": [
            "Author A",
            "Author B",
            "Author C",
            "Author D",
            "Author E",
        ],
        "year": [
            2020,
            2021,
            2019,
            2022,
            2023,
        ],
        "publisher": [
            "Publisher A",
            "Publisher B",
            "Publisher C",
            "Publisher D",
            "Publisher E",
        ],
    })

@pytest.fixture
def sample_images(tmp_path):
    """
    Create three temporary RGB images.
    """
    image_paths = {}
    for isbn in ["1001", "1003", "1005"]:
        image_path = tmp_path / f"{isbn}.jpg"
        image = Image.new(
            "RGB",
            (100, 150),
            color="red",
        )
        image.save(image_path)
        image_paths[isbn] = image_path
    return image_paths
