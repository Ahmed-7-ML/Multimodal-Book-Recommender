import numpy as np
import pytest
from book_recommender.embeddings.fusion import (
    normalize_embeddings,
    fuse_embeddings,
)


class TestNormalizeEmbeddings:
    def test_normalization(self):
        embeddings = np.array([
            [3.0, 4.0],
            [1.0, 0.0],
        ])
        result = normalize_embeddings(embeddings)
        expected = np.array([
            [0.6, 0.8],
            [1.0, 0.0],
        ])
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )

    def test_output_dtype(self):
        embeddings = np.array([
            [3.0, 4.0],
        ], dtype=np.float64)
        result = normalize_embeddings(embeddings)
        assert result.dtype == np.float32

    def test_zero_vector(self):
        embeddings = np.array([
            [0.0, 0.0],
        ])
        result = normalize_embeddings(embeddings)
        np.testing.assert_array_equal(
            result,
            np.zeros((1, 2), dtype=np.float32),
        )

class TestFuseEmbeddings:
    def test_text_and_image_fusion(self):
        text = np.array([
            [1.0, 0.0],
        ])
        image = np.array([
            [0.0, 1.0],
        ])
        result = fuse_embeddings(
            text_embeddings=text,
            image_embeddings=image,
            text_weight=0.5,
            image_weight=0.5,
        )
        expected = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ])
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )

    def test_text_only_fusion(self):
        text = np.array([
            [3.0, 4.0],
        ])
        result = fuse_embeddings(
            text_embeddings=text,
            image_embeddings=None,
        )
        expected = np.array([
            [0.6, 0.8],
        ])
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )

    def test_missing_image_mask(self):
        text = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        image = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        image_mask = np.array([
            True,
            False,
        ])
        result = fuse_embeddings(
            text_embeddings=text,
            image_embeddings=image,
            image_mask=image_mask,
        )
        expected = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1.0, 0.0],
        ])
        np.testing.assert_allclose(
            result,
            expected,
            atol=1e-6,
        )

    def test_invalid_shapes(self):
        text = np.ones((3, 4))
        image = np.ones((2, 4))
        with pytest.raises(ValueError):
            fuse_embeddings(
                text_embeddings=text,
                image_embeddings=image,
            )

    def test_invalid_image_mask_shape(self):
        text = np.ones((3, 4))
        image = np.ones((3, 4))
        mask = np.array([True, False])
        with pytest.raises(ValueError):
            fuse_embeddings(
                text_embeddings=text,
                image_embeddings=image,
                image_mask=mask,
            )

    @pytest.mark.parametrize(
        "text_weight,image_weight",
        [
            (-0.5, 0.5),
            (0.5, -0.5),
            (0.0, 0.0),
        ],
    )
    def test_invalid_weights(self, text_weight, image_weight):
        text = np.ones((2, 4))
        with pytest.raises(ValueError):
            fuse_embeddings(
                text_embeddings=text,
                text_weight=text_weight,
                image_weight=image_weight,
            )
