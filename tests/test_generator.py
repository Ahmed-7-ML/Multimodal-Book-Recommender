
import numpy as np
import pandas as pd
import pytest
from book_recommender.embeddings import EmbeddingGenerator
import book_recommender.embeddings.generator as generator_module


@pytest.fixture
def generator(mock_encoder, monkeypatch):
    # Mock the text description builder.
    monkeypatch.setattr(
        generator_module,
        "create_book_description",
        lambda row: f"{row['title']} by {row['author']}",
    )
    return EmbeddingGenerator(
        encoder=mock_encoder,
        text_batch_size=2,
        image_batch_size=2,
    )

class TestEmbeddingGenerator:
    def test_build_texts(self, generator, sample_books):
        texts = generator.build_texts(sample_books)
        assert len(texts) == len(sample_books)
        assert texts[0] == "Machine Learning by Author A"

    def test_text_embeddings_shape(
        self,
        generator,
        sample_books,
    ):
        texts = generator.build_texts(sample_books)
        embeddings = generator.generate_text_embeddings(texts)
        assert embeddings.shape == (5, 4)
        assert embeddings.dtype == np.float32

    def test_text_batch_processing(
        self,
        generator,
        sample_books,
        mock_encoder,
    ):
        texts = generator.build_texts(sample_books)
        generator.generate_text_embeddings(texts)
        # 5 samples with batch_size=2.
        assert mock_encoder.text_batches == [2, 2, 1]

    def test_image_embeddings_shape(
        self,
        generator,
        sample_images,
        mock_encoder,
    ):
        paths = list(sample_images.values())
        embeddings, image_mask = (
            generator.generate_image_embeddings(paths)
        )
        assert embeddings.shape == (3, 4)
        assert image_mask.shape == (3,)
        assert image_mask.all()
        assert embeddings.dtype == np.float32

    def test_image_batch_processing(
        self,
        generator,
        sample_images,
        mock_encoder,
    ):
        paths = list(sample_images.values())
        generator.generate_image_embeddings(paths)
        # 3 images with batch_size=2.
        assert mock_encoder.image_batches == [2, 1]

    def test_missing_images(self, generator):
        paths = [
            None,
            None,
            None,
        ]
        embeddings, image_mask = (
            generator.generate_image_embeddings(paths)
        )
        assert embeddings.shape == (3, 4)
        assert not image_mask.any()
        assert np.count_nonzero(embeddings) == 0

    def test_invalid_image_path(
        self,
        generator,
        tmp_path,
    ):
        invalid_path = tmp_path / "missing.jpg"
        embeddings, image_mask = (
            generator.generate_image_embeddings([
                invalid_path,
            ])
        )
        assert not image_mask[0]
        assert np.all(embeddings[0] == 0)

    def test_generate_pipeline(
        self,
        generator,
        sample_books,
        sample_images,
    ):
        results = generator.generate(
            books=sample_books,
            image_paths=sample_images,
        )
        assert len(results["ISBN"]) == 5
        assert results["text_embeddings"].shape == (5, 4)
        assert results["image_embeddings"].shape == (5, 4)
        assert results["book_embeddings"].shape == (5, 4)
        assert results["image_mask"].sum() == 3
        assert results["image_mask"].tolist() == [
            True,
            False,
            True,
            False,
            True,
        ]

    def test_final_embeddings_are_normalized(
        self,
        generator,
        sample_books,
        sample_images,
    ):
        results = generator.generate(
            books=sample_books,
            image_paths=sample_images,
        )
        embeddings = results["book_embeddings"]
        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )
        np.testing.assert_allclose(
            norms,
            np.ones(5),
            atol=1e-5,
        )

    def test_isbn_order_is_preserved(
        self,
        generator,
        sample_books,
    ):
        results = generator.generate(
            books=sample_books,
        )
        np.testing.assert_array_equal(
            results["ISBN"],
            sample_books["ISBN"].astype(str).to_numpy(),
        )

    def test_duplicate_isbn_raises_error(
        self,
        generator,
        sample_books,
    ):
        books = pd.concat(
            [sample_books, sample_books.iloc[[0]]],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="unique"):
            generator.generate(books)

    def test_missing_required_column_raises_error(
        self,
        generator,
        sample_books,
    ):
        books = sample_books.drop(columns=["author"])
        with pytest.raises(ValueError, match="Missing required"):
            generator.generate(books)

    def test_empty_dataframe_raises_error(
        self,
        generator,
        sample_books,
    ):
        books = sample_books.iloc[0:0]
        with pytest.raises(ValueError, match="empty"):
            generator.generate(books)

    def test_missing_isbn_raises_error(
        self,
        generator,
        sample_books,
    ):
        books = sample_books.copy()
        books.loc[0, "ISBN"] = None
        with pytest.raises(ValueError, match="ISBN"):
            generator.generate(books)
