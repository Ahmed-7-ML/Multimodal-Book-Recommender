from PIL import Image
import numpy as np
from src.book_recommender.models.clip_encoder import CLIPEncoder

def main():
    # Load the Mdoel once
    encoder = CLIPEncoder()
    
    # Test 1: Single text
    text_embedding = encoder.encode_text(
        "A fantasy book about dragons and magic"
    )
    print("\nSingle Text:")
    print("Shape:", text_embedding.shape)
    print("Norm:", np.linalg.norm(text_embedding))

    # Test 2: Multiple texts
    texts = [
        "A book about artificial intelligence",
        "A historical novel about World War II",
        "A fantasy story about dragons",
    ]
    
    text_embeddings = encoder.encode_texts(
        texts,
        batch_size=2,
    )

    print("\nMultiple Texts:")
    print("Shape:", text_embeddings.shape)

    # Test 3: Single image
    image = Image.new("RGB", (224, 224), color="white")

    image_embedding = encoder.encode_image(image)

    print("\nSingle Image:")
    print("Shape:", image_embedding.shape)
    print("Norm:", np.linalg.norm(image_embedding))

    # Test 4: Multiple images
    images = [
        Image.new("RGB", (224, 224), color="white"),
        Image.new("RGB", (224, 224), color="black"),
    ]

    image_embeddings = encoder.encode_images(
        images,
        batch_size=2,
    )

    print("\nMultiple Images:")
    print("Shape:", image_embeddings.shape)

    # Test 5: Text and image dimensions
    assert text_embedding.shape == (encoder.embedding_dim,)
    assert image_embedding.shape == (encoder.embedding_dim,)

    assert text_embeddings.shape == (3, encoder.embedding_dim)
    assert image_embeddings.shape == (2, encoder.embedding_dim)

    # Check normalization
    assert np.allclose(np.linalg.norm(text_embeddings, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(image_embeddings, axis=1), 1.0)

    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
