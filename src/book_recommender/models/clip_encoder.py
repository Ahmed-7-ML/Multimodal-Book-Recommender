from transformers import CLIPModel, CLIPProcessor
from book_recommender.config import settings
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

np.random.seed(settings.SEED)
MODEL_NAME = settings.CLIP_MODEL_NAME
DEVICE = settings.DEVICE
IMAGE_BATCH_SIZE = settings.IMAGE_BATCH_SIZE
TEXT_BATCH_SIZE = settings.TEXT_BATCH_SIZE

class CLIPEncoder:
    """
    - Load the Model Once and all Methods will use it.
    - Encode Text & Image into a shared CLIP Embedding Space.
    - Use the Model in Eval/Inference Mode -> We don't need Gradient Calculation.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        # 1. Select a Device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # 2. Load CLIP Model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        # 3. Load CLIP Processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 4. Set Model to Evaluation Mode
        self.model.eval()
        
        # 5. Store Model Information
        self.model_name = model_name
        self.embedding_dim = self.model.config.projection_dim
        
        # 7. Display some information
        print(f"CLIP loaded: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {self.embedding_dim}")

    @staticmethod
    def _extract_features(outputs) -> torch.Tensor:
        """
        Unwrap CLIP outputs to a Tensor.

        transformers >= 4.44 / 5.x returns BaseModelOutputWithPooling
        (with projected features in pooler_output) instead of a Tensor,
        so F.normalize() fails with:
            AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
        """
        if isinstance(outputs, torch.Tensor):
            return outputs
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            # [batch, seq, dim] -> CLS token [batch, dim]
            return last_hidden[:, 0, :]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise TypeError(f"Unexpected CLIP output type: {type(outputs)!r}")

    @staticmethod
    def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply L2 Normalization to the Embeddings
        """
        embeddings = CLIPEncoder._extract_features(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)

    @staticmethod
    def _to_numpy(embeddings: torch.Tensor) -> np.ndarray:
        """
        Convert Embeddings to Float32 Numpy Array
        """
        return embeddings.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_text(self, text:str) -> np.ndarray:
        """
        Encode Single Text into Normalized Embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be Empty.")
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        embeddings = self.model.get_text_features(**inputs)
        embeddings = self._normalize(embeddings)
        return self._to_numpy(embeddings)[0]

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int=TEXT_BATCH_SIZE) -> np.ndarray:
        """
        Encode multiple texts using Batch Processing.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty.")

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty values.")
        
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start+batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }
            embeddings = self.model.get_text_features(**inputs)
            embeddings = self._normalize(embeddings)
            all_embeddings.append(self._to_numpy(embeddings))
        # Join a sequence of arrays along an existing axis.
        return np.concatenate(all_embeddings, axis=0)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single PIL image into a normalized embedding.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")
        image = image.convert("RGB")
        inputs = self.processor(
            images=image,
            return_tensors='pt'
        )
        pixel_values = inputs['pixel_values'].to(self.device)
        embeddings = self.model.get_image_features(pixel_values=pixel_values)
        embeddings = self._normalize(embeddings)
        return self._to_numpy(embeddings)[0]

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image], batch_size: int = IMAGE_BATCH_SIZE,) -> np.ndarray:
        """
        Encode multiple PIL images using batch processing.
        """
        if not images:
            raise ValueError("Images list cannot be empty.")

        if any(not isinstance(img, Image.Image) for img in images):
            raise TypeError("All items must be PIL Images.")
        
        all_embeddings = []
        for start in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[start: start+batch_size]]
            inputs = self.processor(
                images=batch,
                return_tensors='pt'
            )
            pixel_values = inputs['pixel_values'].to(self.device)
            embeddings = self.model.get_image_features(pixel_values=pixel_values)
            embeddings = self._normalize(embeddings)
            all_embeddings.append(self._to_numpy(embeddings))
        return np.concatenate(all_embeddings, axis=0)
