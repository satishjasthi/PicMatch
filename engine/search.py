import os
import numpy as np
from typing import List, Tuple
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import sqlite3
from .vector_database import (
    VectorDB,
    ImageEmbeddingCollectionSchema,
    TextEmbeddingCollectionSchema,
)


class ImageSearchModule:
    def __init__(
        self,
        image_embeddings_dir: str,
        original_images_dir: str,
        sqlite_db_path: str = "image_tracker.db",
    ):
        self.image_embeddings_dir = image_embeddings_dir
        self.original_images_dir = original_images_dir
        self.vector_db = VectorDB()
        self.vector_db.create_collection(ImageEmbeddingCollectionSchema)
        self.vector_db.create_collection(TextEmbeddingCollectionSchema)

        self.clip_model = CLIPModel.from_pretrained(
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        )
        self.clip_preprocess = CLIPProcessor.from_pretrained(
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        )
        self.text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.sqlite_conn = sqlite3.connect(sqlite_db_path)
        self._create_sqlite_table()

    def _create_sqlite_table(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS added_images (
                image_name TEXT PRIMARY KEY
            )
        """
        )
        self.sqlite_conn.commit()

    def add_images(self):
        print("Adding images to vector databases")
        cursor = self.sqlite_conn.cursor()

        for filename in tqdm(os.listdir(self.image_embeddings_dir)):
            if filename.startswith("resized_") and filename.endswith("_clip.npy"):
                image_name = filename[
                    8:-9
                ]  # Remove "resized_" prefix and "_clip.npy" suffix

                cursor.execute(
                    "SELECT 1 FROM added_images WHERE image_name = ?", (image_name,)
                )
                if cursor.fetchone() is None:
                    clip_embedding_path = os.path.join(
                        self.image_embeddings_dir, filename
                    )
                    caption_embedding_path = os.path.join(
                        self.image_embeddings_dir, f"resized_{image_name}_caption.npy"
                    )

                    if os.path.exists(clip_embedding_path) and os.path.exists(
                        caption_embedding_path
                    ):
                        with open(clip_embedding_path, "rb") as buffer:
                            image_embedding = np.frombuffer(
                                buffer.read(), dtype=np.float32
                            ).reshape(512)
                        with open(caption_embedding_path, "rb") as buffer:
                            text_embedding = np.frombuffer(
                                buffer.read(), dtype=np.float32
                            ).reshape(384)

                        if self.vector_db.insert_record(
                            ImageEmbeddingCollectionSchema.collection_name,
                            image_embedding,
                            image_name,
                        ):
                            self.vector_db.insert_record(
                                TextEmbeddingCollectionSchema.collection_name,
                                text_embedding,
                                image_name,
                            )
                            cursor.execute(
                                "INSERT INTO added_images (image_name) VALUES (?)",
                                (image_name,),
                            )
                            self.sqlite_conn.commit()

        print("Finished adding images to vector databases")

    def search_by_image(
        self, query_image_path: str, top_k: int = 5, similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        if not os.path.exists(query_image_path):
            print(f"Image file not found: {query_image_path}")
            return []
        try:
            query_image = Image.open(query_image_path)
            query_embedding = self._get_image_embedding(query_image)
            results = self.vector_db.client.search(
                collection_name=ImageEmbeddingCollectionSchema.collection_name,
                data=[query_embedding],
                output_fields=["filename"],
                search_params={"metric_type": "COSINE"},
                limit=top_k,
            ).pop()
            return [(item["entity"]["filename"], item["distance"]) for item in results if item["distance"] >= similarity_threshold]
        except Exception as e:
            print(f"Error processing image: {e}")
            return []

    def search_by_text(
        self, query_text: str, top_k: int = 5,similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        if not query_text.strip():
            print("Empty text query")
            return []
        try:
            query_embedding = self._get_text_embedding(query_text)
            results = self.vector_db.client.search(
                collection_name=TextEmbeddingCollectionSchema.collection_name,
                data=[query_embedding],
                search_params={"metric_type": "COSINE"},
                output_fields=["filename"],
                limit=top_k,
            ).pop()
            return [(item["entity"]["filename"], item["distance"]) for item in results if item["distance"] >= similarity_threshold]
        except Exception as e:
            print(f"Error processing text: {e}")
            return []

    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            image_input = self.clip_preprocess(images=image, return_tensors="pt")[
                "pixel_values"
            ].to(self.clip_model.device)
            image_features = self.clip_model.get_image_features(image_input)
        return image_features.cpu().numpy().flatten()

    def _get_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            embedding = self.text_embedding_model.encode(text).flatten()
        return embedding

    def display_results(self, results: List[Tuple[str, float]]):
        if not results:
            print("No results to display.")
            return

        num_images = min(5, len(results))
        fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
        axes = [axes] if num_images == 1 else axes

        for i, (image_name, similarity) in enumerate(results[:num_images]):
            pattern = os.path.join(
                self.original_images_dir, f"resized_{image_name}" + "*"
            )
            matching_files = glob(pattern)
            if matching_files:
                image_path = matching_files[0]
                img = Image.open(image_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Similarity: {similarity:.2f}")
                axes[i].axis("off")
            else:
                print(f"No matching image found for {image_name}")
                axes[i].text(0.5, 0.5, "Image not found", ha="center", va="center")
                axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def __del__(self):
        if hasattr(self, "sqlite_conn"):
            self.sqlite_conn.close()


if __name__ == "__main__":
    from pathlib import Path
    import requests

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    search = ImageSearchModule(
        image_embeddings_dir=str(PROJECT_ROOT / "data/features"),
        original_images_dir=str(PROJECT_ROOT / "data/images"),
    )
    search.add_images()

    # Search by image
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    raw_image.save(PROJECT_ROOT / "test.jpg")
    image_results = search.search_by_image(str(PROJECT_ROOT / "test.jpg"))
    print("Image search results:")
    search.display_results(image_results)

    # Search by text
    text_results = search.search_by_text("Images of Nature")
    print("Text search results:")
    search.display_results(text_results)