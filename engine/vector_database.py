from dataclasses import dataclass, asdict
from pathlib import Path
import random
import numpy as np
from pymilvus import MilvusClient


@dataclass
class MilvusServer:
    uri: str = "milvus.db"


@dataclass
class EmbeddingCollectionSchema:
    collection_name: str
    vector_field_name: str
    dimension: int
    auto_id: bool
    enable_dynamic_field: bool
    metric_type: str


ImageEmbeddingCollectionSchema = EmbeddingCollectionSchema(
    collection_name="image_embeddings",
    vector_field_name="embedding",
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE",
)

TextEmbeddingCollectionSchema = EmbeddingCollectionSchema(
    collection_name="text_embeddings",
    vector_field_name="embedding",
    dimension=384,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE",
)


class VectorDB:

    def __init__(self, client: MilvusClient = MilvusClient(uri=MilvusServer.uri)):
        self.client = client

    def create_collection(self, schema: EmbeddingCollectionSchema):
        if self.client.has_collection(collection_name=schema.collection_name):
            print(f"Collection {schema.collection_name} already exists")
            return True
            # self.client.drop_collection(collection_name=schema.collection_name)
        print(f"Creating collection {schema.collection_name}")
        self.client.create_collection(**asdict(schema))
        print(f"Collection {schema.collection_name} created")
        return True

    def insert_record(
        self, collection_name: str, embedding: np.ndarray, file_path: str
    ) -> bool:
        try:
            self.client.insert(
                collection_name=collection_name,
                data={"embedding": embedding, "filename": file_path},
            )
        except Exception as e:
            print(f"Error inserting record: {e}")
            return False
        return True
