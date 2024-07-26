import numpy as np
from time import  sleep
from typing import List, Tuple
from milvus import default_server
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType


class VectorDBModule:
    _milvus_server = default_server
    _is_connected = False

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection = None

    @classmethod
    def connect_to_milvus(cls, host: str = "localhost", port: str = "19530"):
        if not cls._is_connected:
            print(f"Starting Milvus default server at {host}:{port}")
            cls._milvus_server.start()
            sleep(3)
            print(f"Connected to Milvus default server at {host}:{port}")
            connections.connect(host="127.0.0.1", port=cls._milvus_server.listen_port)
            cls._is_connected = True
        else:
            print("Already connected to Milvus server")

    def create_collection(self, embedding_dim: int, embedding_type: str):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name=f"{embedding_type}_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim,
            ),
            FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=200),
        ]
        schema = CollectionSchema(
            fields, f"{embedding_type.capitalize()} search collection"
        )
        self.collection = Collection(self.collection_name, schema)

    def create_index(self, embedding_type: str):
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.collection.create_index(
            field_name=f"{embedding_type}_embedding", index_params=index_params
        )

    def load_collection(self):
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def add_embeddings(
        self, embeddings: List[np.ndarray], item_names: List[str], embedding_type: str
    ):
        entities = [[embedding.tolist() for embedding in embeddings], item_names]
        self.collection.insert(entities)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        similarity_threshold: float,
        embedding_type: str,
    ) -> List[Tuple[str, float]]:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field=f"{embedding_type}_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["item_name"],
        )
        filtered_results = [
            (hit.entity.get("item_name"), hit.distance)
            for hit in results[0]
            if hit.distance >= similarity_threshold
        ]
        return filtered_results


class VectorDBFactory:
    @staticmethod
    def create_db(
        collection_name: str, embedding_dim: int, embedding_type: str
    ) -> VectorDBModule:
        VectorDBModule.connect_to_milvus()  # This will only start the server once
        db = VectorDBModule(collection_name)
        db.create_collection(embedding_dim, embedding_type)
        db.create_index(embedding_type)
        db.load_collection()
        return db
