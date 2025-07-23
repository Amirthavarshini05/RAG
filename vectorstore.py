from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid

client = QdrantClient(path="qdrant_data")

def setup_qdrant(vector_size=384):  # match embed size
    client.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def add_chunks_to_qdrant(chunks):
    points = []
    for chunk in chunks:
        points.append(
            PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=chunk["embedding"],
                payload = chunk["payload"]

            )
        )
    client.upsert(collection_name="docs", points=points)

def search(query_embedding, top_k=1):
    return client.search(
        collection_name="docs",
        query_vector=query_embedding,
        limit=top_k
    )
