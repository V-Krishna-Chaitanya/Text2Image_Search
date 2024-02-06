from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Distance
import torch
import json

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "image_embeddings"

# Ensure the collection exists with the desired configuration
if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

# Load embeddings and image paths
embeddings = torch.load('embeddings.pt').tolist()  # Convert tensor to list
with open('image_paths.json', 'r') as f:
    image_paths = json.load(f)

# Upsert all embeddings into Qdrant
for i, (embedding, path) in enumerate(zip(embeddings, image_paths)):
    point = PointStruct(
        id=i,  # Use loop index as integer ID
        vector=embedding,
        payload={"path": path}
    )
    response = qdrant_client.upsert(
        collection_name=collection_name,
        points=[point],
        wait=True
    )

print("Upsert completed")
