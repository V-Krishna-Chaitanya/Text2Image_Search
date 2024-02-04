from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import json

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "your_collection_name"

# Your image paths and embeddings
image_paths = ["C:\\path\\to\\image1.jpg", "C:\\path\\to\\image2.jpg"]  # Example image paths
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Example embeddings

# Function to batch upload points to Qdrant
def batch_upload(embeddings, image_paths):
    points = [
        PointStruct(
            id=str(i),  # Unique ID for each point
            vector=embedding,  # Embedding for the image
            payload={"path": str(path)}  # Any additional metadata, like the image path
        )
        for i, (embedding, path) in enumerate(zip(embeddings, image_paths), start=1)
    ]

    # Perform the upsert operation
    response = qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    print("Upsert response:", response)

# Call the function to upload your embeddings and paths
batch_upload(embeddings, image_paths)
