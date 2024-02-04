from qdrant_client import QdrantClient
from qdrant_client.http.models import CollectionConfig, Distance, VectorParams, PointStruct
import torch
import jso

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "image_embeddings"

# Check if the collection already exists
collections_response = qdrant_client.get_collections()
existing_collections = [col.name for col in collections_response.collections]
if collection_name not in existing_collections:
    # Define the vectors configuration
    vectors_config = VectorParams(
        size=512,  # The dimension of your vectors
        distance=Distance.COSINE  # The distance metric to use for vector comparisons
    )

    # Attempt to create the collection with the specified configuration
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create collection '{collection_name}'. Error: {e}")
else:
    print(f"Collection '{collection_name}' already exists.")

# Diagnostic step: Manually upsert a simplified point
try:
    simple_point = PointStruct(
        id=1,  # Simplified ID
        vector=[0.1, 0.2, 0.3, 0.4],  # Simplified vector
        payload={"example": "test"}  # Simplified payload
    )
    # Perform the upsert operation
    print("Upserting a simplified point for diagnostic purposes...")
    operation_info = qdrant_client.upsert(
        collection_name=collection_name,
        points=[simple_point]
    )
    print("Diagnostic upsert successful:", operation_info)
except Exception as e:
    print("Error during diagnostic upsert:", e)

# Function to batch upload points to Qdrant
def batch_upload(embeddings, metadata, batch_size=100):
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i + batch_size].tolist()  # Convert to list if it's a tensor
        batch_metadata = metadata[i:i + batch_size]
        points = [
            PointStruct(
                id=str(idx),  # Ensure the ID is a string or an integer
                vector=embedding,
                payload={"path": path}
            )
            for idx, (embedding, path) in enumerate(zip(batch_embeddings, batch_metadata), start=i + 1)
        ]

        # Diagnostic print to inspect the first point's structure
        print("First point in batch:", points[:1])

        # Use the `upsert` method as per Qdrant documentation
        try:
            operation_info = qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Optionally wait for the operation to complete
            )
            print("Batch upsert successful:", operation_info)
        except Exception as e:
            print("Error during batch upsert:", e)

# Load embeddings
embeddings = torch.load('embeddings.pt')

# Load image paths
with open('image_paths.json', 'r') as f:
    image_paths = json.load(f)

# Call the batch upload function
batch_upload(embeddings, image_paths)
