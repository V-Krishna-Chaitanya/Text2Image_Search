import streamlit as st
from PIL import Image
import clip
import torch
from qdrant_client import QdrantClient


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "image_embeddings"

def search_images(text_query):
    text_inputs = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().numpy().tolist()[0]
    
    
    search_response = qdrant_client.search(
        collection_name=collection_name,
        query_vector=text_features,
        limit=5,
    )

    images_path = []
    for hit in search_response:
        image_path = [hit.payload['path'] for hit in search_response] 
        images_path.append(image_path)
    
    return images_path


def main():
    st.title("Text to Image Search")
    user_input = st.text_input("Enter your search query:", "")
    if user_input:
        images_paths = search_images(user_input)  # This should return a list of image paths
        if images_paths:
            # Assuming images_paths is a list of lists, flatten it to a single list of paths
            flat_list_of_paths = [item for sublist in images_paths for item in sublist]

            # Create columns for the number of images we want to display
            cols = st.columns(5)  # Adjust the number of columns to the number of images you expect
            for col, image_path in zip(cols, flat_list_of_paths[:5]):  # Display only the first 5 images
                try:
                    image = Image.open(image_path)
                    col.image(image, use_column_width=True)
                except Exception as e:
                    col.error(f"Error loading image: {e}")
        else:
            st.write("No images found matching the query.")

if __name__ == "__main__":
    main()