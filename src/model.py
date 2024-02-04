import torch
from PIL import Image
import clip
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def generate_image_embeddings(data_loader, model):
    model.eval()
    embeddings = []
    image_paths = []  # Store paths of the images to save later
    with torch.no_grad():
        for images, paths in data_loader:  # Unpack the tuple returned by the dataloader
            images = torch.stack([preprocess(Image.open(path).convert('RGB')) for path in paths]).to(device)
            image_features = model.encode_image(images)
            embeddings.append(image_features.cpu())
            image_paths.extend(paths)
    embeddings = torch.cat(embeddings, dim=0)

    print("Image Embeddings:")
    print(embeddings)

    # Save embeddings and paths
    torch.save(embeddings, 'embeddings.pt')
    with open('image_paths.json', 'w') as f:
        json.dump(image_paths, f)

    return embeddings

if __name__ == "__main__":
    from data_loader import dataloader
    generate_image_embeddings(dataloader, model)
