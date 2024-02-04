import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, fname))]
        # Filter out non-existing paths
        self.image_paths = [path for path in self.image_paths if os.path.exists(path)]
        if len(self.image_paths) == 0:
            logging.warning("No images found in the specified directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            logging.error(f"File not found: {img_path}, skipping.")
            return None  # You might need to handle this None in your dataloader
        return image, img_path

dataset = CustomImageDataset(image_dir=r"C:\Users\V Chaitanya\Documents\GitHub\Text_to_image_Search\data\subfolder-0\0", transform=preprocess)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: zip(*x))  # Adjust the collate_fn to handle the tuple of images and paths
