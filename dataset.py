import json
import os
import cv2
import torch
from torch.utils.data import Dataset

class CROHMEDataset(Dataset):
    def __init__(self, json_path, image_root, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = os.path.normpath(item["image"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.resize(img, (256, 64))
        img = img.astype("float32") / 255.0
        img = torch.tensor(img).unsqueeze(0)
        img = img.repeat(3, 1, 1)

        tgt = torch.tensor(
            self.tokenizer.encode(item["latex"]),
            dtype=torch.long
        )

        return img, tgt, img_path
