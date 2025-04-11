import os
from PIL import Image
import numpy as np


class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions, tokenizer, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load captions
        self.captions = captions

        self.img_names = list(self.captions.keys())

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = np.random.choice(self.captions[img_name])  # Chọn caption ngẫu nhiên
        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=20,
            return_tensors="pt")['input_ids'][0]
        return {
            'image': image,
            'caption': encoded_caption
        }
