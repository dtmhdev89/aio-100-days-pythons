import os
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from nltk.tokenize import word_tokenize
from collections import Counter
from homeworks.module9.week1_text_to_image.download_dataset import download_flickr8k
from dataset_builder import Flickr8kDataset

download_flickr8k()


from preprocessing_dataset import caption as dataset_captions
from preprocessing_dataset import text as dataset_text
from tokenizers import Tokenizer, pre_tokenizers, trainers, models
from transformers import PreTrainedTokenizerFast
import nltk
nltk.download('punkt_tab')


class TokenizerBuilder():
    def __init__(self, dataset_text, dataset_captions) -> None:
        self.dataset_text = dataset_text
        self.dataset_captions = dataset_captions

    def build(self):
        # Tạo tokenizer dạng word-based
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        trainer = trainers.WordLevelTrainer(
            vocab_size=10000,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>"]
        )

        # Huấn luyện tokenizer
        tokenizer.train_from_iterator(self.dataset_text, trainer)

        # Lưu tokenizer
        tokenizer.save("tokenizer.json")

        # Load từ điển từ tokenizer
        vocab = tokenizer.get_vocab()  # Trích xuất từ điển
        word_to_id = lambda word: vocab.get(word, vocab["<unk>"])

        return tokenizer, vocab, word_to_id


tokenizer_builder = TokenizerBuilder(
    dataset_text=dataset_text,
    dataset_captions=dataset_captions
)

tokenizer, vocab, word_to_id = tokenizer_builder.build()

# Load tokenizer đã train vào PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    unk_token="<unk>", pad_token="<pad>"
)

print("*" * 18)
print("Test tokenizer:---")
print(tokenizer("i go to school"))

transform = transforms.Compose([
    transforms.Resize((8, 8)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = Flickr8kDataset(
    img_dir="/content/Flickr8k/Flicker8k_Dataset",
    captions=dataset_captions,
    tokenizer=tokenizer,
    transform=transform
)

print("*"*20)
print("----Test dataset")
sample = next(iter(dataset))

print(sample)
print(sample['image'].shape)
print(sample['caption'].shape)


# ===========================
# 2️⃣ Word Embeddings + LSTM
# ===========================
class embedding_text(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(embedding_text, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, captions):
        embeds = self.embedding(captions)
        _, (hidden, _) = self.lstm(embeds)
        return hidden[-1]


# ===========================
# 3️⃣ DCGAN Model
# ===========================

class Generator(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 4 * 4),
            #       latent_dim (64) + embed_dim (256)
            # Input:(batch_size,320) - Output:(batch_size,2048)
            nn.ReLU(True),

            nn.Unflatten(1, (128, 4, 4)),
            # Input:(batch_size, 2048) - Output:(batch_size,128,4,4)

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            # outsize = (insize - 1) * stride - 2 * padding + kernel_size
            # outsize = (insize=4 - 1) * stride=2 -2 * padding=1 + kernel_size=4 => outsize = 8
            # Input:(batch_size,128,4,4) - Output:(batch_size,64, 8, 8)
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),
            # Input:(batch_size,64, 8, 8) - Output:(batch_size, 3, 8, 8)
            nn.Tanh()
        )

    def forward(self, noise, caption_embed):
        x = torch.cat((noise, caption_embed), dim=1)
        #  (batch_size,64) + (batch_size,256) = (batch_size,320)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super(Discriminator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            # Input:(batch_size,3,8,8) - Output:(batch_size,64,4,4)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            # Input:(batch_size,64,4,4) - Output:(batch_size,128,2,2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
            # Input:(batch_size,128,2,2) - Output:(batch_size,512)
        )

        self.fc = nn.Linear(512 + embed_dim, 1)
        # Input:(batch_size,512) - Output:(batch_size,1)

    def forward(self, img, caption_embed):
        img_features = self.cnn(img)
        x = torch.cat((img_features, caption_embed), dim=1)
        #  (batch_size,512) + (batch_size,256) = (batch_size,768)
        return torch.sigmoid(self.fc(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
generator = Generator(latent_dim=64, embed_dim=256).to(device)
discriminator = Discriminator(embed_dim=256).to(device)
embed_text = embedding_text(
    vocab_size=len(tokenizer),
    embed_dim=256,
    hidden_dim=256).to(device)

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)
optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)
criterion = nn.BCELoss()

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print("*"*20)
print("Test dataloader")
batch_sample = next(iter(dataloader))
print(batch_sample)
print(batch_sample['image'].shape)
print(batch_sample['caption'].shape)

# Create a noise image
noise = torch.randn(batch_sample['image'].size(0), 64, device=device)
caption_embeddings = embed_text(batch_sample['caption'].to(device))
fake_images = generator(noise, caption_embeddings)


for epoch in range(5):
    for batch in dataloader:
        # load tensor images
        images = batch['image'].to(device)
        # load one-hot vector tokenizer
        captions = batch['caption'].to(device)

        caption_embeddings = embed_text(captions)
        #  (batch_size, 256)
        noise = torch.randn(images.size(0), 64, device=device)
        #  (batch_size, 64)

        fake_images = generator(noise, caption_embeddings)
        #  (batch_size, 3, 8, 8)
        real_labels = torch.ones(images.size(0), 1, device=device)
        #  (batch_size, 1)
        fake_labels = torch.zeros(images.size(0), 1, device=device)
        #  (batch_size, 1)

        real_loss = criterion(
            discriminator(images, caption_embeddings),
            real_labels
        )
        fake_loss = criterion(
            discriminator(fake_images.detach(), caption_embeddings),
            fake_labels
        )
        '''
        Detaches fake_images from the computation graph so that gradients
        do not flow back to the generator during Discriminator training.
        '''

        d_loss = real_loss + fake_loss
        optimizer_D.zero_grad()
        d_loss.backward(retain_graph=True)
        #  retain_graph=True
        optimizer_D.step()

        g_loss = criterion(
            discriminator(fake_images, caption_embeddings),
            real_labels
        )

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/5], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
