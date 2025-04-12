from download_dataset import download_cvpr2016_flowers
import os
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from torch.utils.data import DataLoader
from dataset_builder import FlowerDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import torch.nn as nn


extracted_path = download_cvpr2016_flowers()
print(extracted_path)


def load_captions(captions_folder, image_folder):
    captions = {}
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_name = image_file.split('.')[0]
        caption_file = os.path.join(captions_folder, image_name + ".txt")
        with open(caption_file, "r") as f:
            caption = f.readlines()[0].strip()
        if image_name not in captions:
            captions[image_name] = caption

    return captions


captions_folder = os.path.join(
    extracted_path,
    "content",
    "cvpr2016_flowers",
    "captions"
)
image_folder = os.path.join(
    extracted_path,
    "content",
    "cvpr2016_flowers",
    "images"
)

captions = load_captions(captions_folder, image_folder)

# Caption Encoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model_name = "all-mpnet-base-v2"
bert_model = SentenceTransformer(bert_model_name).to(device)


def encode_captions(bert_model, captions):
    encoded_captions = {}
    for image_name in captions.keys():
        caption = captions[image_name]
        encoded_captions[image_name] = {
            'embed': torch.tensor(bert_model.encode(caption)),
            'text': caption
        }

    return encoded_captions
    

encoded_captions = encode_captions(bert_model, captions)


# Preprocess

IMG_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ds = FlowerDataset(
    img_dir=image_folder,
    captions=encoded_captions,
    transform=transform
)


def show_grid(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig(f"image_{str(int(time.time()))}")


BATCH_SIZE = 1024
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
batch_sample = next(iter(dataloader))
show_grid(torchvision.utils.make_grid(batch_sample['image'], normalize=True))


# Model
class Generator(nn.Module):

    def __init__(
        self,
        noise_size,
        feature_size,
        num_channels,
        embedding_size,
        reduced_dim_size
    ):
        super(Generator, self).__init__()
        self.reduced_dim_size = reduced_dim_size

        # 768-->256
        self.textEncoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_size,
                out_features=reduced_dim_size
            ),
            nn.BatchNorm1d(num_features=reduced_dim_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.upsamplingBlock = nn.Sequential(
            # 256+100 --> 1024
            nn.ConvTranspose2d(
                noise_size + reduced_dim_size,
                feature_size * 8,
                4,
                1,
                0,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 1024 --> 512
            nn.ConvTranspose2d(
                feature_size * 8,
                feature_size * 4,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(True),

            # 512 --> 256
            nn.ConvTranspose2d(
                feature_size * 4,
                feature_size * 2,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(True),

            # 256 --> 128
            nn.ConvTranspose2d(
                feature_size * 2,
                feature_size,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # 128 --> 128
            nn.ConvTranspose2d(
                feature_size,
                feature_size,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(True),

            # 128 --> 3
            nn.ConvTranspose2d(
                feature_size,
                num_channels,
                4,
                2,
                1,
                bias=False
            ),
            nn.Tanh()

        )

    def forward(self, noise, text_embeddings):
        encoded_text = self.textEncoder(text_embeddings)
        concat_input = torch.cat([noise, encoded_text], dim=1)\
                            .unsqueeze(2).unsqueeze(2)
        output = self.upsamplingBlock(concat_input)

        return output
    

class Discriminator(nn.Module):

    def __init__(
        self,
        num_channels,
        feature_size,
        embedding_size,
        reduced_dim_size
    ):
        super(Discriminator, self).__init__()
        self.reduced_dim_size = reduced_dim_size

        self.imageEncoder = nn.Sequential(
            # 3 -> 128
            nn.Conv2d(
                num_channels,
                feature_size,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 128
            nn.Conv2d(
                feature_size,
                feature_size,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 256
            nn.Conv2d(
                feature_size,
                feature_size * 2,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 -> 512
            nn.Conv2d(
                feature_size * 2,
                feature_size * 4,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 512 -> 1024
            nn.Conv2d(
                feature_size * 4,
                feature_size * 8,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.textEncoder = nn.Sequential(
            nn.Linear(
                in_features=embedding_size,
                out_features=reduced_dim_size
            ),
            nn.BatchNorm1d(num_features=reduced_dim_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.finalBlock = nn.Sequential(
            nn.Conv2d(
                feature_size * 8 + reduced_dim_size,
                1,
                4,
                1,
                0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, input_img, text_embeddings):
        image_encoded = self.imageEncoder(input_img)
        text_encoded = self.textEncoder(text_embeddings)
        replicated_text = text_encoded.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        concat_layer = torch.cat([image_encoded, replicated_text], 1)

        x = self.finalBlock(concat_layer)

        return x.view(-1, 1)


generator = Generator(100, 128, 3, 768, 256).to(device)
discriminator = Discriminator(3, 128, 768, 256).to(device)

bce_loss = nn.BCELoss()
plt_o_text_embeddings = ds[0]['embed_caption'].unsqueeze(0)
fixed_noise = torch.randn(size=(1, 100))

show_grid(torchvision.utils.make_grid(ds[0]['image'], normalize=True))


# Training
def plot_output(generator, plt_o_text_embeddings):
    plt.clf()
    with torch.no_grad():

        generator.eval()
        test_images = generator(
            fixed_noise.to(device),
            plt_o_text_embeddings.to(device)
        )
        generator.train()

        grid = torchvision.utils.make_grid(test_images.cpu(), normalize=True)
        show_grid(grid)


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

epochs = 500

for epoch in range(epochs):
    d_losses, g_losses = [], []

    epoch_time = time.time()

    for batch in dataloader:
        images = batch['image'].to(device)  # Đưa ảnh lên GPU
        embed_captions = batch['embed_caption'].to(device)  # Đưa captions lên GPU (đã là tensor)

        real_labels = torch.ones(images.size(0), 1, device=device)
        fake_labels = torch.zeros(images.size(0), 1, device=device)

        # Training the discriminator
        optimizer_D.zero_grad()

        noise = torch.randn(size=(images.size(0), 100), device=device)
        fake_images = generator(noise, embed_captions)

        real_labels = torch.ones(images.size(0), 1, device=device)
        fake_labels = torch.zeros(images.size(0), 1, device=device)

        real_loss = criterion(
            discriminator(images, embed_captions),
            real_labels
        )
        fake_loss = criterion(
            discriminator(fake_images.detach(), embed_captions),
            fake_labels
        )
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        d_losses.append(d_loss.item())

        # Training generator
        optimizer_G.zero_grad()
        noise = torch.randn(size=(images.size(0), 100), device=device)
        fake_images = generator(noise, embed_captions)

        g_loss = criterion(
            discriminator(fake_images, embed_captions),
            real_labels
        )

        g_loss.backward()
        optimizer_G.step()

        g_losses.append(g_loss.item())

    avg_d_loss = sum(d_losses)/len(d_losses)
    avg_g_loss = sum(g_losses)/len(g_losses)

    if (epoch+1) % 10 == 0:
        plot_output(generator, plt_o_text_embeddings)

    print('Epoch [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
        epoch+1, epochs,
        avg_d_loss,
        avg_g_loss,
        time.time() - epoch_time)
