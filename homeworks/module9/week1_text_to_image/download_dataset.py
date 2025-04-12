import os
import urllib.request
import zipfile
import gdown


def download_flickr8k(dataset_dir="Flickr8k"):
    os.makedirs(dataset_dir, exist_ok=True)

    # Danh sách các URL cần tải
    urls = {
        "images": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
        "captions": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    }

    for key, url in urls.items():
        zip_path = os.path.join(dataset_dir, f"{key}.zip")
        extract_path = os.path.join(dataset_dir, key)

        if not os.path.exists(extract_path):
            print(f"📥 Downloading {key} dataset...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"📂 Extracting {key} dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

            os.remove(zip_path)  # Xóa file ZIP sau khi giải nén
    
    print("✅ Dataset downloaded & extracted!")

    return os.path.join(dataset_dir)


def download_cvpr2016_flowers():
    base_path = os.path.join('./cvpr2016_flowers')
    os.makedirs(base_path, exist_ok=True)

    gdown.download(
        id='1JJjMiNieTz7xYs6UeVqd02M3DW4fnEfU',
        output=base_path
    )

    zip_path = os.path.join(base_path, "cvpr2016_flowers.zip")

    count = 1
    while count < 5:
        if not os.path.exists(zip_path):
            count += 1
        else:
            break

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)

    return base_path
