import os
import urllib.request
import zipfile


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
    
    return os.path.join(dataset_dir)

    print("✅ Dataset downloaded & extracted!")
