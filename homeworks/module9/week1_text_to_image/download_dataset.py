import os
import urllib.request
import zipfile
import gdown
import time


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

    max_attempts = 3
    retry_delay = 5

    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"Attempting download #{attempt}...")

        try:
            gdown.download(
                id='1JJjMiNieTz7xYs6UeVqd02M3DW4fnEfU',
                use_cookies=True
            )
            zip_path = os.path.join("./", "cvpr2016_flowers.zip")

            if os.path.exists(zip_path):
                print("Download successful!")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(base_path)

                os.remove(zip_path)
                
                return base_path
            else:
                print("Download failed to produce the ZIP file.")
                time.sleep(retry_delay)
        except Exception as e:
            print(f"An error occurred during download: {e}")
            time.sleep(retry_delay)

    return None
