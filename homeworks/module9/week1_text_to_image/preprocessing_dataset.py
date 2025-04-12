import os


def process_data():
    captions_file = "Flickr8k/Flickr8k.token.txt"
    image_dir = "Flickr8k/Flicker8k_Dataset"  # Thư mục chứa ảnh
    captions: dict = dict()
    text = []

    with open(captions_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0].split("#")[0]

            # Nếu có ".1" ở cuối file thì loại bỏ
            if img_name.endswith(".1"):
                img_name = img_name[:-2]  # Bỏ ký tự ".1" ở cuối

            # Kiểm tra xem file có tồn tại không
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"⚠️ File không tồn tại: {img_name}")  # Cảnh báo file bị thiếu
                continue  # Bỏ qua file không tồn tại

            caption = parts[1].lower()
            text.append(caption)

            if img_name not in captions:
                captions[img_name] = []
            
            captions[img_name].append(caption)
    
    return captions, text
