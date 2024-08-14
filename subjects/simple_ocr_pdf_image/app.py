# from langchain_community.document_loaders import UnstructuredImageLoader
import os
import easyocr
from tqdm import tqdm

def get_text(reader, image_path):
    result = reader.readtext(image_path, detail=False)
    result = ' '.join(result)

    return result

def extract_content_to_file(reader, extract_filename):
    image_dir = os.path.join('images')
    number_of_images = len(os.listdir(image_dir))
    pbar = tqdm(total=number_of_images)
    with open(extract_filename, 'w') as file:
        for img_filename in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_filename)
            extracted_text = get_text(reader, img_path)
            file.write(extracted_text + '\n')
            pbar.update(1)
        

def main():
    # loader = UnstructuredImageLoader(image_path, mode='single', strategy='fast')
    # docs = loader.load()
    extract_filename = "cauhoi.txt"
    reader = easyocr.Reader(['vi'], gpu=False)
    extract_content_to_file(reader, extract_filename)

if __name__ == "__main__":
    main()
