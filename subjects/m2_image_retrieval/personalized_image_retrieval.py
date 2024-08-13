from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.request
import time
import os
from tqdm import tqdm
import concurrent.futures
import json
from PIL import Image
import shutil
from collections import defaultdict

class ImageDownloader:
    def __init__(self, json_file, download_dir='Dataset', max_workers=4, delay=1) -> None:
        self.json_file = json_file
        self.download_dir = download_dir
        self.max_workers = max_workers
        self.delay = delay # to send periodly to prevent server filter out your IP
        self.filename = set()
        self.setup_directory()
    
    def setup_directory(self):
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def read_json(self):
        """
        Read JSON file and return the data

        Returns:
        data (dict): The data read from the JSON file
        """
        with open(self.json_file, 'r') as file:
            data = json.load(file)

        return data

    def is_valid_url(self, url):
        """
        Check if a url is valid

        Parameters:
        url (str): The URL to be checked

        Returns:
        bool: True if the URL is valid, otherwise False.
        """
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200 and 'image' in response.info().get_content_type():
                    return True
        except:
            return False
    
    def download_image(self, url, category, term, pbar):
        """
        Download the image form the given URL.

        Parameters:
        url (str): The URL of the image to be downloaded
        category (str): The category of the image
        term (str): The term or keyword associated with the image.
        pbar (tqdm): The progress bar object.

        Returns:
        str: A message indicating the status of the download
        """
        if not self.is_valid_url(url):
            pbar.update(1)
            
            return f"Invalid URL: {url}"
        
        category_dir = os.path.join(self.download_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        term_dir = os.path.join(category_dir, term)
        if not os.path.exists(term_dir):
            os.makedirs(term_dir)

        filename = os.path.join(term_dir, os.path.basename(urlparse(url).path))
        self.filename.add(filename)

        try:
            urllib.request.urlretrieve(url, filename)
            pbar.update(1)

            return f"Downloaded: {url}"
        except Exception as e:
            pbar.update(1)
            return f"Failed to download {url}: {str(e)}"
    
    def download_images(self):
        """
        Download images from JSON file

        Returns:
        None
        """
        data = self.read_json()
        download_tasks = []

        total_images = sum(len(urls) for terms in data.values() for urls in terms.values())
        with tqdm(total=total_images, desc="Downloading images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for category, terms in data.items():
                    for term, urls in terms.items():
                        for url in urls:
                            download_tasks.append(executor.submit(self.download_image, url, category, term, pbar))
                            time.sleep(self.delay)
                
                for future in concurrent.futures.as_completed(download_tasks):
                    print(future.result())
        
        self.export_filename()
    
    def export_filename(self):
        """
        Export filename directory into a text file

        Returns:
        None
        """
        with open('filename.txt', 'w') as file:
            for filename in sorted(self.filename):
                file.write(f"{filename}\n")

class UrlScraper:
    def __init__(self, url_template, max_images=50, max_workers=4) -> None:
        self.url_template = url_template
        self.max_images = max_images
        self.max_workers = max_workers
        self.setup_environment()
    
    def setup_environment(self):
        os.environ['PATH'] += ':/usr/lib/chromium-browser/'
        os.environ['PATH'] += ':/usr/lib/chromium-browser/chromedriver/'

    def get_url_images(self, term):
        """
        Crawl the urls of images by term

        Parameters:
        term(str): List of urls of images
        """
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)

        url = self.url_template.format(search_term=term)
        driver.get(url)

        urls = []
        more_content_available = True

        pbar = tqdm(total=self.max_images, desc=f'Fetching images for {term}')

        while len(urls) < self.max_images and more_content_available:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            img_tags = soup.find_all('img')

            for img in img_tags:
                if len(urls) >= self.max_images:
                    break
                if 'src' in img.attrs:
                    href = img.attrs['src']
                    img_path = urljoin(url, href)
                    img_path = img_path.replace("_m.jpg", "_b.jpg").replace("_n.jpg", "_b.jpg").replace("_w.jpg", "_b.jpg")
                    
                    if img_path == "https://combo.staticflickr.com/ap/build/images/getty/IStock_corporate_logo.svg":
                        continue

                    urls.append(img_path)
                    pbar.update(1)
            
            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(By.XPATH, '//button[@id="yui_3_16_0_1_1721642285931_28620"]')
                )
                load_more_button.click()
                time.sleep(2)
            except:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                new_soup = BeautifulSoup(driver.page_source, "html.parser")
                new_img_tags = new_soup.find_all("img", loading_='lazy')
                if len(new_img_tags) == len(img_tags):
                    more_content_available = False
                
                img_tags = new_img_tags
        
        pbar.close()
        driver.quit()

        return urls

    def scrape_urls(self, categories):
        """
        Call get_url_images method to get all urls of any object in categories

        Parameters:
        categories (dictionary): the dict of all object we need to collect image with format
            categories{"name_object": [value1, value2, ...]}
        
        Returns:
        all_urls (dictionary): dictionary of urls of images
        """

        all_urls = {category: {} for category in categories}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_term = {executor.submit(self.get_url_images, term): (category, term) for category, terms in categories.items() for term in terms}

            for future in tqdm(concurrent.futures.as_completed(future_to_term), total=len(future_to_term), desc="Overal Progress"):
                category, term = future_to_term[future]
                try:
                    urls = future.result()
                    all_urls[category][term] = urls
                    print(f"\nNumber of images retrieved for {term}: {len(urls)}")
                except Exception as exc:
                    print(f"\n{term} generated as an exception: {exc}")
        
        return all_urls
    
    def save_to_file(self, data, filename):
        """
        Save data to JSON file.

        Parameters:
        data (dict): The data to be saved
        filename (str): The name of JSON file

        Returns:
        None
        """

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Data saved to {filename}")

def check_and_preprocess_image(image_dir):
    """
    Check and preprocess images in the specific directory

    Parameters:
    image_dir (str): The directory containing the images to be checked and preprocessed

    Returns:
    None
    """
    for root, _, files in os.walk(image_dir):
        for img_file in files:
            file_path = os.path.join(root, img_file)
            try:
                with Image.open(file_path) as img:
                    if img.size[0] < 50 or img.size[1] < 50:
                        os.remove(file_path)
                        print(f"Deleted {file_pathe}: Image's too small ({img.size[0]}x{img.size[1]})")
                        continue

                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(file_path)
                        print(f"Converted {file_path} to RGB")

            except Exception as e:
                os.remove(file_path)
                print(f"Delete {file_path}: Not an image or corrupted files ({str(e)})")

def restructure_images(source_dir, train_dir, test_dir):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_files = defaultdict(list)

    with open('filename.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('/')
                class_name = parts[2]
                class_files[class_name].append(line)

    for class_name, files in class_files.items():
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        shutil.copy(files[0], test_class_dir)
        [shutil.copy(file_path, train_class_dir) for file_path in files[1:20]]
    
    print("Dataset organization completed")
        

def main():
    categories = {
        "animal": [
            "Monkey", "Elephant", "cows", "Cat", "Dog", "bear", "fox", "Civet", "Pangolins",
            "Rabbit", "Bats", "Whale", "Cock", "Owl", "flamingo", "Lizard", "Turtle", "Snake", "Frog",
            "Fish", "shrimp", "Crab", "Snail", "Coral", "Jellyfish", "Butterfly", "Flies",
            "Mosquito", "Ants", "Cockroaches", "Spider", "scorpion", "tiger", "bird", "horse", "pig",
            "Alligator", "Alpaca", "Anteater", "donkey", "Bee", "Buffalo", "Camel", "Caterpillar",
            "Cheetah", "Chicken", "Dragonfly", "Duck", "panda", "Giraffe"
        ],
        "plant": [
            "Bamboo", "Apple", "Apricot", "Banana", "Bean", "Wildflower", "Flower", "Mushroom",
            "Weed", "Fern", "Reed", "Shrub", "Moss", "Grass", "Palmtree", "Corn", "Tulip", "Rose",
            "Clove", "Dogwood", "Durian", "Ferns", "Fig", "Flax", "Frangipani", "Lantana", "Hibiscus",
            "Bougainvillea", "Pea", "OrchidTree", "RangoonCreeper", "Jackfruit", "Cottonplant",
            "Corneliantree", "Coffeeplant", "Coconut", "wheat", "watermelon", "radish", "carrot"
        ],
        "furniture": [
            "bed", "cabinet", "chair", "chests", "clock", "desks", "table", "Piano", "Bookcase",
            "Umbrella", "Clothes", "cart", "sofa", "ball", "spoon", "Bowl", "fridge", "pan", "book"
        ],
        "scenery": [
            "Cliff", "Bay", "Coast", "Mountains", "Forests", "Waterbodies", "Lake", "desert",
            "farmland", "river", "hedges", "plain", "sky", "cave", "cloud", "flowergarden","glacier",
            "grassland", "horizon", "lighthouse", "plateau", "savannah", "valley", "volcano", "waterfall"
        ]
    }

    urltopic = {"flickr": "https://www.flickr.com/search/?text={search_term}"}
    json_filename = 'image_urls.json'
    if not os.path.isfile(json_filename):
        scraper = UrlScraper(url_template=urltopic["flickr"], max_images=20, max_workers=5)
        image_urls = scraper.scrape_urls(categories=categories)
        scraper.save_to_file(image_urls, json_filename)

    download_dir = 'Dataset'
    downloader = ImageDownloader(json_file="image_urls.json", download_dir=download_dir, max_workers=4, delay=1)

    if len(os.listdir(download_dir)) < 4:
        downloader.download_images()

    image_dir = 'Dataset'
    check_and_preprocess_image(image_dir)
    
    train_dir = 'images_data/train'
    test_dir = 'images_data/test'
    restructure_images(source_dir=image_dir, train_dir=train_dir, test_dir=test_dir)

if __name__ == "__main__":
    main()
