import os
import nltk
from nltk.data import path as nltk_path
from utils.common_util import get_cache_dir
from utils.constant import DIR_NAME


def nltk_download(corpus_list: list):
    cache_dir = get_cache_dir(subdir=DIR_NAME)
    nltk_download_path = os.path.join(cache_dir, 'nltk_data')
    nltk_path.append(nltk_download_path)
    for corpus_lib in corpus_list:
        nltk.download(corpus_lib, download_dir=nltk_download_path)


if __name__ == "__main__":
    pass
