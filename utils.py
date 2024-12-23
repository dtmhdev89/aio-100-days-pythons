import platform
import os
from typing import Optional
import nltk
from nltk.data import path as nltk_path


DIR_NAME = 'aio_100_days_python'


class UnsupportedOSError(Exception):
    """Custom exception for unsupported operating systems."""
    def __init__(self,
                 message={
                    "Unsupported OS. Please contact"
                    "app's support for more details"
                 }):
        super().__init__(message)


def get_cache_dir(subdir: Optional[str] = None):
    if platform.system() == "Windows":
        return _compose_dir(os.getenv("APPDATA"), subdir)
    elif platform.system() == "Darwin":
        return _compose_dir(os.path.expanduser("~/Library/Caches"), subdir)
    elif platform.system() == "Linux":
        return _compose_dir(os.path.join(os.path.expanduser("~"), ".cache"),
                            subdir)
    else:
        raise(UnsupportedOSError)


def _compose_dir(base_dir, subdir):
    composed_dir = os.path.join(
        '/'.join(list(filter(None, [base_dir, subdir])))
    )
    return composed_dir


def nltk_download(corpus_list: list):
    cache_dir = get_cache_dir(subdir=DIR_NAME)
    nltk_download_path = os.path.join(cache_dir, 'nltk_data')
    nltk_path.append(nltk_download_path)
    for corpus_lib in corpus_list:
        nltk.download(corpus_lib, download_dir=nltk_download_path)


if __name__ == "__main__":
    pass
