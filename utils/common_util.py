import platform
import os
from typing import Optional


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


if __name__ == "__main__":
    pass
