from common_libs import torch
import datetime


def get_timestamp():
    timestamp_format = "%Y%m%d_%H%M%S_%f"
    timestamp = datetime.datetime.now().strftime(timestamp_format)

    return timestamp


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return device


if __name__ == "__main__":
    pass
