import argparse
from safetensors.torch import save_file, load_model
import numpy as np
import datetime
import torch
import matplotlib.pyplot as plt
import os


# Constant
DEFAULT_SYS_OPTIONS = {
    'download_data': True,
    'train': True,
    'inference': False
}

LABELS = {
    'cat': 0,
    'dog': 1
}

REVERSE_LABELS = {v: k for k, v in LABELS.items()}


def args_parser():
    parser = argparse.ArgumentParser(
        description='More options to run a function'
    )
    parser.add_argument(
        "--show-help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument('k_v_options',
                        nargs='*',
                        help=(
                            "key-value pair arguments."
                            "Currently supported:"
                            "train_stage=true/false, "
                            "inference_stage=true/false"
                        ))
    args = parser.parse_args()
    options = {}
    allowed_keys = DEFAULT_SYS_OPTIONS.keys()

    for item in args.k_v_options:
        try:
            key, value = item.split("=", 1)
            if key in allowed_keys:
                if value.lower() not in ['true', 'false']:
                    raise ValueError(f'Invalid value of {key}.',
                                     'Allowed: true/false')
                value = value.lower() == 'true'
                options[key] = value
        except ValueError as e:
            err_msg = f"Invalid key-value pair: {item}. Use key=value format."
            err_msg = f'Error:\t {e.args[0]}' if e.args else err_msg
            print(err_msg)
            exit(1)

    return options


def get_optional_args():
    sys_options = args_parser()
    default_options = DEFAULT_SYS_OPTIONS.copy()
    if len(sys_options) > 0:
        default_options.update(sys_options)

    return default_options


def model_save_in_safetensors(model, safetensors_path):
    save_file(model.state_dict(), safetensors_path)
    print(f"Save successfully at {safetensors_path}")


def de_normalize(img,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)

    return result


def get_timestamp():
    timestamp_format = "%Y%m%d_%H%M%S_%f"
    timestamp = datetime.datetime.now().strftime(timestamp_format)

    return timestamp


def compute_textbox_coordinate(img, cls):
    img_height, img_width, _ = img.shape
    x_margin = img_width * 0.05  # 5% margin from the left
    y_margin = img_height * 0.05  # 5% margin from the top
    # Get text bounding box (to calculate text height)
    # and add padding for text background
    text_str = f'Prediction: {REVERSE_LABELS[int(cls)]}'
    bbox = dict(facecolor='white', edgecolor='none', pad=1)
    text_bbox = plt.gca().text(
        x_margin, y_margin, text_str,
        color='white', fontsize=12,
        bbox=bbox).get_window_extent()
    text_height = text_bbox.height / plt.gcf().dpi

    # Calculate y-coordinate for bottom margin
    # calculate y with dpi
    y = img_height - text_height * plt.gcf().dpi

    return x_margin, y, bbox, text_str


def predicted_results(model, dataloader, model_path, device='cpu'):
    model.to(device)
    load_model(model, model_path, strict=True, device=device)
    model.eval()
    with torch.no_grad():
        for batch_images, _batch_labels in dataloader:
            scores = model(batch_images)
            predictions = scores.max(1)
            plt.figure(figsize=(10, 8))
            max_img_in_a_row = 4
            rows = int((len(batch_images) // max_img_in_a_row) + 1)
            col = 1
            for img, cls in zip(batch_images, predictions.indices):
                plt.subplot(rows, max_img_in_a_row, col)
                col += 1
                plt.axis('off')
                x, y, bbox, text_str = compute_textbox_coordinate(img=img,
                                                                  cls=cls)
                plt.text(x=x, y=y,
                         s=text_str,
                         color='black',
                         bbox=bbox)
                plt.imshow(de_normalize(img.numpy().transpose(1, 2, 0)))

            result_path = os.path.join('results')
            os.makedirs(result_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    result_path,
                    f'predicted_img_{get_timestamp()}'
                )
            )
            break


def predicted_results_w_bbox(model, dataloader, model_path, device='cpu'):
    pass


if __name__ == "__main__":
    pass
