import argparse
from safetensors import save_file, load_model


# Constant
DEFAULT_SYS_OPTIONS = {
    'download_data': True,
    'train': True,
    'inference': False
}


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
    allowed_keys = ['train_stage', 'inference_stage']

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


if __name__ == "__main__":
    pass
