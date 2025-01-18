from common_libs import torch, np, plt
import datetime
import os
from safetensors.torch import save_file


def de_normalize(img,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)

    return result


@torch.inference_mode()
def display_prediction(model, image, target, device, predicted_path):
    model.eval()
    img = image[None, ...].to(device)
    output = model(img)
    pred = torch.argmax(output, axis=1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("Input Image")
    plt.imshow(de_normalize(image.numpy().transpose(1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title("Prediction")
    plt.imshow(pred.cpu().squeeze())

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("Ground Truth")
    plt.imshow(target)

    plt.savefig(
        os.path.join(predicted_path, f'predicted_img_{get_timestamp()}')
    )
    # plt.show()


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    return test_loss


def get_timestamp():
    timestamp_format = "%Y%m%d_%H%M%S_%f"
    timestamp = datetime.datetime.now().strftime(timestamp_format)

    return timestamp


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return device


def make_results_dir():
    results_path = os.path.join('results')
    os.makedirs(results_path, exist_ok=True)
    print('created results folder')

    return results_path


def make_model_dir():
    model_path = os.path.join('model')
    os.makedirs(model_path, exist_ok=True)

    return model_path


def model_save_in_safetensors(model, safetensors_path):
    save_file(model.state_dict(), safetensors_path)
    print(f"Save successfully at {safetensors_path}")


if __name__ == "__main__":
    pass
