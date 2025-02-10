from utils import save_detection_results
import os
from ultralytics import YOLOWorld
from ultralytics.engine.results import Boxes


if __name__ == "__main__":
    model_weights = 'yolov8s-world.pt'
    model = YOLOWorld(model_weights)

    # Define custom classes
    custom_classes = ["phone", "glasses", "mask"]
    model.set_classes(custom_classes)

    prediction_img_path = os.path.join('samples', 'vietnam-3.jpg')
    results: Boxes = model.predict(prediction_img_path)
    save_detection_results(results)
