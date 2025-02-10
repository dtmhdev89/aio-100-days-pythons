import cv2
import os
from ultralytics import solutions

if __name__ == "__main__":
    video_path = os.path.join('samples', 'highway.mp4')
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    # Define region points
    # region_points = [(20, 400), (1080, 400)] # For line counting
    region_points = [
        (430, 700),
        (1600, 700),
        (1600, 1080),
        (430, 1080)
    ]
    # For rectangle region counting: top left, top right, bottom, right, bottom left

    # Video writer
    output_path = os.path.join('run')
    os.makedirs(output_path, exist_ok=True)
    video_output_path = os.path.join(output_path, 'highway_counted.mp4')
    video_writer = cv2.VideoWriter(
        video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # Init Object Counter
    # # Using yolo11n-obb.pt for object counting using YOLO11 OBB model
    counting_model = "yolo11x.pt"
    counter = solutions.ObjectCounter(
        show=False,  # Display the output
        region=region_points,
        model=counting_model
    )

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print('Video frame is empty or '
                  'video processing has been successfully completed')
            break
        
        im0 = counter.count(im0)
        video_writer.write(im0)
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
