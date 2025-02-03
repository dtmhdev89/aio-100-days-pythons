from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os


if __name__ == "__main__":
    yolo_weights_name = "yolo11l.pt"
    model = YOLO(yolo_weights_name)
    video_path = "samples/vietnam.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    video_name = video_path.split('/')[-1]
    os.makedirs(os.path.join('run'), exist_ok=True)
    output_path = os.path.join(f"run/{video_name.split('.')[0]}_tracked.mp4")
    fourrc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourrc, fps, (width, height))

    # Store tracked history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, show=False)

            # Get the boxes and track IDs (with error handling)
            boxes = results[0].boxes.xywh.cpu()
            try:
                track_ids = results[0].boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                else:
                    # No track found in this frame
                    track_ids = []
            except AttributeError:
                # Handle cases where tracking failure
                track_ids = []
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks only if we have valid tracking data
            if track_ids:
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    # x, y center point
                    track.append((float(x), float(y)))

                    # retain 30 tracks for 30 frames
                    if len(track) > 120:
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=4
                    )
            # Write the frame to output video
            out.write(annotated_frame)
        else:
            # Break loop if the end of video is reached
            break
    
    # Release everything
    cap.release()
    out.release()
    print(f'Video has been saved to {output_path}')

