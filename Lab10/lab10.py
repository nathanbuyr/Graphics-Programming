import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "youtube_Gr0HpDM8Ki8_1920x1080_h264.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Analyze movement direction
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box.numpy()  # Convert to numpy for easier handling
            center_x = float(x)  # x center point
            center_y = float(y)  # y center point

            # Update track history
            track = track_history[track_id]
            track.append((center_x, center_y))  # Store center point

            # Keep track of movement direction
            if len(track) > 2:  # Need at least 2 points to determine direction
                prev_x = track[-2][0]
                curr_x = track[-1][0]

                if curr_x > prev_x:
                    direction = "Moving Right"
                elif curr_x < prev_x:
                    direction = "Moving Left"
                else:
                    direction = "Stationary"

                # Display direction on the frame
                cv2.putText(
                    annotated_frame,
                    f"ID: {track_id} {direction}",
                    (int(center_x), int(center_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Limit track history to avoid memory issues
            if len(track) > 30:  # Retain history for 30 frames
                track.pop(0)

        # Display the annotated frame
        cv2.namedWindow('YOLO11 Tracking', cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        cv2.resizeWindow('YOLOv11 Tracking', 1240, 700)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()