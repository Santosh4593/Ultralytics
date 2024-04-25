import cv2
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import torch

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Move model to CUDA if available
if torch.cuda.is_available():
    model = model.to(torch.device('cuda:0'))

# RTSP stream URL
rtsp_url = "rtsp://admin:admin@123@192.168.20.19:554/cam/realmonitor?channel=1&subtype=1"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
assert cap.isOpened(), "Error reading video file"

# Video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Line or region points for counting
line_points = [(0, 300), (10800, 300)]

# Classes to count
classes_to_count = [0, 2, 3]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Initialize Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

# Main loop for processing frames
while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Object detection and tracking
    tracks = model.track(img, persist=True, show=False, classes=classes_to_count)

    # Apply object counting
    img = counter.start_counting(img, tracks)

    # Write annotated frame to output video
    video_writer.write(img)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# results = model.predict(source='rtsp://admin:admin@123@192.168.20.19:554/cam/realmonitor?channel=1&subtype=1', show=True, stream=True,)

# for r in results:
#         boxes = r.boxes  # Boxes object for bbox outputs
#         masks = r.masks  # Masks object for segment masks outputs
#         probs = r.probs  # Class probabilities for classification outputs









