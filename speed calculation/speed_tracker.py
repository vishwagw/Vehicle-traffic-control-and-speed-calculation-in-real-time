import numpy as np
import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# load model:
model = YOLO('./models/yolov8n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

def calculate_speed(prev_position, curr_position, fps, scaling_factor):
    pixel_distance = np.linalg.norm(np.array(curr_position) - np.array(prev_position))
    real_distance = pixel_distance * scaling_factor  # Convert to meters
    speed = (real_distance * fps) * 3.6  # Convert to km/h
    return speed

cap = cv2.VideoCapture("./vids/sample_video.mp4")  # Use 0 for webcam
fps = cap.get(cv2.CAP_PROP_FPS)

# Store previous positions
vehicle_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 400))
    # Run YOLOv8 inference
    results = model(frame)

    detections = []
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, score, class_id = box.cpu().numpy()
            if class_id == 2 or class_id == 3 or class_id == 5:  # Car, Truck, Bus
                detections.append(([x1, y1, x2, y2], score))

    # Track objects
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        bbox = track.to_ltwh()  # Get bounding box

        # Speed Calculation
        if track_id in vehicle_positions:
            prev_position = vehicle_positions[track_id]
            curr_position = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            speed = calculate_speed(prev_position, curr_position, fps, scaling_factor=0.05)
            vehicle_positions[track_id] = curr_position
        else:
            vehicle_positions[track_id] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            speed = 0

        # Draw BBox and Speed
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}, Speed: {speed:.2f} km/h", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection and Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


