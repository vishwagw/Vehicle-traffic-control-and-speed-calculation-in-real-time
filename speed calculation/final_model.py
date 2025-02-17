import numpy as np
import cv2
import time
from ultralytics import YOLO


# Load YOLOv8 model
model = YOLO("./models/yolov8n.pt")

# Load YOLOv8 model
model = YOLO("./models/yolov8n.pt")

def detect_vehicles(frame):
    results = model(frame)
    detected_vehicles = []
    
    for result in results:
        for box in result.boxes:
            if box.conf > 0.5 and box.cls == 2:  # class 2 is for 'car' in COCO dataset
                x, y, w, h = box.xywh[0], box.xywh[1], box.xywh[2], box.xywh[3]
                detected_vehicles.append((int(x), int(y), int(w), int(h)))
    
    return detected_vehicles

class VehicleTracker:
    def __init__(self):
        self.vehicles = {}

    def update(self, detected_vehicles, timestamp):
        speeds = []
        for (x, y, w, h) in detected_vehicles:
            center_x = x + w / 2
            center_y = y + h / 2
            vehicle_id = self._get_vehicle_id(center_x, center_y)
            if vehicle_id in self.vehicles:
                last_position, last_timestamp = self.vehicles[vehicle_id]
                distance = ((center_x - last_position[0]) ** 2 + (center_y - last_position[1]) ** 2) ** 0.5
                time_elapsed = timestamp - last_timestamp
                speed = distance / time_elapsed
                speeds.append(speed)
            self.vehicles[vehicle_id] = ((center_x, center_y), timestamp)
        return speeds

    def _get_vehicle_id(self, center_x, center_y):
        # Implement a method to assign a unique ID to each vehicle based on its position
        return f"{int(center_x)}_{int(center_y)}"

tracker = VehicleTracker()
# intialize camera or recorded video:
vid = './vids/sample_video.mp4'
cap = cv2.VideoCapture(vid)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = time.time()
    detected_vehicles = detect_vehicles(frame)
    speeds = tracker.update(detected_vehicles, timestamp)
    frame = cv2.resize(frame, (640, 360))
    for (x, y, w, h), speed in zip(detected_vehicles, speeds):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{speed:.2f} m/s", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Drone footage: Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
