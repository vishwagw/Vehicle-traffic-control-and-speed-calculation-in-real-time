import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_vehicles(frame):
    results = model(frame)
    detected_vehicles = []
    
    for result in results:
        for box in result.boxes:
            if box.conf > 0.5 and box.cls == 2:  # class 2 is for 'car' in COCO dataset
                x, y, w, h = box.xywh[0], box.xywh[1], box.xywh[2], box.xywh[3]
                detected_vehicles.append((int(x), int(y), int(w), int(h)))
    
    return detected_vehicles

# Example usage
cap = cv2.VideoCapture("drone_footage.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    detected_vehicles = detect_vehicles(frame)
    for (x, y, w, h) in detected_vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()