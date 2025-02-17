import cv2
from ultralytics import YOLO

# load model:
model = YOLO("./models/traffic_best.pt")  # Load trained model
# video intialize:
video = './input/input1.mp4'
cap = cv2.VideoCapture(video)  # Use drone camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Perform detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box
            label = result.names[int(box.cls[0])]  # Get class name
            confidence = box.conf[0]  # Get confidence score

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Drone View", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()