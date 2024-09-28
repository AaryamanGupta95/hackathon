import random
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Open the class file and read the class names
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Video capture from the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Gesture detection logic (simple example for punching)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Simple punching detection: if index and middle fingers are below wrist
            if (index_finger_tip.y > wrist.y) and (middle_finger_tip.y > wrist.y):
                cv2.putText(frame, "Punching", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Predict on the current frame using YOLO
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, f"{class_list[int(clsID)]} {round(conf, 3) * 100}%", 
                        (int(bb[0]), int(bb[1]) - 10), font, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
